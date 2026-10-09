//! npbench_crc16_crc16__crc16 (NPBench crc16), halo2-base baseline.
//! Kernel: CRC-16-CCITT of the low bytes of `data: Int[L]` with a private
//! int64 `poly`, bit by bit: `crc` starts at 65535; for each of the
//! `STEPS = 8 L` data bits `d_t` (low bit first), `c_t = (crc & 1) ^ d_t`,
//! then `crc = (crc >> 1) ^ poly` if `c_t` else `crc >> 1`; the result is
//! `~crc & 65535` with its two bytes swapped.
//! `crc` is an int64 throughout (`>>` is arithmetic, so its bit 63 repeats),
//! and the steps are linear over GF(2) given the `c_t`: bit `i` of `crc`
//! after step `t` is bit `min(63, i + t)` of 65535 XOR the bits
//! `c_u & p[min(63, i + t - 1 - u)]` for `u < t`, `p` the bits of `poly`.
//! An XOR of bits is the parity of their integer sum, so each `c_t` is the
//! parity of `d_t + bit_t(65535) + sum_u c_u p[t - 1 - u]` (the terms with
//! index 63 summed as `p[63]` times a running count of the `c_u`), and the
//! 16 output bits likewise. Each parity is a hinted bit pinned by
//! `sum = 2 h + bit` with `h` range-checked. This costs one product per
//! (step, earlier step) pair, about 7,000, against two gates per bit and
//! step when `crc` is carried as 64 bits.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const L: usize = 16; // = data_0
const STEPS: usize = 8 * L;
// The parity sums are at most STEPS + 2 (bits and products of bits), so
// their halves fit SUM_HALF_BITS bits.
const SUM_HALF_BITS: usize = 7; // = bits(8 * data_0 + 2) - 1

/// The initial `crc`, from the kernel's text.
const INIT: u64 = 65535;
/// The bits of an int64, and the index of its sign bit.
const WORD: usize = 64;
const SIGN: usize = WORD - 1;
/// The output keeps the low 16 bits.
const OUT_BITS: usize = 16;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.ints("data", &[L])?;
    let poly = inputs.int("poly")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_ints(ctx, &data);
    let poly = q.load_int(ctx, poly);
    let d: Vec<Cell> = data.iter().flat_map(|b| low_byte_bits(&q, ctx, *b)).collect();
    let p = twos_complement_bits(&q, ctx, poly);
    let init_bit = |i: usize| (INIT >> i.min(SIGN)) & 1;

    // c[t], and count[x] = c[0] + ... + c[x - 1].
    let mut c: Vec<Cell> = Vec::with_capacity(STEPS);
    let mut count: Vec<Cell> = vec![ctx.load_zero()];
    // The parity of: bit `i + t` of INIT, `extra`, and `c_u p[i + t - 1 - u]`
    // for u < t (index capped at 63), i.e. bit i of crc after t steps
    // (XOR `extra`).
    let crc_bit = |ctx: &mut Ctx, c: &[Cell], count: &[Cell], i: usize, t: usize, extra: Option<Cell>| -> Cell {
        // Below index 63: u from i + t - 63 on.
        let first = (i + t).saturating_sub(SIGN);
        let products = q.gate.inner_product(
            ctx,
            c[first..t].iter().map(|x| Existing(*x)),
            (first..t).map(|u| Existing(p[i + t - 1 - u])),
        );
        let capped = q.gate.mul_add(ctx, p[SIGN], count[first], products);
        let mut sum = q.add(ctx, capped, Constant(Fr::from(init_bit(i + t))));
        if let Some(e) = extra {
            sum = q.add(ctx, sum, e);
        }
        parity(&q, ctx, sum)
    };
    for t in 0..STEPS {
        let bit = crc_bit(ctx, &c, &count, 0, t, Some(d[t]));
        let next = q.add(ctx, count[t], bit);
        c.push(bit);
        count.push(next);
    }
    // ~crc & 65535 with its bytes swapped: output bit k is NOT crc bit
    // (k + 8) % 16.
    let crc: Vec<Cell> = (0..OUT_BITS).map(|i| crc_bit(ctx, &c, &count, i, STEPS, None)).collect();
    let weights = (0..OUT_BITS).map(|i| Constant(Fr::from(1u64 << ((i + OUT_BITS / 2) % OUT_BITS))));
    let set = q.gate.inner_product(ctx, crc.iter().map(|b| Existing(*b)), weights);
    let result = q.sub(ctx, Constant(Fr::from((1u64 << OUT_BITS) - 1)), set);
    out.int(result);
    Ok(())
}

/// The parity of a sum of at most `STEPS + 2` bits: a hinted bit, pinned by
/// `sum = 2 h + bit` with `bit` a bit and `h` below `2^SUM_HALF_BITS` (the
/// sum is a small non-negative integer, so `(h, bit)` is its unique
/// decomposition).
fn parity(q: &Q32, ctx: &mut Ctx, sum: Cell) -> Cell {
    let v = crate::q32::signed(sum.value());
    // Hints, pinned as above.
    let bit = ctx.load_witness(fe(v & 1));
    let half = ctx.load_witness(fe(v >> 1));
    q.gate.assert_bit(ctx, bit);
    q.range.range_check(ctx, half, SUM_HALF_BITS);
    let recomposed = q.gate.mul_add(ctx, half, Constant(Fr::from(2)), bit);
    ctx.constrain_equal(&recomposed, &sum);
    bit
}

/// The eight low bits of an int64 `b` (`255 & b`, two's complement): hinted
/// bits `d_j` and `h`, pinned by `b + 2^63 = sum_j 2^j d_j + 256 h` with each
/// `d_j` a bit and `h` in `[0, 2^56)` (the decomposition of a value in
/// `[0, 2^64)` is unique; `2^63` is a multiple of 256, so the low bits are
/// `b`'s).
fn low_byte_bits(q: &Q32, ctx: &mut Ctx, b: Cell) -> Vec<Cell> {
    let offset = q.add(ctx, b, Constant(Fr::from(1u64 << 63)));
    let v = crate::q32::signed(offset.value());
    // Hints, pinned as above.
    let bits: Vec<Cell> = (0..8).map(|j| ctx.load_witness(fe((v >> j) & 1))).collect();
    let high = ctx.load_witness(fe(v >> 8));
    for bit in &bits {
        q.gate.assert_bit(ctx, *bit);
    }
    q.range.range_check(ctx, high, WORD - 8);
    let weights = (0..8).map(|j| Constant(Fr::from(1u64 << j)));
    let low = q.gate.inner_product(ctx, bits.iter().map(|x| Existing(*x)), weights);
    let recomposed = q.gate.mul_add(ctx, high, Constant(Fr::from(256)), low);
    ctx.constrain_equal(&recomposed, &offset);
    bits
}

/// The 64 two's-complement bits of an int64 `x` (low first): the bits of
/// `x + 2^63` (`num_to_bits`, unique below `2^64`) with bit 63 flipped.
fn twos_complement_bits(q: &Q32, ctx: &mut Ctx, x: Cell) -> Vec<Cell> {
    let offset = q.add(ctx, x, Constant(Fr::from(1u64 << 63)));
    let mut bits = q.gate.num_to_bits(ctx, offset, WORD);
    bits[SIGN] = q.gate.not(ctx, bits[SIGN]);
    bits
}
