//! Source CRC recurrence, carried as 64 Boolean bits. Each of 8*L steps
//! shifts the current CRC, XORs poly conditionally, and shifts cur_byte.
//! No linear-GF2 parity-history reformulation. Arithmetic right shift repeats
//! the sign bit, including for a negative private polynomial.
use crate::prelude::*;
// Sizes (hw_check.sizes): from fixture.json.
const L: usize = 16; // = data_0
const WORD: usize = 64;
const SIGN: usize = WORD - 1;
const OUT_BITS: usize = 16;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.ints("data", &[L])?;
    let poly = inputs.int("poly")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let data = q.load_ints(ctx, &data);
    let poly = q.load_int(ctx, poly);
    let p = twos_complement_bits(&q, ctx, poly);
    let mut crc: Vec<Cell> = (0..WORD).map(|i| ctx.load_constant(Fr::from((i < OUT_BITS) as u64))).collect();
    for b in data {
        let mut cur_byte = low_byte_bits(&q, ctx, b);
        for _ in 0..8 {
            let feedback = q.gate.xor(ctx, crc[0], cur_byte[0]);
            let shifted: Vec<Cell> = (0..WORD).map(|i| crc[(i + 1).min(SIGN)]).collect();
            for i in 0..WORD {
                let xored = q.gate.xor(ctx, shifted[i], p[i]);
                crc[i] = q.gate.select(ctx, xored, shifted[i], feedback);
            }
            cur_byte.remove(0);
            cur_byte.push(ctx.load_zero());
        }
    }
    // Complement, mask to 16 bits, swap bytes, mask again.
    let complemented: Vec<Cell> = crc[..OUT_BITS].iter().map(|b| q.gate.not(ctx, *b)).collect();
    let weights = (0..OUT_BITS).map(|i| Constant(Fr::from(1u64 << ((i + 8) % OUT_BITS))));
    let result = q.gate.inner_product(ctx, complemented, weights);
    out.int(result);
    Ok(())
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
