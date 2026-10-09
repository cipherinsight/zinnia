//! npbench_crc16_crc16__crc16: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Bit arrays implement signed shifts, XOR and masks directly; standard bit decomposition only.
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

    // crc = 65535: the low 16 bits set, the rest clear.
    let mut crc: Vec<Cell> = (0..WORD)
        .map(|i| ctx.load_constant(Fr::from((i < OUT_BITS) as u64)))
        .collect();
    let p = twos_complement_bits(&q, ctx, poly);

    for b in data {
        // cur_byte = 255 & b.
        let mut cur_byte = low_byte_bits(&q, ctx, b);
        for _ in 0..8 {
            // if crc & 1 ^ cur_byte & 1: crc = crc >> 1 ^ poly else: crc >>= 1
            let feedback = q.gate.xor(ctx, crc[0], cur_byte[0]);
            let shifted: Vec<Cell> = (0..WORD).map(|i| crc[(i + 1).min(SIGN)]).collect();
            let mut next_crc = Vec::with_capacity(WORD);
            for i in 0..WORD {
                let xored = q.gate.xor(ctx, shifted[i], p[i]);
                next_crc.push(q.gate.select(ctx, xored, shifted[i], feedback));
            }
            crc = next_crc;
            // cur_byte >>= 1
            cur_byte.remove(0);
            cur_byte.push(ctx.load_zero());
        }
    }

    // crc = ~crc & 65535
    let mut masked = vec![ctx.load_zero(); WORD];
    for i in 0..WORD {
        let complemented = q.gate.not(ctx, crc[i]);
        masked[i] = if i < OUT_BITS {
            complemented
        } else {
            ctx.load_zero()
        };
    }
    // crc = crc << 8 | crc >> 8 & 255
    let mut combined = vec![ctx.load_zero(); WORD];
    for i in 0..WORD {
        let left = if i >= 8 {
            masked[i - 8]
        } else {
            ctx.load_zero()
        };
        let right = if i < 8 {
            masked[i + 8]
        } else {
            ctx.load_zero()
        };
        combined[i] = q.gate.or(ctx, left, right);
    }
    // return crc & 65535
    let result = q.gate.inner_product(
        ctx,
        combined[..OUT_BITS].iter().copied(),
        (0..OUT_BITS).map(|i| Constant(Fr::from(1u64 << i))),
    );
    out.int(result);
    Ok(())
}

fn low_byte_bits(q: &Q32, ctx: &mut Ctx, b: Cell) -> Vec<Cell> {
    twos_complement_bits(q, ctx, b)[..8].to_vec()
}

/// The 64 two's-complement bits of an int64 `x` (low first): the bits of
/// `x + 2^63` (`num_to_bits`, unique below `2^64`) with bit 63 flipped.
fn twos_complement_bits(q: &Q32, ctx: &mut Ctx, x: Cell) -> Vec<Cell> {
    let offset = q.add(ctx, x, Constant(Fr::from(1u64 << 63)));
    let mut bits = q.gate.num_to_bits(ctx, offset, WORD);
    bits[SIGN] = q.gate.not(ctx, bits[SIGN]);
    bits
}
