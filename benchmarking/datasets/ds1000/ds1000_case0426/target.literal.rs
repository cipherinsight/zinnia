//! DS-1000 case 0426: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Bitwise AND is independently decomposed per broadcast element; materialize masked integers, then compare and cast.
use crate::prelude::*;

const M: usize = 15;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.ints("a", &[20])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &av);
    // arange, left shift and reversal are compile-time constants.
    let mut masked = Vec::new();
    for i in 0..20 {
        for j in 0..M {
            // One independent scalar bitwise AND: two's-complement low bits.
            let (_, low) = q.floor_div_const(ctx, a[i], 1u64 << M);
            let bits = q.gate.num_to_bits(ctx, low, M);
            masked.push(q.gate.mul(ctx, bits[M - 1 - j], Constant(fe(1i128 << (M - 1 - j)))));
        }
    }
    let mut positive = Vec::new();
    for value in masked {
        let vb = q.bound(ctx, value);
        let zero = q.int_const(ctx, 0);
        let zb = q.bound(ctx, zero);
        positive.push(q.gt(ctx, vb, zb));
    }
    // Boolean-to-integer cast keeps the 0/1 encoding.
    out.ints(&positive);
    Ok(())
}
