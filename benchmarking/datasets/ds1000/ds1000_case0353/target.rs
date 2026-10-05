//! DS-1000 case 0353, halo2-base baseline.
//! Kernel: `dims = np.maximum(B.max(0), A.max(0)) + 1;
//! A[~np.in1d(np.ravel_multi_index(A.T, dims), np.ravel_multi_index(B.T, dims))]`
//! on `A: Integer[2, 3]`, `B: Integer[7, 3]`: the rows of A that are not rows
//! of B (data-dependent number of rows).
//! `ravel_multi_index` raises ValueError (no proof) for a negative coordinate
//! and when `prod(dims)` exceeds `2^63 - 1`; both are checked (each entry in
//! `[0, 2^63)`, the product in `[0, 2^63)`). The flat index of a row is then
//! `(x0 * d1 + x1) * d2 + x2`, an exact integer, injective on the rows.
//! Output: the number of kept rows, then the kept rows of A, zero-padded to
//! 2 rows of 3.

use crate::prelude::*;

const RA: usize = 2;
const RB: usize = 7;
const C: usize = 3;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b) = (inputs.ints("A", &[RA, C])?, inputs.ints("B", &[RB, C])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let b = q.load_ints(ctx, &b);
    for x in a.iter().chain(&b) {
        q.range.range_check(ctx, *x, 63);
    }
    let dims: Vec<Cell> = (0..C)
        .map(|c| {
            let column: Vec<Cell> = a.iter().chain(&b).skip(c).step_by(C).copied().collect();
            let max = q.max_of(ctx, &column);
            q.add(ctx, max, Constant(Fr::ONE))
        })
        .collect();
    let d12 = q.gate.mul(ctx, dims[1], dims[2]);
    let size = q.gate.mul(ctx, dims[0], d12);
    q.range.range_check(ctx, size, 63);

    let flat = |ctx: &mut Ctx, row: &[Cell]| {
        let hi = q.gate.mul_add(ctx, row[0], dims[1], row[1]);
        q.gate.mul_add(ctx, hi, dims[2], row[2])
    };
    let flat_a: Vec<Cell> = a.chunks(C).map(|row| flat(ctx, row)).collect();
    let flat_b: Vec<Cell> = b.chunks(C).map(|row| flat(ctx, row)).collect();
    let keep: Vec<Cell> = flat_a
        .iter()
        .map(|fa| {
            let mut acc: Option<Cell> = None;
            for fb in &flat_b {
                let same = q.eq(ctx, *fa, *fb);
                acc = Some(match acc {
                    None => q.gate.not(ctx, same),
                    Some(acc) => q.gate.mul_not(ctx, same, acc),
                });
            }
            acc.expect("B has rows")
        })
        .collect();

    // Kept rows to the front: slot 0 is A[0] if kept, else A[1] if kept;
    // slot 1 is A[1] if both are kept.
    let count = q.add(ctx, keep[0], keep[1]);
    let both = q.gate.mul(ctx, keep[0], keep[1]);
    let mut slots = Vec::with_capacity(RA * C);
    for c in 0..C {
        let second = q.gate.mul(ctx, keep[1], a[C + c]);
        slots.push(q.gate.select(ctx, a[c], second, keep[0]));
    }
    for c in 0..C {
        slots.push(q.gate.mul(ctx, both, a[C + c]));
    }
    out.dynamic(count, &slots, Kind::Int, C);
    Ok(())
}
