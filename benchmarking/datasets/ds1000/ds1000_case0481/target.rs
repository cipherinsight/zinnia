//! DS-1000 case 0481, halo2-base baseline.
//! Kernel: `((x == a) & (y == b)).nonzero()[0]` on `x, y: Integer[9]` (data-dependent length).
//! Output: the number of matching indices, then the indices in increasing
//! order, zero-padded to 9. Compaction: `q.compact` (a triangular shift
//! network, no equality tests); the payload of item `i` is the constant `i`.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (x, y) = (inputs.ints("x", &[9])?, inputs.ints("y", &[9])?);
    let (a, b) = (inputs.int("a")?, inputs.int("b")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (x, y) = (q.load_ints(ctx, &x), q.load_ints(ctx, &y));
    let (a, b) = (q.load_int(ctx, a), q.load_int(ctx, b));
    let hit: Vec<Cell> = (0..9)
        .map(|i| {
            let x_hit = q.eq(ctx, x[i], a);
            let y_hit = q.eq(ctx, y[i], b);
            q.gate.and(ctx, x_hit, y_hit)
        })
        .collect();
    let count = q.sum(ctx, hit.iter().copied());
    let weights: Vec<Vec<Option<Cell>>> = (0..9)
        .map(|i| {
            vec![match i {
                0 => None,
                1 => Some(hit[1]),
                _ => Some(q.gate.mul(ctx, hit[i], Constant(fe(i as i128)))),
            }]
        })
        .collect();
    let indices = q.compact(ctx, &hit, &weights);
    out.dynamic(count, &indices, Kind::Int, 1);
    Ok(())
}

