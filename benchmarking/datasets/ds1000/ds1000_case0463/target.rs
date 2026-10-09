//! Input is a 20x1 column; Boolean indexing then reshape(-1, 1) retains
//! the same row-major values. The public length counts rows of one value.
//! DS-1000 case 0463, halo2-base baseline.
//! Kernel: `selection[1:] = a[1:] != a[:-1]` (first element selected),
//! `selection &= a != 0`, `a[selection]` on `Integer[20, 1]`: drop zeros and
//! repeats of the previous value (data-dependent length).
//! Output: the count of kept values, then the kept values in order,
//! zero-padded to 20. Only equalities decide the selection (no range check).
//! Compaction by a running one-hot of the number of kept values before
//! element `i` (one select per pair, no equality tests).

use crate::prelude::*;

const N: usize = 20;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[N, 1])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let mut keep = Vec::with_capacity(N);
    for i in 0..N {
        let zero = q.gate.is_zero(ctx, a[i]);
        let nonzero = q.gate.not(ctx, zero);
        keep.push(if i == 0 {
            nonzero
        } else {
            let repeat = q.eq(ctx, a[i], a[i - 1]);
            q.gate.mul_not(ctx, repeat, nonzero)
        });
    }
    let (count, slots) = q.select_kept(ctx, &a, &keep);
    out.dynamic(count, &slots, Kind::Int, 1);
    Ok(())
}

