//! DS-1000 case 0403, halo2-base LITERAL baseline (LITERAL_BASELINES.md).
//! Literal translation of kernel.py:
//!   dims = np.array([8, 9, 8, 8, 8])
//!   result = np.ravel_multi_index(index, dims=dims, order='C')
//! `np.ravel_multi_index` has no source to copy (L5): the textbook definition
//! is the defining sum `flat = sum_k index[k] * stride[k]` with the C-order
//! strides (4608, 512, 64, 8, 1), accumulated term by term in index order
//! k = 0..5 (not reversed, unlike the optimized baseline, which reorders to
//! save a cell on the leading stride-1 term: L3 forbids that reordering).
//! NumPy raises unless `0 <= index[k] < dims[k]`; this is also exactly the
//! input-domain check of CONVENTIONS.md 1.6 for an int64 (a check the
//! kernel's own semantics already impose and that implies the domain
//! replaces the generic one, as 1.6 says by name for this case). Each
//! index[k] gets exactly one such check, via halo2-base's own
//! `check_less_than_safe`, the plain (non-cell-optimized) "0 <= a < b"
//! gadget, instead of the optimized baseline's hand-rolled `assert_below`
//! lookup-table trick.
//! `a` (the `Float[8,9,8,8,8]` parameter) is never read by the kernel, so it
//! is not loaded and gets no check (CONVENTIONS.md 1.6, L7 plumbing).

use crate::prelude::*;

const DIMS: [u64; 5] = [8, 9, 8, 8, 8];
const STRIDES: [u64; 5] = [4608, 512, 64, 8, 1];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let index = inputs.ints("index", &[5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // flat = sum_k index[k] * stride[k], each term added as it is formed,
    // in source (index) order; 0 <= index[k] < dims[k] checked at the same
    // point, once per element.
    let mut flat = q.int_const(ctx, 0);
    for k in 0..5 {
        let x = ctx.load_witness(fe(index[k] as i128));
        q.range.check_less_than_safe(ctx, x, DIMS[k]);
        let term = q.gate.mul(ctx, x, Constant(Fr::from(STRIDES[k])));
        flat = q.gate.add(ctx, flat, term);
    }
    out.int(flat);
    Ok(())
}
