//! DS-1000 case 0402, halo2-base baseline.
//! Kernel: `np.ravel_multi_index(index, dims=(8, 9, 8, 8, 8), order='F')` on a
//! private `Integer[5]` `index`; the `Float[8, 9, 8, 8, 8]` parameter `a` is
//! never read by the kernel and is not loaded. NumPy raises unless
//! `0 <= index[k] < dims[k]`, so each index is proven to lie there
//! (`assert_below`); the flat index is the inner product with the constant
//! Fortran strides (1, 8, 72, 576, 4608), which also assigns the indices.

use crate::prelude::*;

const DIMS: [u64; 5] = [8, 9, 8, 8, 8];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let index = inputs.ints("index", &[5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // Stride k is dims[0] * ... * dims[k - 1]; the leading stride 1 costs no cell.
    let strides = DIMS.iter().scan(1u64, |s, d| {
        let stride = *s;
        *s *= d;
        Some(stride)
    });
    let (flat, index) = q.gate.inner_product_left(
        ctx,
        index.iter().map(|v| Witness(fe(*v as i128))),
        strides.map(|s| Constant(Fr::from(s))),
    );
    for (x, d) in index.iter().zip(DIMS) {
        assert_below(&q, ctx, *x, d);
    }
    out.int(flat);
    Ok(())
}

/// Proves `0 <= x < d` for a constant `d` within the lookup table `[0, 2^L)`:
/// `x` and `x + 2^L - d` are both looked up. The first lookup makes `x` an
/// integer in `[0, 2^L)`, so the sum does not wrap and the second gives
/// `x < d`. One addition (4 cells) and two lookups, against the 7-cell
/// comparison of halo2-base's `check_less_than_safe`, which remains the
/// fallback for a table smaller than `d` (only in the sizing dry runs at
/// `L < 4`, which `params.rs` never picks for a circuit with lookups).
fn assert_below(q: &Q32, ctx: &mut Ctx, x: Cell, d: u64) {
    let bits = q.range.lookup_bits();
    let table = 1u64 << bits;
    if d > table {
        q.range.check_less_than_safe(ctx, x, d);
        return;
    }
    q.range.range_check(ctx, x, bits);
    if d < table {
        let shifted = q.gate.add(ctx, x, Constant(Fr::from(table - d)));
        q.range.range_check(ctx, shifted, bits);
    }
}
