//! Diagonals of static arrays: `np.diagonal` (and through it `np.trace`)
//! and the lower-triangular constructor `np.tri`.

use crate::builder::IRBuilder;
use crate::helpers::dtype::Dtype;
use crate::helpers::shape_arith::{resolve_axis, row_major_strides};
use crate::types::{NumberType, Value};

use super::operand::StaticOperand;

/// `np.diagonal(a, offset, axis1, axis2)`: the diagonal of the 2-D
/// sub-arrays spanned by `axis1` and `axis2`. The result has the remaining
/// axes of `a` in order, then one axis along the diagonal: element
/// `(..., i)` is `a[..., i, ..., i + offset, ...]` (row `i - offset`,
/// column `i` for a negative offset). The cells are copied; the dtype is
/// `a`'s.
pub fn np_diagonal(b: &mut IRBuilder, a: &Value, offset: i64, axis1: i64, axis2: i64) -> Value {
    let a = StaticOperand::read(b, a, "np.diagonal");
    let rank = a.shape.len();
    if rank < 2 {
        panic!("np.diagonal: the array must have at least two dimensions, got {rank}");
    }
    let axis1 = resolve_axis(axis1, rank, "np.diagonal");
    let axis2 = resolve_axis(axis2, rank, "np.diagonal");
    if axis1 == axis2 {
        panic!("np.diagonal: axis1 and axis2 cannot be the same");
    }
    // The diagonal starts at (row_start, col_start) of each 2-D sub-array.
    let (row_start, col_start) = if offset >= 0 { (0, offset) } else { (-offset, 0) };
    let length = (a.shape[axis1] as i64 - row_start)
        .min(a.shape[axis2] as i64 - col_start)
        .max(0) as usize;

    let strides = row_major_strides(&a.shape);
    let rest: Vec<usize> = (0..rank).filter(|&d| d != axis1 && d != axis2).collect();
    let rest_shape: Vec<usize> = rest.iter().map(|&d| a.shape[d]).collect();
    let rest_count: usize = rest_shape.iter().product();
    let rest_strides = row_major_strides(&rest_shape);

    let mut cells = Vec::with_capacity(rest_count * length);
    for outer in 0..rest_count {
        let base: usize = rest
            .iter()
            .enumerate()
            .map(|(k, &d)| (outer / rest_strides[k]) % rest_shape[k] * strides[d])
            .sum();
        for i in 0..length {
            let row = row_start as usize + i;
            let col = col_start as usize + i;
            cells.push(a.cells[base + row * strides[axis1] + col * strides[axis2]].clone());
        }
    }
    let mut shape = rest_shape;
    shape.push(length);
    crate::helpers::static_array::base::build_static_array_from_flat(b, cells, shape, a.dtype)
}

/// `np.tri(N, M=None, k=0, dtype=float)`: the `N x M` array with ones at
/// and below the `k`-th diagonal (`j <= i + k`) and zeros elsewhere.
pub fn np_tri(b: &mut IRBuilder, n: usize, m: Option<usize>, k: i64, dtype: Option<Dtype>) -> Value {
    let m = m.unwrap_or(n);
    let one = b.ir_constant_int(1);
    let zero = b.ir_constant_int(0);
    let cells = (0..n)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| if (j as i64) <= i as i64 + k { one.clone() } else { zero.clone() })
        .collect();
    let ones = crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        cells,
        vec![n, m],
        NumberType::Integer,
    );
    crate::helpers::dtype::convert(b, &ones, dtype.unwrap_or(Dtype::Float))
}
