//! Products of static arrays: `np.kron`, `np.vdot` and `np.cross`.

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::row_major_strides;
use crate::types::Value;

use super::operand::{broadcast_source_index, build_array, cell_op, promoted, zero, StaticOperand};

/// `np.kron(a, b)`: the Kronecker product. The operand of lower rank gets
/// leading axes of length one; along every axis `d` the result has length
/// `a.shape[d] * b.shape[d]`, and the cell at `i * b.shape[d] + j` (per
/// axis) is `a[i] * b[j]`.
pub fn np_kron(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Value {
    let lhs = StaticOperand::read_numeric(b, lhs, "np.kron");
    let rhs = StaticOperand::read_numeric(b, rhs, "np.kron");
    let rank = lhs.shape.len().max(rhs.shape.len());
    let padded = |shape: &[usize]| -> Vec<usize> {
        let mut out = vec![1; rank - shape.len()];
        out.extend_from_slice(shape);
        out
    };
    let (lhs_shape, rhs_shape) = (padded(&lhs.shape), padded(&rhs.shape));
    let shape: Vec<usize> = lhs_shape.iter().zip(&rhs_shape).map(|(m, n)| m * n).collect();
    let (out_strides, lhs_strides, rhs_strides) =
        (row_major_strides(&shape), row_major_strides(&lhs_shape), row_major_strides(&rhs_shape));

    let total: usize = shape.iter().product();
    let mut cells = Vec::with_capacity(total);
    for index in 0..total {
        let (mut l, mut r) = (0, 0);
        for d in 0..rank {
            let coordinate = index / out_strides[d] % shape[d];
            l += coordinate / rhs_shape[d] * lhs_strides[d];
            r += coordinate % rhs_shape[d] * rhs_strides[d];
        }
        cells.push(cell_op(b, "mul", &lhs.cells[l], &rhs.cells[r]));
    }
    build_array(b, cells, shape, promoted(lhs.dtype, rhs.dtype))
}

/// `np.vdot(a, b)`: the dot product of the two arrays flattened (they must
/// have the same number of elements), as `np.matmul` of two 1-D arrays.
pub fn np_vdot(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Value {
    let lhs = StaticOperand::read_numeric(b, lhs, "np.vdot");
    let rhs = StaticOperand::read_numeric(b, rhs, "np.vdot");
    if lhs.cells.len() != rhs.cells.len() {
        panic!(
            "np.vdot: the arrays must have the same number of elements, got {} and {}",
            lhs.cells.len(),
            rhs.cells.len()
        );
    }
    let lhs_flat = build_array(b, lhs.cells, vec![lhs.shape.iter().product()], lhs.dtype);
    let rhs_flat = build_array(b, rhs.cells, vec![rhs.shape.iter().product()], rhs.dtype);
    crate::helpers::value_ops::apply_binary_op(b, "mat_mul", &lhs_flat, &rhs_flat)
}

/// `np.cross(a, b)`: the cross product of the vectors along the last axis
/// of `a` and of `b` (2 or 3 components each), the other axes broadcast.
/// Two 3-vectors give a 3-vector; two 2-vectors give the z component of
/// their product, a scalar per vector pair; a 2-vector with a 3-vector is
/// taken with a zero z component and gives a 3-vector. NumPy 2.5 rejects
/// 2-vectors; Zinnia keeps NumPy 2.2's results, see
/// "Removed NumPy Names" in docs-site/content/programming-guide/numpy-api.mdx.
pub fn np_cross(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Value {
    let lhs = StaticOperand::read_numeric(b, lhs, "np.cross");
    let rhs = StaticOperand::read_numeric(b, rhs, "np.cross");
    let components = |operand: &StaticOperand| match operand.shape.last() {
        Some(&n @ (2 | 3)) => n,
        _ => panic!(
            "np.cross: incompatible dimensions for cross product (dimension must be 2 or 3), got shape {:?}",
            operand.shape
        ),
    };
    let (m, n) = (components(&lhs), components(&rhs));
    let lhs_vectors = &lhs.shape[..lhs.shape.len() - 1];
    let rhs_vectors = &rhs.shape[..rhs.shape.len() - 1];
    let vectors = crate::helpers::shape_arith::broadcast_shapes(lhs_vectors, rhs_vectors)
        .unwrap_or_else(|| {
            panic!("np.cross: shapes {:?} and {:?} are not broadcast compatible", lhs.shape, rhs.shape)
        });
    let dtype = promoted(lhs.dtype, rhs.dtype);
    let count: usize = vectors.iter().product();
    let out_components = if m == 2 && n == 2 { 1 } else { 3 };

    let mut cells = Vec::with_capacity(count * out_components);
    for v in 0..count {
        let x = &lhs.cells[broadcast_source_index(v, &vectors, lhs_vectors) * m..][..m];
        let y = &rhs.cells[broadcast_source_index(v, &vectors, rhs_vectors) * n..][..n];
        // `x[i] * y[j] - x[j] * y[i]`, a missing third component read as 0.
        let mut term = |i: usize, j: usize| -> Value {
            let first = (i < m && j < n).then(|| cell_op(b, "mul", &x[i], &y[j]));
            let second = (j < m && i < n).then(|| cell_op(b, "mul", &x[j], &y[i]));
            match (first, second) {
                (Some(p), Some(q)) => cell_op(b, "sub", &p, &q),
                (Some(p), None) => p,
                (None, Some(q)) => {
                    let zero = zero(b, dtype);
                    cell_op(b, "sub", &zero, &q)
                }
                (None, None) => unreachable!("a cross-product component has a defined term"),
            }
        };
        if out_components == 1 {
            cells.push(term(0, 1));
        } else {
            cells.extend([term(1, 2), term(2, 0), term(0, 1)]);
        }
    }
    let mut shape = vectors;
    if out_components == 3 {
        shape.push(3);
    }
    build_array(b, cells, shape, dtype)
}
