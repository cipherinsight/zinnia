//! `np.linalg.det` of static square matrices.
//!
//! NumPy factors the matrix (LU, in floating point). A circuit gains
//! nothing from pivoting, so the determinant is instead expanded by minors
//! along the rows (Laplace), sharing every minor between the expansions
//! that use it: the minor on rows `k..n` and a set of `n - k` columns is
//! computed once, so an `n x n` determinant costs `n * 2^(n-1)`
//! products instead of `n!`.
//!
//! The arithmetic is fixed point (Q32), like every float in Zinnia. Each
//! minor is one product sum (`IRBuilder::ir_product_sum_f`), rounded once,
//! so it is off by less than 2^-32 plus the error its sub-minors carry,
//! weighted by the entries of its row. A matrix of integers has an exact
//! determinant. A value outside the fixed-point window has no proof.

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

use super::operand::{build_array, StaticOperand};

/// The largest matrix order accepted: the expansion keeps `2^n` minors.
const MAX_ORDER: usize = 16;

/// `np.linalg.det(a)`: the determinant of each `n x n` matrix in the last
/// two axes of `a`, a float; a scalar for a single matrix.
pub fn np_linalg_det(b: &mut IRBuilder, a: &Value) -> Value {
    let a = StaticOperand::read(b, a, "np.linalg.det");
    let rank = a.shape.len();
    if rank < 2 || a.shape[rank - 1] != a.shape[rank - 2] {
        panic!("np.linalg.det: last 2 dimensions of the array must be square, got shape {:?}", a.shape);
    }
    let n = a.shape[rank - 1];
    if n > MAX_ORDER {
        panic!("np.linalg.det: matrices larger than {MAX_ORDER} x {MAX_ORDER} are not supported, got {n} x {n}");
    }
    let cells: Vec<Value> = a
        .cells
        .iter()
        .map(|cell| crate::helpers::value_ops::ensure_float(b, cell))
        .collect();
    let dets: Vec<Value> = if n == 0 {
        // An empty matrix has determinant 1 (the empty product).
        let count = a.shape[..rank - 2].iter().product();
        vec![b.ir_constant_float(1.0); count]
    } else {
        cells.chunks(n * n).map(|matrix| determinant(b, matrix, n)).collect()
    };
    let stack = a.shape[..rank - 2].to_vec();
    if stack.is_empty() {
        return dets.into_iter().next().expect("one matrix has one determinant");
    }
    build_array(b, dets, stack, NumberType::Float)
}

/// The determinant of the `n x n` matrix `m` (row-major float cells).
fn determinant(b: &mut IRBuilder, m: &[Value], n: usize) -> Value {
    // minors[columns] is the minor on the last `|columns|` rows and the
    // columns in the bit set `columns`; the full set gives the determinant.
    let zero = b.ir_constant_float(0.0);
    let mut minors: Vec<Option<Value>> = vec![None; 1 << n];
    let mut by_size: Vec<usize> = (1..1usize << n).collect();
    by_size.sort_by_key(|columns| columns.count_ones());
    for columns in by_size {
        let row = n - columns.count_ones() as usize;
        let minor = if columns.count_ones() == 1 {
            m[row * n + columns.trailing_zeros() as usize].clone()
        } else {
            // Expand along `row`: sum over its columns `j` (the `t`-th of
            // the set) of (-1)^t * m[row][j] * minor(columns without j).
            let terms: Vec<(Value, Value)> = (0..n)
                .filter(|j| columns & (1 << j) != 0)
                .enumerate()
                .map(|(t, j)| {
                    let entry = &m[row * n + j];
                    let entry = if t % 2 == 0 { entry.clone() } else { b.ir_sub_f(&zero, entry) };
                    let rest = minors[columns & !(1 << j)].clone().expect("smaller minors come first");
                    (entry, rest)
                })
                .collect();
            b.ir_product_sum_f(&terms)
        };
        minors[columns] = Some(minor);
    }
    minors[(1 << n) - 1].take().expect("the full minor is computed")
}
