//! DS-1000 case 0356: reorder `b` along the first axis by the order that sorts
//! `a` along that axis, `c[i, j, k] = b[np.argsort(a, axis=0)[i, j, k], j, k]`.
//!
//!     def kernel(a: Float[Array, "3 3 3"], b: Integer[Array, "3 3 3"]):
//!         sort_indices = np.argsort(a, axis=0)
//!         static_indices = np.indices(a.shape)
//!         c = b[sort_indices, static_indices[1], static_indices[2]]
//!         return c
// output: [i64]
use crate::np::argsort;
use crate::q32::Q32;

/// Length of the sorted first axis.
const DEPTH: u32 = 3;
/// Number of positions on the last two axes (3 x 3), flattened.
const PLANE: u32 = 3 * 3;

#[executable]
fn main(a: Array<Q32>, b: Array<i64>) -> Array<i64> {
    assert!(a.len() == DEPTH * PLANE && b.len() == DEPTH * PLANE, "inputs have the wrong length");
    let orders = argsort_axis0(a.span());
    let mut c = array![];
    for i in 0..DEPTH {
        for pos in 0..PLANE {
            let order = *orders.at(pos);
            c.append(*b.at(*order.at(i) * PLANE + pos));
        }
    }
    c
}

/// `np.argsort(a, axis=0)`: for every position of the last two axes, the
/// first-axis indices that sort `a` there (stable, ties in index order).
fn argsort_axis0(a: Span<Q32>) -> Array<Span<u32>> {
    let mut orders = array![];
    for pos in 0..PLANE {
        let mut column = array![];
        for i in 0..DEPTH {
            column.append(*a.at(i * PLANE + pos));
        }
        orders.append(argsort(column.span()).span());
    }
    orders
}
