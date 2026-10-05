//! DS-1000 case 0374: the empirical CDF of `grades` evaluated at the integers
//! in `eval`.
//!
//!     @zk_chip
//!     def ecdf_result(x):
//!         xs = np.sort(x)
//!         ys = np.arange(1, len(xs)+1)/float(len(xs))
//!         return xs, ys
//!
//!     @zk_circuit
//!     def kernel(grades: Float[Array, "50"], eval: Integer[Array, "5"]):
//!         resultx, resulty = ecdf_result(grades)
//!         result = np.zeros_like(eval, dtype=float)
//!         for i, element in enumerate(eval):
//!             if element < resultx[0]:
//!                 result[i] = 0
//!             elif element >= resultx[-1]:
//!                 result[i] = 1
//!             else:
//!                 result[i] = resulty[(resultx > element).argmax()-1]
//!         return result
// output: [q32]
use crate::np::sort;
use crate::q32::{ONE, Q32, Q32Trait, ZERO};

const N: u32 = 50;
const EVALS: u32 = 5;

#[executable]
fn main(grades: Array<Q32>, eval: Array<i64>) -> Array<Q32> {
    assert!(grades.len() == N && eval.len() == EVALS, "inputs have the wrong length");
    let xs = sort(grades.span()).span();
    let mut result = array![];
    for element in eval {
        result.append(ecdf_at(xs, Q32Trait::from_int(element)));
    }
    result
}

/// `resulty[k] = (k + 1) / n`, computed only where it is used.
fn ecdf_y(k: u32) -> Q32 {
    Q32Trait::from_int((k + 1).into()).div_int(N.into())
}

/// The loop body: the CDF of the sorted data `xs` at `element`.
fn ecdf_at(xs: Span<Q32>, element: Q32) -> Q32 {
    if element < *xs.at(0) {
        return ZERO;
    }
    if element >= *xs.at(N - 1) {
        return ONE;
    }
    // `(resultx > element).argmax()`: the first index where `xs` exceeds
    // `element`. It exists (`xs[-1] > element`) and is at least 1
    // (`xs[0] <= element`).
    let mut first_above = 1;
    while *xs.at(first_above) <= element {
        first_above += 1;
    }
    ecdf_y(first_above - 1)
}
