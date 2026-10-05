//! DS-1000 case 0374: the empirical CDF of `grades` evaluated at the integers
//! in `eval`.
//!
//! ```python
//! @zk_chip
//! def ecdf_result(x):
//!     xs = np.sort(x)
//!     ys = np.arange(1, len(xs)+1)/float(len(xs))
//!     return xs, ys
//!
//! @zk_circuit
//! def kernel(grades: Float[Array, "50"], eval: Integer[Array, "5"]):
//!     resultx, resulty = ecdf_result(grades)
//!     result = np.zeros_like(eval, dtype=float)
//!     for i, element in enumerate(eval):
//!         if element < resultx[0]:
//!             result[i] = 0
//!         elif element >= resultx[-1]:
//!             result[i] = 1
//!         else:
//!             result[i] = resulty[(resultx > element).argmax()-1]
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 50;
const EVALS: usize = 5;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let mut xs: [Q32; N] = input.read_array();
    let eval: [i64; EVALS] = input.read_array();
    xs.sort_unstable();
    eval.iter().map(|&element| ecdf_at(&xs, Q32::from_int(element))).collect()
}

/// `resulty[k] = (k + 1) / n`, computed only where it is used.
fn ecdf_y(k: usize) -> Q32 {
    Q32::from_int(k as i64 + 1) / N as i64
}

/// The loop body: the CDF of the sorted data `xs` at `element`.
fn ecdf_at(xs: &[Q32; N], element: Q32) -> Q32 {
    if element < xs[0] {
        Q32::ZERO
    } else if element >= xs[N - 1] {
        Q32::ONE
    } else {
        // `(resultx > element).argmax()`: the first index where `xs` exceeds
        // `element`. It exists (`xs[-1] > element`) and is at least 1
        // (`xs[0] <= element`).
        let first_above = xs.iter().position(|&x| x > element).expect("xs[-1] > element");
        ecdf_y(first_above - 1)
    }
}
