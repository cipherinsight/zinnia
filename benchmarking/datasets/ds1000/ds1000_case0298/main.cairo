//! DS-1000 case 0298: one-hot encoding of `a` over its sorted distinct values.
//!
//!     def kernel(a: Float[Array, "5"]):
//!         vals, idx = np.unique(a, return_inverse=True)
//!         b = np.zeros((a.size, vals.size))
//!         b[np.arange(a.size), idx] = 1
//!         return b
//!
//! The output has shape `(5, vals.size)`; its length implies the shape.
// output: [q32]
use crate::np::sort;
use crate::q32::{ONE, Q32, ZERO};

const N: u32 = 5;

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == N, "a has the wrong length");
    let vals = unique(a.span());
    let mut b = array![];
    for v in a.span() {
        let column = position(vals, *v);
        for j in 0..vals.len() {
            b.append(if j == column {
                ONE
            } else {
                ZERO
            });
        }
    }
    b
}

/// `np.unique`: the distinct values in increasing order.
fn unique(values: Span<Q32>) -> Span<Q32> {
    let sorted = sort(values);
    let mut distinct = array![*sorted.at(0)];
    for i in 1..sorted.len() {
        if *sorted.at(i) != *sorted.at(i - 1) {
            distinct.append(*sorted.at(i));
        }
    }
    distinct.span()
}

/// The index of `value` in `vals` (the `return_inverse` entry).
fn position(vals: Span<Q32>, value: Q32) -> u32 {
    let mut i = 0;
    while *vals.at(i) != value {
        i += 1;
    }
    i
}
