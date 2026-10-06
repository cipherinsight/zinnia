//! Pythran nd_local_maxima: marks the entries of a 4-D array that are at
//! least every neighbour in a window as large as the array, with
//! wrap-around indexing (the default `mode`).
//!
//! ```python
//! def wrap(pos, offset, bound):
//!     return (pos + offset) % bound
//!
//! def local_maxima(data: Float[Array, '8 6 4 2'], mode=wrap):
//!     wsize = data.shape
//!     result = np.ones(data.shape, bool)
//!     for pos in np.ndindex(data.shape):
//!         myval = data[pos]
//!         for offset in np.ndindex(wsize):
//!             neighbor_idx = tuple((mode(p, o - w // 2, w) for p, o, w in zip(pos, offset, wsize)))
//!             result[pos] &= data[neighbor_idx] <= myval
//!     return result
//! ```
//!
//! `mode` is not a kernel parameter (the signature takes `data` only), so it
//! is `wrap`. The window spans the whole array, so each entry is compared
//! with every entry.
use crate::prelude::*;
use ndarray::indices;

/// The kernel's `wrap`: `(pos + offset) % bound` with Python's `%`.
fn wrap(pos: usize, offset: i64, bound: usize) -> usize {
    floor_mod(pos as i64 + offset, bound as i64) as usize
}

pub mod local_maxima {
    use super::*;

    pub type Output = Array4<bool>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let data: Array4<Q32> = args.array4();
        let wsize = data.dim();
        let (w0, w1, w2, w3) = wsize;
        let neighbour = |p: usize, o: usize, w: usize| wrap(p, o as i64 - (w / 2) as i64, w);
        let mut result = Array4::from_elem(wsize, true);
        for ((p0, p1, p2, p3), is_max) in result.indexed_iter_mut() {
            let myval = data[(p0, p1, p2, p3)];
            for (o0, o1, o2, o3) in indices(wsize) {
                let idx = (
                    neighbour(p0, o0, w0),
                    neighbour(p1, o1, w1),
                    neighbour(p2, o2, w2),
                    neighbour(p3, o3, w3),
                );
                *is_max &= data[idx] <= myval;
            }
        }
        result
    }
}
