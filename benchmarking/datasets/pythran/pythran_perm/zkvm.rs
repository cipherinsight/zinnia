//! Pythran perm: all permutations of `iterable`, in `itertools.permutations`
//! order.
//!
//! ```python
//! def permutations(iterable: Integer[Array, '5']):
//!     out = []
//!     pool = tuple(iterable)
//!     n = len(pool)
//!     r = n
//!     indices = list(range(n))
//!     cycles = list(range(n - r + 1, n + 1))[::-1]
//!     out.append(tuple([pool[i] for i in indices[:r]]))
//!     while 1:
//!         for i in reversed(range(r)):
//!             cycles[i] -= 1
//!             if cycles[i] == 0:
//!                 indices[i:] = indices[i + 1:] + indices[i:i + 1]
//!                 cycles[i] = n - i
//!             else:
//!                 j = cycles[i]
//!                 indices[i], indices[-j] = (indices[-j], indices[i])
//!                 out.append(tuple([pool[i] for i in indices[:r]]))
//!                 break
//!         else:
//!             return out
//! ```
use crate::prelude::*;

pub mod permutations {
    use super::*;

    /// One row per permutation (`n!` rows), as NumPy holds the list of tuples.
    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let pool: Array1<i64> = args.array1();
        let n = pool.len();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut cycles: Vec<usize> = (1..=n).rev().collect();
        let mut out: Vec<i64> = indices.iter().map(|&i| pool[i]).collect();
        let mut rows = 1;
        'next: loop {
            for i in (0..n).rev() {
                cycles[i] -= 1;
                if cycles[i] == 0 {
                    indices[i..].rotate_left(1);
                    cycles[i] = n - i;
                } else {
                    indices.swap(i, n - cycles[i]);
                    out.extend(indices.iter().map(|&i| pool[i]));
                    rows += 1;
                    continue 'next;
                }
            }
            // `for ... else: return out`
            return Array2::from_shape_vec((rows, n), out).expect("each permutation has n entries");
        }
    }
}
