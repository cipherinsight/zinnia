//! Pythran pivot: column pivoting step of a complex linear solver. Finds the
//! entry of row `i` (1-based) with the largest modulus from column `i` on,
//! and swaps that column with column `i` (rows `i..n`) and the matching
//! entries of `b`.
//!
//! ```python
//! @requires(lambda n, i, a, b: 0 <= n <= 1024)
//! def pivot(n: Integer, i: Integer, a: Complex[Array, '2 2'], b: Complex[Array, '2']):
//!     i0 = i
//!     amp0 = abs(a[i - 1][i - 1])
//!     for j in range(i + 1, n + 1):
//!         amp = abs(a[i - 1][j - 1])
//!         if amp > amp0:
//!             i0 = j
//!             amp0 = amp
//!     if i == i0:
//!         return
//!     temp = b[i - 1]
//!     b[i - 1] = b[i0 - 1]
//!     b[i0 - 1] = temp
//!     for j in range(i, n + 1):
//!         temp = a[j - 1][i - 1]
//!         a[j - 1][i - 1] = a[j - 1][i0 - 1]
//!         a[j - 1][i0 - 1] = temp
//!     return (a, b)
//! ```
use crate::prelude::*;

pub mod pivot {
    use super::*;

    /// `None` where the kernel returns without a value (no swap needed).
    pub type Output = Option<(Array2<C32>, Array1<C32>)>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let i: i64 = args.scalar();
        let mut a: Array2<C32> = args.array2();
        let mut b: Array1<C32> = args.array1();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        let (rows, cols) = a.dim();
        // Python indices (`i - 1` may be -1).
        let at = |r: i64, c: i64| (np_index(r, rows), np_index(c, cols));

        let mut i0 = i;
        let mut amp0 = a[at(i - 1, i - 1)].abs();
        for j in i + 1..=n {
            let amp = a[at(i - 1, j - 1)].abs();
            if amp > amp0 {
                (i0, amp0) = (j, amp);
            }
        }
        if i == i0 {
            return None;
        }
        b.swap(np_index(i - 1, b.len()), np_index(i0 - 1, b.len()));
        for j in i..=n {
            a.swap(at(j - 1, i - 1), at(j - 1, i0 - 1));
        }
        Some((a, b))
    }
}
