//! Pythran gauss: solves `a x = b` by Gaussian elimination with pivoting
//! (the pivot search runs along row `i - 1` and swaps columns), in place;
//! returns `b`, which then holds the solution.
//!
//! ```python
//! def pivot(n, i, a, b):
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
//!
//! @requires(lambda n, a, b: 0 <= n <= 1024)
//! def gauss(n: Integer, a: Float[Array, '4 4'], b: Float[Array, '4']):
//!     for i in range(1, n + 1):
//!         if i < n:
//!             pivot(n, i, a, b)
//!         a[i - 1][i - 1] = 1.0 / a[i - 1][i - 1]
//!         b[i - 1] = b[i - 1] * a[i - 1][i - 1]
//!         if i < n:
//!             for j in range(i + 1, n + 1):
//!                 a[j - 1][i - 1] = a[j - 1][i - 1] * a[i - 1][i - 1]
//!             for k in range(i + 1, n + 1):
//!                 b[k - 1] = b[k - 1] - a[i - 1][k - 1] * b[i - 1]
//!                 for j in range(i + 1, n + 1):
//!                     a[j - 1][k - 1] = a[j - 1][k - 1] - a[i - 1][k - 1] * a[j - 1][i - 1]
//!     for i in range(n - 1, 0, -1):
//!         for j in range(i, n):
//!             b[i - 1] = b[i - 1] - a[j][i - 1] * b[j]
//!     return b
//! ```
//!
//! Below, indices are 0-based: the kernel's `i - 1` is `r`.
use crate::prelude::*;

/// `pivot(n, r + 1, a, b)`: the column `p >= r` with the largest `|a[r, p]|`
/// (the first one on ties) is swapped with column `r` in rows `r..n`, and
/// `b[p]` with `b[r]`.
fn pivot(n: usize, r: usize, a: &mut Array2<Q32>, b: &mut Array1<Q32>) {
    let mut best = r;
    let mut amp0 = a[(r, r)].abs();
    for p in r + 1..n {
        let amp = a[(r, p)].abs();
        if amp > amp0 {
            best = p;
            amp0 = amp;
        }
    }
    if best == r {
        return;
    }
    b.swap(r, best);
    for row in r..n {
        a.swap((row, r), (row, best));
    }
}

pub mod gauss {
    use super::*;

    /// The solution, in `b`.
    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let mut a: Array2<Q32> = args.array2();
        let mut b: Array1<Q32> = args.array1();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        let n = n as usize;
        for r in 0..n {
            if r + 1 < n {
                pivot(n, r, &mut a, &mut b);
            }
            let inverse = Q32::ONE / a[(r, r)];
            a[(r, r)] = inverse;
            b[r] = b[r] * inverse;
            if r + 1 < n {
                for j in r + 1..n {
                    a[(j, r)] = a[(j, r)] * inverse;
                }
                for k in r + 1..n {
                    let factor = a[(r, k)];
                    b[k] = b[k] - factor * b[r];
                    for j in r + 1..n {
                        a[(j, k)] = a[(j, k)] - factor * a[(j, r)];
                    }
                }
            }
        }
        // Back substitution.
        for r in (0..n.saturating_sub(1)).rev() {
            for j in r + 1..n {
                b[r] = b[r] - a[(j, r)] * b[j];
            }
        }
        b
    }
}
