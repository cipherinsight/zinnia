//! Pythran fannkuch: the maximum number of pancake flips over the
//! permutations of `0..n` (Fannkuch-redux), computed and discarded:
//! `fannkuch2` returns `None`.
//!
//! ```python
//! def fannkuch(n):
//!     count = list(range(1, n + 1))
//!     max_flips = 0
//!     m = n - 1
//!     r = n
//!     check = 0
//!     perm1 = list(range(n))
//!     perm = list(range(n))
//!     while 1:
//!         if check < 30:
//!             check += 1
//!         while r != 1:
//!             count[r - 1] = r
//!             r -= 1
//!         if perm1[0] != 0 and perm1[m] != m:
//!             perm = perm1[:]
//!             flips_count = 0
//!             k = perm[0]
//!             while k:
//!                 perm[:k + 1] = perm[k::-1]
//!                 flips_count += 1
//!                 k = perm[0]
//!             if flips_count > max_flips:
//!                 max_flips = flips_count
//!         while r != n:
//!             perm1.insert(r, perm1.pop(0))
//!             count[r] -= 1
//!             if count[r] > 0:
//!                 break
//!             r += 1
//!         else:
//!             return max_flips
//!
//! @requires(lambda n: 0 <= n <= 1024)
//! def fannkuch2(n: Integer):
//!     fannkuch(n)
//! ```
//!
//! `check` is never read and is left out. `perm[:k + 1] = perm[k::-1]`
//! reverses the first `k + 1` entries; `perm1.insert(r, perm1.pop(0))`
//! rotates the first `r + 1` entries left by one.
use crate::prelude::*;

fn fannkuch(n: usize) -> usize {
    // `count[r - 1]` with `r = n = 0` indexes an empty list.
    assert!(n >= 1, "list index out of range");
    let mut count: Vec<usize> = (1..=n).collect();
    let mut max_flips = 0;
    let m = n - 1;
    let mut r = n;
    let mut perm1: Vec<usize> = (0..n).collect();
    let mut perm = perm1.clone();
    loop {
        while r != 1 {
            count[r - 1] = r;
            r -= 1;
        }
        if perm1[0] != 0 && perm1[m] != m {
            perm.copy_from_slice(&perm1);
            let mut flips_count = 0;
            let mut k = perm[0];
            while k != 0 {
                perm[..=k].reverse();
                flips_count += 1;
                k = perm[0];
            }
            max_flips = max_flips.max(flips_count);
        }
        loop {
            if r == n {
                return max_flips;
            }
            perm1[..=r].rotate_left(1);
            count[r] -= 1;
            if count[r] > 0 {
                break;
            }
            r += 1;
        }
    }
}

pub mod fannkuch2 {
    use super::*;

    /// Always `None`: `fannkuch2` discards `fannkuch(n)` and returns nothing.
    pub type Output = Option<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        fannkuch(n as usize);
        None
    }
}
