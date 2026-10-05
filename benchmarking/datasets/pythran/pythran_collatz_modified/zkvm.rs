//! Pythran collatz_modified: the smallest start whose Collatz sequence
//! reaches 1 in exactly `target` steps (no answer, so no return, for a
//! negative `target`).
//!
//! ```python
//! def collatz_modified(target: Integer):
//!     start = 1
//!     while True:
//!         i = start
//!         steps = 0
//!         while True:
//!             if i == 1:
//!                 break
//!             if i % 2 == 0:
//!                 i = i // 2
//!             else:
//!                 i = 3 * i + 1
//!             steps += 1
//!         if steps == target:
//!             return start
//!         start += 1
//! ```
use crate::prelude::*;

/// The number of Collatz steps from `i` (positive) to 1.
fn collatz_steps(mut i: i64) -> i64 {
    let mut steps = 0;
    while i != 1 {
        // i stays positive, so `%` and `/` agree with Python's.
        i = if i % 2 == 0 { i / 2 } else { 3 * i + 1 };
        steps += 1;
    }
    steps
}

pub mod collatz_modified {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let target: i64 = args.scalar();
        (1..).find(|&start| collatz_steps(start) == target).expect("the search does not end")
    }
}
