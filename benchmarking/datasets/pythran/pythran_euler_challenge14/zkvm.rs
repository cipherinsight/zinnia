//! Pythran euler_challenge14 (Project Euler 14): the start below `MAX_NUM` of
//! the longest Collatz chain, with the chain lengths memoized in a dict.
//!
//! ```python
//! def next_num(num):
//!     if num & 1:
//!         return 3 * num + 1
//!     else:
//!         return num // 2
//!
//! def series_length(num, lengths):
//!     if num in lengths:
//!         return lengths[num]
//!     else:
//!         num2 = next_num(num)
//!         result = 1 + series_length(num2, lengths)
//!         lengths[num] = result
//!         return result
//!
//! def euler14(MAX_NUM: Integer):
//!     num_with_max_length = 1
//!     max_length = 0
//!     lengths = {1: 0}
//!     for ii in range(1, MAX_NUM):
//!         length = series_length(ii, lengths)
//!         if length > max_length:
//!             max_length = length
//!             num_with_max_length = ii
//!     return (num_with_max_length, max_length)
//! ```
use alloc::collections::BTreeMap;

use crate::prelude::*;

fn next_num(num: i64) -> i64 {
    if num & 1 != 0 {
        3 * num + 1
    } else {
        floor_div(num, 2)
    }
}

/// The number of steps from `num` to 1, memoized in `lengths`.
fn series_length(num: i64, lengths: &mut BTreeMap<i64, i64>) -> i64 {
    if let Some(&length) = lengths.get(&num) {
        return length;
    }
    let result = 1 + series_length(next_num(num), lengths);
    lengths.insert(num, result);
    result
}

pub mod euler14 {
    use super::*;

    pub type Output = (i64, i64);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let max_num: i64 = args.scalar();
        let mut num_with_max_length = 1;
        let mut max_length = 0;
        let mut lengths = BTreeMap::from([(1, 0)]);
        for ii in 1..max_num {
            let length = series_length(ii, &mut lengths);
            if length > max_length {
                max_length = length;
                num_with_max_length = ii;
            }
        }
        (num_with_max_length, max_length)
    }
}
