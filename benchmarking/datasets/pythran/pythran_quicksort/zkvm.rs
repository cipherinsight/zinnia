//! Pythran quicksort: in-place quicksort of `l[s..=e]` with the last element
//! as pivot (Hoare-style hole filling).
//!
//! ```python
//! def partition(list, start, end):
//!     pivot = list[end]
//!     bottom = start - 1
//!     top = end
//!     done = 0
//!     while not done:
//!         while not done:
//!             bottom = bottom + 1
//!             if bottom == top:
//!                 done = 1
//!                 break
//!             if list[bottom] > pivot:
//!                 list[top] = list[bottom]
//!                 break
//!         while not done:
//!             top = top - 1
//!             if top == bottom:
//!                 done = 1
//!                 break
//!             if list[top] < pivot:
//!                 list[bottom] = list[top]
//!                 break
//!     list[top] = pivot
//!     return top
//!
//! def do_quicksort(list, start, end):
//!     if start < end:
//!         split = partition(list, start, end)
//!         do_quicksort(list, start, split - 1)
//!         do_quicksort(list, split + 1, end)
//!
//! def quicksort(l: Integer[Array, '10'], s: Integer, e: Integer):
//!     do_quicksort(l, s, e)
//! ```
//!
//! Indices are those of a NumPy array: `s` and `e` may be negative and then
//! count from the end.
use crate::prelude::*;

fn partition(list: &mut Array1<i64>, start: i64, end: i64) -> i64 {
    let n = list.len();
    let at = |i: i64| np_index(i, n);
    let pivot = list[at(end)];
    let mut bottom = start - 1;
    let mut top = end;
    let mut done = false;
    while !done {
        while !done {
            bottom += 1;
            if bottom == top {
                done = true;
                break;
            }
            if list[at(bottom)] > pivot {
                list[at(top)] = list[at(bottom)];
                break;
            }
        }
        while !done {
            top -= 1;
            if top == bottom {
                done = true;
                break;
            }
            if list[at(top)] < pivot {
                list[at(bottom)] = list[at(top)];
                break;
            }
        }
    }
    list[at(top)] = pivot;
    top
}

fn do_quicksort(list: &mut Array1<i64>, start: i64, end: i64) {
    if start < end {
        let split = partition(list, start, end);
        do_quicksort(list, start, split - 1);
        do_quicksort(list, split + 1, end);
    }
}

pub mod quicksort {
    use super::*;

    // outputs: mutated l
    pub type Output = Array1<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut l: Array1<i64> = args.array1();
        let s: i64 = args.scalar();
        let e: i64 = args.scalar();
        do_quicksort(&mut l, s, e);
        l
    }
}
