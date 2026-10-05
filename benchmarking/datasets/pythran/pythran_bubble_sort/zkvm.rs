//! Pythran bubble_sort: bubble sort of a copy of an integer list.
//!
//! ```python
//! def bubble_sort(list0: Integer[Array, '11']):
//!     list1 = [x for x in list0]
//!     for i in range(0, len(list1) - 1):
//!         swap_test = False
//!         for j in range(0, len(list1) - i - 1):
//!             if list1[j] > list1[j + 1]:
//!                 list1[j], list1[j + 1] = (list1[j + 1], list1[j])
//!             swap_test = True
//!         if swap_test == False:
//!             break
//!     return list1
//! ```
//!
//! `swap_test = True` sits outside the `if`, so it is set whenever the inner
//! loop runs, which it always does (`i < len - 1`): the `break` is never
//! taken and every pass runs.
use crate::prelude::*;

pub mod bubble_sort {
    use super::*;

    pub type Output = Array1<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut list: Array1<i64> = args.array1();
        let n = list.len();
        for i in 0..n.saturating_sub(1) {
            for j in 0..n - i - 1 {
                if list[j] > list[j + 1] {
                    list.swap(j, j + 1);
                }
            }
        }
        list
    }
}
