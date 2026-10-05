//! Pythran another_quicksort: recursive quicksort that partitions around the
//! middle element.
//!
//! ```python
//! def swap(l, idx1, idx2):
//!     if idx1 != idx2:
//!         tmp = l[idx1]
//!         l[idx1] = l[idx2]
//!         l[idx2] = tmp
//!
//! def partition(l):
//!     size = len(l)
//!     pivot_idx = size // 2
//!     val = l[pivot_idx]
//!     idx = size - 1
//!     if pivot_idx != idx:
//!         swap(l, pivot_idx, idx)
//!     idx = idx - 1
//!     i = 0
//!     while i <= idx:
//!         if l[i] > val:
//!             while l[idx] > val and idx > i:
//!                 idx = idx - 1
//!             if idx != i:
//!                 swap(l, i, idx)
//!                 idx = idx - 1
//!             else:
//!                 break
//!         i = i + 1
//!     assert idx == i or idx + 1 == i
//!     swap(l, i, size - 1)
//!     return i
//!
//! def QuickSort(l: Integer[Array, '10']):
//!     size = len(l)
//!     if size > 1:
//!         idx = partition(l)
//!         l1 = []
//!         l2 = []
//!         for i in range(0, idx):
//!             l1.append(l[i])
//!         for i in range(idx, size):
//!             l2.append(l[i])
//!         QuickSort(l1)
//!         QuickSort(l2)
//!         for i in range(0, len(l1)):
//!             l[i] = l1[i]
//!         for i in range(0, len(l2)):
//!             l[len(l1) + i] = l2[i]
//!     return l
//! ```
//!
//! Sorting the copies `l1 = l[:idx]` and `l2 = l[idx:]` and writing them
//! back is sorting the two parts of `l` in place.
use crate::prelude::*;

/// `partition`: moves the elements greater than the middle element `val`
/// after the others, puts `val` at the boundary and returns its index.
fn partition(l: &mut [i64]) -> usize {
    let size = l.len();
    let pivot_idx = size / 2;
    let val = l[pivot_idx];
    let mut idx = size - 1;
    l.swap(pivot_idx, idx);
    // `idx` never drops below 0: it starts at `size - 2 >= 0` and is only
    // decremented while above `i >= 0`.
    idx -= 1;
    let mut i = 0;
    while i <= idx {
        if l[i] > val {
            while l[idx] > val && idx > i {
                idx -= 1;
            }
            if idx != i {
                l.swap(i, idx);
                idx -= 1;
            } else {
                break;
            }
        }
        i += 1;
    }
    assert!(idx == i || idx + 1 == i, "assert idx == i or idx + 1 == i");
    l.swap(i, size - 1);
    i
}

fn quick_sort(l: &mut [i64]) {
    if l.len() > 1 {
        let idx = partition(l);
        let (l1, l2) = l.split_at_mut(idx);
        quick_sort(l1);
        quick_sort(l2);
    }
}

#[allow(non_snake_case)]
pub mod QuickSort {
    use super::*;

    pub type Output = Array1<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut l: Array1<i64> = args.array1();
        quick_sort(l.as_slice_mut().expect("standard layout"));
        l
    }
}
