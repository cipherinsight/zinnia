//! Pythran insertion_sort: insertion sort of a float list, in place.
//!
//! ```python
//! def insertion_sort(list2: Float[Array, '10']):
//!     for i in range(1, len(list2)):
//!         save = list2[i]
//!         j = i
//!         while j > 0 and list2[j - 1] > save:
//!             list2[j] = list2[j - 1]
//!             j -= 1
//!         list2[j] = save
//! ```
use crate::prelude::*;

pub mod insertion_sort {
    use super::*;

    // outputs: mutated list2
    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut list: Array1<Q32> = args.array1();
        for i in 1..list.len() {
            let save = list[i];
            let mut j = i;
            while j > 0 && list[j - 1] > save {
                list[j] = list[j - 1];
                j -= 1;
            }
            list[j] = save;
        }
        list
    }
}
