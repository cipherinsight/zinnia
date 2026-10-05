//! DS-1000 case 0441: whether `number` occurs in `a` (Python `in` on an array
//! is `(a == number).any()`).
//!
//!     def kernel(a: Integer[Array, "4"], number: Integer):
//!         is_contained = number in a
//!         return is_contained
// output: bool
#[executable]
fn main(a: Array<i64>, number: i64) -> bool {
    assert!(a.len() == 4, "a has the wrong length");
    for v in a {
        if v == number {
            return true;
        }
    }
    false
}
