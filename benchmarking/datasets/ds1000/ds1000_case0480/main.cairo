//! DS-1000 case 0480: the first index `i` with `(x[i], y[i]) == (a, b)`, or -1
//! if there is none.
//!
//!     def kernel(x: Integer[Array, "8"], y: Integer[Array, "8"], a: Integer, b: Integer):
//!         result = ((x == a) & (y == b)).argmax()
//!         if x[result] != a or y[result] != b:
//!             result = -1
//!         return result
//!
//! `argmax` of the match mask is the first match, or 0 when nothing matches; in
//! that case index 0 fails the check and the result is -1.
// output: i64
const N: u32 = 8;

#[executable]
fn main(x: Array<i64>, y: Array<i64>, a: i64, b: i64) -> i64 {
    assert!(x.len() == N && y.len() == N, "inputs have the wrong length");
    match first_match(x.span(), y.span(), a, b) {
        Some(i) => i.into(),
        None => -1,
    }
}

fn first_match(x: Span<i64>, y: Span<i64>, a: i64, b: i64) -> Option<u32> {
    for i in 0..N {
        if *x.at(i) == a && *y.at(i) == b {
            return Some(i);
        }
    }
    None
}
