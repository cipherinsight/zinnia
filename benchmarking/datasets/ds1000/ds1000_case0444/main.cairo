//! DS-1000 case 0444: the entries of `A` strictly inside `(B[0], B[1])` or
//! `(B[1], B[2])`, in order.
//!
//!     def kernel(A: Integer[Array, "20"], B: Integer[Array, "3"]):
//!         C = A[np.logical_and(A > B[0], A < B[1]) | np.logical_and(A > B[1], A < B[2])]
//!         return C
// output: [i64]
#[executable]
fn main(A: Array<i64>, B: Array<i64>) -> Array<i64> {
    assert!(A.len() == 20 && B.len() == 3, "inputs have the wrong length");
    let (b0, b1, b2) = (*B.at(0), *B.at(1), *B.at(2));
    let mut kept = array![];
    for v in A {
        if strictly_between(v, b0, b1) || strictly_between(v, b1, b2) {
            kept.append(v);
        }
    }
    kept
}

fn strictly_between(v: i64, low: i64, high: i64) -> bool {
    low < v && v < high
}
