//! DS-1000 case 0443: the entries of `A` that occur in `B`, in order.
//!
//!     def kernel(A: Integer[Array, "20"], B: Integer[Array, "3"]):
//!         C = A[np.in1d(A,B)]
//!         return C
// output: [i64]
#[executable]
fn main(A: Array<i64>, B: Array<i64>) -> Array<i64> {
    assert!(A.len() == 20 && B.len() == 3, "inputs have the wrong length");
    let b = B.span();
    let mut kept = array![];
    for v in A {
        if contains(b, v) {
            kept.append(v);
        }
    }
    kept
}

/// `np.in1d` for one value: whether `value` occurs in `set`.
fn contains(set: Span<i64>, value: i64) -> bool {
    for v in set {
        if *v == value {
            return true;
        }
    }
    false
}
