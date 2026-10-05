//! DS-1000 case 0462: drop the zeros and the repeats of the previous entry
//! (keep the first entry of each run of equal values, unless it is zero).
//!
//!     def kernel(a: Integer[Array, "20"]):
//!         selection = np.ones(len(a), dtype = bool)
//!         selection[1:] = a[1:] != a[:-1]
//!         selection &= a != 0
//!         result = a[selection]
//!         return result
// output: [i64]
const N: u32 = 20;

#[executable]
fn main(a: Array<i64>) -> Array<i64> {
    assert!(a.len() == N, "a has the wrong length");
    let mut kept = array![];
    for i in 0..N {
        let v = *a.at(i);
        if v != 0 && (i == 0 || v != *a.at(i - 1)) {
            kept.append(v);
        }
    }
    kept
}
