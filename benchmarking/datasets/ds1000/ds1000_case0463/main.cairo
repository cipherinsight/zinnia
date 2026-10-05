//! DS-1000 case 0463: case 0462 on a column vector: drop the zeros and the
//! repeats of the previous entry, as a `(k, 1)` column (the shape follows
//! from the count).
//!
//!     def kernel(a: Integer[Array, "20 1"]):
//!         selection = np.ones((len(a), 1), dtype = bool)
//!         selection[1:] = a[1:] != a[:-1]
//!         selection &= a != 0
//!         result = a[selection].reshape(-1, 1)
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
