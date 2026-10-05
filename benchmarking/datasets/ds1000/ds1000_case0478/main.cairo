//! DS-1000 case 0478: `a` with everything off the diagonal set to zero,
//! written through the diagonal view `np.einsum('ii->i', a)`.
//!
//!     def kernel(a: Integer[Array, "4 4"]):
//!         result = np.einsum('ii->i', a)
//!         save = result.copy()
//!         a[...] = 0
//!         result[...] = save
//!         return a
// output: [i64]

const N: u32 = 4;

/// `np.einsum('ii->i', a)` is a writeable view of the diagonal of `a`:
/// `a[...] = 0` clears it with the rest of `a`, and `result[...] = save`
/// writes the saved diagonal back into `a`, which the kernel returns.
#[executable]
fn main(a: Array<i64>) -> Array<i64> {
    assert!(a.len() == N * N, "a has the wrong length");
    let mut result = array![];
    for i in 0..N {
        for j in 0..N {
            result.append(if i == j {
                *a.at(i * N + i)
            } else {
                0
            });
        }
    }
    result
}
