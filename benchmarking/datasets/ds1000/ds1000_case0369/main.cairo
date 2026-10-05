//! DS-1000 case 0369: is every row of `a` close to its first entry?
//!
//!     def kernel(a: Float[Array, "3 3"]):
//!         result =np.isclose(a, a[:, 0].reshape(-1, 1), atol=0).all()
//!         return result
// output: bool
use crate::q32::{Q32, Q32Trait};

const N: u32 = 3;

/// NumPy's default `rtol = 1e-5`, quantized: `round(1e-5 * 2^32)`.
const RTOL: Q32 = Q32 { bits: 42950 };

#[executable]
fn main(a: Array<Q32>) -> bool {
    assert!(a.len() == N * N, "a has the wrong length");
    // Column 0 is trivially close to itself.
    for r in 0..N {
        let first = *a.at(r * N);
        for c in 1..N {
            if !isclose(*a.at(r * N + c), first) {
                return false;
            }
        }
    }
    true
}

/// `np.isclose(x, y, atol=0)`: `|x - y| <= rtol * |y|`.
fn isclose(x: Q32, y: Q32) -> bool {
    (x - y).abs() <= RTOL * y.abs()
}
