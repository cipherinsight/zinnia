//! DS-1000 case 0400: an empty array, `np.array([])` (float64, shape `(0,)`).
//!
//!     def kernel():
//!         result = np.array([])
//!         return result
// output: [q32]
use crate::q32::Q32;

#[executable]
fn main() -> Array<Q32> {
    array![]
}
