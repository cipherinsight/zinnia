//! DS-1000 case 0409: element-wise sum of two matrices.
//!
//!     def kernel(x: Float[Array, "8 6"], y: Float[Array, "8 6"]):
//!         x_new = np.array(x)
//!         y_new = np.array(y)
//!         z = x_new + y_new
//!         return z
// output: [q32]
use crate::q32::Q32;

const N: u32 = 8 * 6;

#[executable]
fn main(x: Array<Q32>, y: Array<Q32>) -> Array<Q32> {
    assert!(x.len() == N && y.len() == N, "inputs have the wrong length");
    let mut z = array![];
    for i in 0..N {
        z.append(*x.at(i) + *y.at(i));
    }
    z
}
