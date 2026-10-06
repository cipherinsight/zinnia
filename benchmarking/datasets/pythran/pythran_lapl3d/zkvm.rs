//! Pythran lapl3d: magnitude of a five-point Laplacian over the first two
//! axes of a 3-D image, normalized by its maximum (at least 1).
//!
//! ```python
//! def laplacien(image: Float[Array, '8 8 8']):
//!     out_image = np.abs(4 * image[1:-1, 1:-1] - image[0:-2, 1:-1] - image[2:, 1:-1] - image[1:-1, 0:-2] - image[1:-1, 2:])
//!     valmax = np.max(out_image)
//!     valmax = max(1.0, valmax) + 1e-09
//!     out_image /= valmax
//!     return out_image
//! ```
use crate::prelude::*;

/// `round(1e-9 * 2^32)`.
const TINY: Q32 = Q32::from_bits(4);

pub mod laplacien {
    use super::*;

    pub type Output = Array3<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let image: Array3<Q32> = args.array3();
        let (rows, cols, depth) = image.dim();
        let (rows, cols) = (rows.saturating_sub(2), cols.saturating_sub(2));
        let mut out = Array3::from_shape_fn((rows, cols, depth), |(i, j, k)| {
            let (i, j) = (i + 1, j + 1);
            (image[(i, j, k)] * 4
                - image[(i - 1, j, k)]
                - image[(i + 1, j, k)]
                - image[(i, j - 1, k)]
                - image[(i, j + 1, k)])
                .abs()
        });
        let valmax = *out
            .iter()
            .max()
            .expect("zero-size array to reduction operation maximum");
        let valmax = Q32::ONE.max(valmax) + TINY;
        out.mapv_inplace(|v| v / valmax);
        out
    }
}
