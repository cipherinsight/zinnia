//! NPBench weather_stencils/hdiff: horizontal diffusion (COSMO): a Laplacian,
//! flux limiting in x and y, and the update, written to `out_field`.
//!
//! ```python
//! def hdiff(in_field: Float[Array, '68 68 60'], out_field: Float[Array, '64 64 60'], coeff: Float[Array, '64 64 60']):
//!     I, J, K = (out_field.shape[0], out_field.shape[1], out_field.shape[2])
//!     lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] + in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])
//!     res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
//!     flx_field = np.where(res * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :]) > 0, 0, res)
//!     res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
//!     fly_field = np.where(res * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :]) > 0, 0, res)
//!     out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] - fly_field[:, :-1, :])
//! ```
//!
//! `in_field` has a halo of 2 around the `I x J` output columns, so
//! `lap_field[a, b]` is centred on `in_field[a + 1, b + 1]`.
use crate::prelude::*;

/// `np.where(res * grad > 0, 0, res)`: the flux, dropped where it runs up
/// the gradient.
fn limit(res: Q32, grad: Q32) -> Q32 {
    if res * grad > Q32::ZERO {
        Q32::ZERO
    } else {
        res
    }
}

pub mod hdiff {
    use super::*;

    // outputs: mutated out_field
    // unused parameters: out_field
    // (`out_field` is overwritten whole and only its shape is read, which is
    // `coeff`'s since the two combine elementwise: its values are no input.)
    pub type Output = Array3<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let input: Array3<Q32> = args.array3();
        let coeff: Array3<Q32> = args.array3();
        let (ni, nj, nk) = coeff.dim();
        let (rows, cols, depth) = input.dim();
        assert!(rows >= ni + 4 && cols >= nj + 4 && depth == nk, "operands could not be broadcast together");

        let lap = Array3::from_shape_fn((ni + 2, nj + 2, nk), |(a, b, k)| {
            let (i, j) = (a + 1, b + 1);
            input[(i, j, k)] * 4
                - (input[(i + 1, j, k)] + input[(i - 1, j, k)] + input[(i, j + 1, k)] + input[(i, j - 1, k)])
        });
        // flx[a, b] lies between lap rows a and a + 1, in output column b.
        let flx = Array3::from_shape_fn((ni + 1, nj, nk), |(a, b, k)| {
            let res = lap[(a + 1, b + 1, k)] - lap[(a, b + 1, k)];
            limit(res, input[(a + 2, b + 2, k)] - input[(a + 1, b + 2, k)])
        });
        // fly[a, b] lies between lap columns b and b + 1, in output row a.
        let fly = Array3::from_shape_fn((ni, nj + 1, nk), |(a, b, k)| {
            let res = lap[(a + 1, b + 1, k)] - lap[(a + 1, b, k)];
            limit(res, input[(a + 2, b + 2, k)] - input[(a + 2, b + 1, k)])
        });
        Array3::from_shape_fn((ni, nj, nk), |(i, j, k)| {
            let divergence = flx[(i + 1, j, k)] - flx[(i, j, k)] + fly[(i, j + 1, k)] - fly[(i, j, k)];
            input[(i + 2, j + 2, k)] - coeff[(i, j, k)] * divergence
        })
    }
}
