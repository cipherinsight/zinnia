//! NPBench weather_stencils/vadv: implicit vertical advection, one tridiagonal
//! system along k per column (i, j), solved by the Thomas algorithm; the
//! result is written into `utens_stage`.
//!
//! ```python
//! BET_M = 0.5
//! BET_P = 0.5
//!
//! def vadv(utens_stage: Float[Array, '60 60 40'], u_stage: Float[Array, '60 60 40'], wcon: Float[Array, '61 60 40'], u_pos: Float[Array, '60 60 40'], utens: Float[Array, '60 60 40'], dtr_stage: Float):
//!     I, J, K = (utens_stage.shape[0], utens_stage.shape[1], utens_stage.shape[2])
//!     ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
//!     dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
//!     data_col = np.ndarray((I, J), dtype=utens_stage.dtype)
//!     for k in range(1):
//!         gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
//!         cs = gcv * BET_M
//!         ccol[:, :, k] = gcv * BET_P
//!         bcol = dtr_stage - ccol[:, :, k]
//!         correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
//!         dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
//!         divided = 1.0 / bcol
//!         ccol[:, :, k] = ccol[:, :, k] * divided
//!         dcol[:, :, k] = dcol[:, :, k] * divided
//!     for k in range(1, K - 1):
//!         gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
//!         gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
//!         as_ = gav * BET_M
//!         cs = gcv * BET_M
//!         acol = gav * BET_P
//!         ccol[:, :, k] = gcv * BET_P
//!         bcol = dtr_stage - acol - ccol[:, :, k]
//!         correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
//!         dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
//!         divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
//!         ccol[:, :, k] = ccol[:, :, k] * divided
//!         dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
//!     for k in range(K - 1, K):
//!         gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
//!         as_ = gav * BET_M
//!         acol = gav * BET_P
//!         bcol = dtr_stage - acol
//!         correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
//!         dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
//!         divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
//!         dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
//!     for k in range(K - 1, K - 2, -1):
//!         datacol = dcol[:, :, k]
//!         data_col[:] = datacol
//!         utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
//!     for k in range(K - 2, -1, -1):
//!         datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
//!         data_col[:] = datacol
//!         utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
//! ```
use crate::prelude::*;

const BET_M: Q32 = Q32::from_bits(1 << 31); // 0.5
const BET_P: Q32 = Q32::from_bits(1 << 31); // 0.5
const QUARTER: Q32 = Q32::from_bits(1 << 30); // 0.25

pub mod vadv {
    use super::*;

    // outputs: mutated utens_stage
    pub type Output = Array3<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut utens_stage: Array3<Q32> = args.array3();
        let u_stage: Array3<Q32> = args.array3();
        let wcon: Array3<Q32> = args.array3();
        let u_pos: Array3<Q32> = args.array3();
        let utens: Array3<Q32> = args.array3();
        let dtr_stage: Q32 = args.scalar();
        let (ni, nj, nk) = utens_stage.dim();
        for other in [u_stage.dim(), u_pos.dim(), utens.dim()] {
            assert_eq!(
                other,
                (ni, nj, nk),
                "operands could not be broadcast together"
            );
        }
        assert_eq!(
            (wcon.dim().0, wcon.dim().1),
            (ni + 1, nj),
            "operands could not be broadcast together"
        );

        // Every statement of the kernel is elementwise over (i, j), so the
        // columns can be solved one after the other with the same arithmetic;
        // ccol and dcol then only need one column.
        let mut ccol = vec![Q32::ZERO; nk];
        let mut dcol = vec![Q32::ZERO; nk];
        for i in 0..ni {
            for j in 0..nj {
                let u = u_stage.slice(s![i, j, ..]);
                let pos = u_pos.slice(s![i, j, ..]);
                let tens = utens.slice(s![i, j, ..]);
                let (w_lo, w_hi) = (wcon.slice(s![i, j, ..]), wcon.slice(s![i + 1, j, ..]));
                let mut stage = utens_stage.slice_mut(s![i, j, ..]);

                // Forward sweep, k = 0.
                let gcv = QUARTER * (w_hi[1] + w_lo[1]);
                let cs = gcv * BET_M;
                ccol[0] = gcv * BET_P;
                let bcol = dtr_stage - ccol[0];
                let correction_term = -cs * (u[1] - u[0]);
                dcol[0] = dtr_stage * pos[0] + tens[0] + stage[0] + correction_term;
                let divided = Q32::ONE / bcol;
                ccol[0] *= divided;
                dcol[0] *= divided;

                // Forward sweep, 0 < k < K - 1.
                for k in 1..nk - 1 {
                    let gav = -QUARTER * (w_hi[k] + w_lo[k]);
                    let gcv = QUARTER * (w_hi[k + 1] + w_lo[k + 1]);
                    let as_ = gav * BET_M;
                    let cs = gcv * BET_M;
                    let acol = gav * BET_P;
                    ccol[k] = gcv * BET_P;
                    let bcol = dtr_stage - acol - ccol[k];
                    let correction_term = -as_ * (u[k - 1] - u[k]) - cs * (u[k + 1] - u[k]);
                    dcol[k] = dtr_stage * pos[k] + tens[k] + stage[k] + correction_term;
                    let divided = Q32::ONE / (bcol - ccol[k - 1] * acol);
                    ccol[k] *= divided;
                    dcol[k] = (dcol[k] - dcol[k - 1] * acol) * divided;
                }

                // Forward sweep, k = K - 1.
                let k = nk - 1;
                let gav = -QUARTER * (w_hi[k] + w_lo[k]);
                let as_ = gav * BET_M;
                let acol = gav * BET_P;
                let bcol = dtr_stage - acol;
                let correction_term = -as_ * (u[k - 1] - u[k]);
                dcol[k] = dtr_stage * pos[k] + tens[k] + stage[k] + correction_term;
                let divided = Q32::ONE / (bcol - ccol[k - 1] * acol);
                dcol[k] = (dcol[k] - dcol[k - 1] * acol) * divided;

                // Back substitution, k = K - 1 down to 0.
                let mut data_col = dcol[k];
                stage[k] = dtr_stage * (data_col - pos[k]);
                for k in (0..nk - 1).rev() {
                    data_col = dcol[k] - ccol[k] * data_col;
                    stage[k] = dtr_stage * (data_col - pos[k]);
                }
            }
        }
        utens_stage
    }
}
