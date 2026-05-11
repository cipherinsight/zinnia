# Source: NPBench weather_stencils/hdiff (hdiff_numpy.py)
# Original signature: hdiff(in_field, out_field, coeff) — in_field is (I+4, J+4, K),
#   out_field and coeff are (I, J, K) float arrays.
# Migration notes:
#   - I, J, K hoisted to module-level constants; from "S" preset (I=J=64, K=60).
from zinnia import *

I = 64
J = 64
K = 60


@zk_circuit
def hdiff(in_field: NDArray[Float, 68, 68, 60],
          out_field: NDArray[Float, 64, 64, 60],
          coeff: NDArray[Float, 64, 64, 60]):
    I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (
        in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
        in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    flx_field = np.where(
        (res *
         (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res,
    )

    res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    fly_field = np.where(
        (res *
         (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res,
    )

    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])
