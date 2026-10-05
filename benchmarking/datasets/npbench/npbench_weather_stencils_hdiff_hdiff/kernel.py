# npbench program weather_stencils/hdiff/hdiff_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def hdiff(in_field: Float[Array, '68 68 60'], out_field: Float[Array, '64 64 60'], coeff: Float[Array, '64 64 60']):
    I, J, K = (out_field.shape[0], out_field.shape[1], out_field.shape[2])
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] + in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])
    res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    flx_field = np.where(res * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :]) > 0, 0, res)
    res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    fly_field = np.where(res * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :]) > 0, 0, res)
    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] - fly_field[:, :-1, :])
