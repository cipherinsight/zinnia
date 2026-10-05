from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, mask-where
# Expected outcome: compile/prove candidate
# Precondition: Record mask shape equality and active-element bounds for where/nonzero-style indexing.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def hdiff(in_field: Float[Array, 'N<=1024 M<=1024 K<=1024'], out_field: Float[Array, 'N<=1024 M<=1024 K<=1024'], coeff: Float[Array, 'N<=1024 M<=1024 K<=1024']):
    I, J, K = (out_field.shape[0], out_field.shape[1], out_field.shape[2])
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] + in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])
    res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    flx_field = np.where(res * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :]) > 0, 0, res)
    res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    fly_field = np.where(res * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :]) > 0, 0, res)
    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] - fly_field[:, :-1, :])
