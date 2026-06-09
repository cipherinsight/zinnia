from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.

@zk_circuit
def specialconvolve(a: Integer[Array, 'N<=1024 M<=1024']):
    rowconvol = a[1:-1, :] + a[:-2, :] + a[2:, :]
    colconvol = rowconvol[:, 1:-1] + rowconvol[:, :-2] + rowconvol[:, 2:] - 9 * a[1:-1, 1:-1]
    return colconvol
