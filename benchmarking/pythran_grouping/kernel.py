from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: mask-where
# Expected outcome: compile/prove candidate
# Precondition: Record mask shape equality and active-element bounds for where/nonzero-style indexing.

@zk_circuit
def grouping(values: Integer[Array, 'N<=1024']):
    import numpy as np
    diff = np.concatenate(([1], np.diff(values)))
    idx = np.concatenate((np.where(diff)[0], [len(values)]))
    return (values[idx[:-1]], np.diff(idx))
