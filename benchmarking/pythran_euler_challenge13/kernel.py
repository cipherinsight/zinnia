from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.

@zk_circuit
def solve(v: Float):
    """ Sum big numbers and get first digits. """
    t = (37107, 46376, 74324, 91942, 23067, 89261, 28112, 44274, 47451, 70386)
    return str(sum(t) + v)[0:10]
