from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.

@zk_circuit
@requires(lambda n, m: 0 <= n <= 1024)
@requires(lambda n, m: 0 <= m <= 1024)
def times(n: Integer, m: Integer):
    return n * m

@zk_circuit
def check2(x: Float, y: Float):
    if y:
        return (x, y)
    else:
        return (x + x, y)
