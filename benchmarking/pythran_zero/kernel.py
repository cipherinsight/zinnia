from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.

@zk_circuit
@requires(lambda n, m: 0 <= n <= 1024)
@requires(lambda n, m: 0 <= m <= 1024)
def zero(n: Integer, m: Integer):
    return [[0 for row in range(n)] for col in range(m)]
