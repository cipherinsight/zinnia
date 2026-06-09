from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.

@zk_chip
def call(i: Integer, j: Integer):
    return i + j

@zk_circuit
@requires(lambda size: 0 <= size <= 1024)
def approximated_callgraph(size: Integer):
    out = list()
    for i in range(size):
        out.append(list(map(lambda j: call(i, j), range(size))))
    return out
