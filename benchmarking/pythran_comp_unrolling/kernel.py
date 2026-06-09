from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.

@zk_chip
def foo(cc: Float, x: Float, y: Float):
    for a in cc:
        if a:
            return True
        return False

@zk_circuit
def list_comp(cc: Integer[Array, 'N<=1024 M<=1024']):
    return [(x, y) for x in range(1) for y in range(2) if foo(cc, x, y)]
