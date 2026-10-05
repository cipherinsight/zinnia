from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, denominator-obligation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_circuit
def is_square(a_positive_int: Float):
    """ Check if it is a perfect square. """
    x = a_positive_int // 2
    seen = {x}
    while x * x != a_positive_int:
        x = (x + a_positive_int // x) // 2
        if x in seen:
            return False
        seen.add(x)
    return True
