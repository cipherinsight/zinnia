from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.

@zk_circuit
@requires(lambda n: 0 <= n <= 1024)
def fibo(n: Integer):
    """ fibonaccie compuation. """
    a, b = (1, 1)
    for _ in range(n):
        a, b = (a + b, a)
    return a
