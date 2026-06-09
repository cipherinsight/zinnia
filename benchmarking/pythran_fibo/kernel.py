from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.

@zk_chip
def rfibo(n: Integer):
    if n < 2:
        return n
    else:
        n_1 = rfibo(n - 1)
        n_2 = rfibo(n - 2)
        return n_1 + n_2

@zk_chip
def fibo(n: Integer):
    if n < 10:
        return rfibo(n)
    else:
        n_1 = 0
        'omp task shared(n,n_1)'
        n_1 = fibo(n - 1)
        n_2 = fibo(n - 2)
        'omp taskwait'
        return n_1 + n_2

@zk_circuit
@requires(lambda n: 0 <= n <= 1024)
def test(n: Integer):
    """omp parallel"""
    'omp single'
    f = fibo(n)
    return f
