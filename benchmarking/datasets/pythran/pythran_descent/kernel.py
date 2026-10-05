from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded, matmul-dot
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Declare shared matrix/vector dimensions explicitly; when not expressible as @requires, encode with matching symbolic annotation names and keep as review obligation.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import itertools as it
import numpy as np

@zk_circuit
@requires(lambda x, d, mu, N_epochs: 0 <= N_epochs <= 1024)
def np_descent(x: Float[Array, 'N<=1024'], d: Float[Array, 'N<=1024'], mu: Float, N_epochs: Integer):
    N = len(x)
    f = 2 / N
    y = np.zeros(N)
    err = np.zeros(N)
    w = np.zeros(2)
    grad = np.empty(2)
    for _ in it.repeat(None, N_epochs):
        err[:] = d - y
        grad[:] = (f * np.sum(err), f * np.dot(err, x))
        w += mu * grad
        y = w[0] + w[1] * x
    return w
