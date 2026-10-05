# pythran program descent.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import itertools as it
import numpy as np

@zk_circuit
@requires(lambda x, d, mu, N_epochs: 0 <= N_epochs <= 1024)
def np_descent(x: Float[Array, '10000'], d: Float[Array, '10000'], mu: Float, N_epochs: Integer):
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
