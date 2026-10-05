# pythran program slowparts.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
from numpy import zeros, power, tanh

@zk_circuit
def slowparts(d: Integer, re: Integer, preDz: Float[Array, '16 9 5'], preWz: Float[Array, '8 9 5'], SRW: Float[Array, '8 16'], RSW: Float[Array, '8 16'], yxV: Float[Array, '9 5 8'], xyU: Float[Array, '9 5 8'], resid: Integer):
    """ computes the linear algebra intensive part of the gradients of the grae
    """
    fprime = lambda x: 1 - power(tanh(x), 2)
    partialDU = zeros((d + 1, re, 2 * d, d))
    for k in range(2 * d):
        for i in range(d):
            partialDU[:, :, k, i] = fprime(preDz[k]) * fprime(preWz[i]) * (SRW[i, k] + RSW[i, k]) * yxV[:, :, i]
    return partialDU
