from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
from numpy import zeros, power, tanh

@zk_circuit
def slowparts(d: Integer, re: Integer, preDz: Float[Array, 'N<=1024 M<=1024 K<=1024'], preWz: Float[Array, 'N<=1024 M<=1024 K<=1024'], SRW: Float[Array, 'N<=1024 M<=1024'], RSW: Float[Array, 'N<=1024 M<=1024'], yxV: Float[Array, 'N<=1024 M<=1024 K<=1024'], xyU: Float[Array, 'N<=1024 M<=1024 K<=1024'], resid: Integer):
    """ computes the linear algebra intensive part of the gradients of the grae
    """
    fprime = lambda x: 1 - power(tanh(x), 2)
    partialDU = zeros((d + 1, re, 2 * d, d))
    for k in range(2 * d):
        for i in range(d):
            partialDU[:, :, k, i] = fprime(preDz[k]) * fprime(preWz[i]) * (SRW[i, k] + RSW[i, k]) * yxV[:, :, i]
    return partialDU
