from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_circuit
def smoothing(x: Float[Array, 'N<=1024'], alpha: Float):
    """
  Exponential smoothing of a time series
  For x = 10**6 floats
  - Python runtime: 9 seconds
  - Parakeet runtime: .01 seconds
  """
    s = x.copy()
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
    return s
