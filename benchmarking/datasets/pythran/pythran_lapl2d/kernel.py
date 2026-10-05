from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_circuit
@requires(lambda In, Out, niter: 0 <= niter <= 1024)
def lapl2d(In: Float[Array, 'N<=1024'], Out: Float[Array, 'N<=1024'], niter: Integer):
    siz = In.shape[0]
    h2 = (1.0 / siz) ** 2
    for it in range(0, niter):
        Out[1:siz - 1, 1:siz - 1] = h2 * (In[0:siz - 2, 1:siz - 1] + In[1:siz - 1, 0:siz - 2] - 4.0 * In[1:siz - 1, 1:siz - 1] + In[2:siz, 1:siz - 1] + In[1:siz - 1, 2:siz])
        In, Out = (Out, In)
