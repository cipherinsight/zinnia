from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy

@zk_circuit
@requires(lambda data, mean, sigma: sigma != 0.0)
def log_likelihood(data: Float[Array, 'N<=1024'], mean: Float, sigma: Float):
    s = (data - mean) ** 2 / (2 * sigma ** 2)
    pdfs = numpy.exp(-s)
    pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
    return numpy.log(pdfs).sum()
