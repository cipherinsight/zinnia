# pythran program log_likelihood.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy

@zk_circuit
@requires(lambda data, mean, sigma: sigma != 0.0)
def log_likelihood(data: Float[Array, '10000'], mean: Float, sigma: Float):
    s = (data - mean) ** 2 / (2 * sigma ** 2)
    pdfs = numpy.exp(-s)
    pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
    return numpy.log(pdfs).sum()
