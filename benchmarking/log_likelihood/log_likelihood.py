# Source: Pythran tests/cases/log_likelihood.py
# Original #pythran export: log_likelihood(float64[], float64, float64)
from zinnia import *

N = 64


@zk_circuit
def log_likelihood(data: NDArray[Float, 64], mean: float, sigma: float):
    s = (data - mean) ** 2 / (2 * (sigma ** 2))
    pdfs = np.exp(-s)
    pdfs /= np.sqrt(2 * np.pi) * sigma
    _zinnia_result = np.log(pdfs).sum()
