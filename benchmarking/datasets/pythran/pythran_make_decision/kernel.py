# pythran program make_decision.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def md(E: Complex[Array, '160'], symbols: Complex[Array, '80']):
    L = E.shape[0]
    M = symbols.shape[0]
    syms_out = np.zeros(L, dtype=E.dtype)
    for i in range(L):
        im = np.argmin(abs(E[i] - symbols) ** 2)
        syms_out[i] = symbols[im]
    return syms_out
