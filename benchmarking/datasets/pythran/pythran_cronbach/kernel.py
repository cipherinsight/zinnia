# pythran program cronbach.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def cronbach(itemscores: Float[Array, '16 16']):
    itemvars = itemscores.var(1, None, None, 1)
    tscores = itemscores.sum(0)
    nitems = len(itemscores)
    return nitems / (nitems - 1) * (1 - itemvars.sum() / tscores.var(None, None, None, 1))
