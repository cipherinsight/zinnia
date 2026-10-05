# pythran program periodic_bc.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def periodic_bc(f: Float[Array, '3 3 3']):
    f[:, 0, :] = f[:, -2, :]
    return f
