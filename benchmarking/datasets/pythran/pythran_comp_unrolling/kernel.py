# pythran program comp_unrolling.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def foo(cc, x, y):
    for a in cc:
        if a:
            return True
        return False

@zk_circuit
def list_comp(cc):
    return [(x, y) for x in range(1) for y in range(2) if foo(cc, x, y)]
