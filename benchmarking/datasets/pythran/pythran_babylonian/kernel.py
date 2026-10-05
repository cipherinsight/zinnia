# pythran program babylonian.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def is_square(a_positive_int: Integer):
    """ Check if it is a perfect square. """
    x = a_positive_int // 2
    seen = {x}
    while x * x != a_positive_int:
        x = (x + a_positive_int // x) // 2
        if x in seen:
            return False
        seen.add(x)
    return True
