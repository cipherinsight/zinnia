# pythran program euler_challenge13.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def solve(v: Integer):
    """ Sum big numbers and get first digits. """
    t = (37107, 46376, 74324, 91942, 23067, 89261, 28112, 44274, 47451, 70386)
    return str(sum(t) + v)[0:10]
