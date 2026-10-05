# pythran program pselect.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda n: 0 <= n <= 1024)
def pselect(n: Integer):
    l = list()
    for k in (n, not n):
        if k:
            a = sel0
        else:
            a = sel1
        a(l)
    return l

@zk_chip
def sel0(n):
    n.append(1)

@zk_chip
def sel1(n):
    n.append(2.0)
