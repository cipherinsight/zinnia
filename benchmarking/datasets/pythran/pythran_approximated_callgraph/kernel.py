# pythran program approximated_callgraph.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def call(i, j):
    return i + j

@zk_circuit
@requires(lambda size: 0 <= size <= 1024)
def approximated_callgraph(size: Integer):
    out = list()
    for i in range(size):
        out.append(list(map(lambda j: call(i, j), range(size))))
    return out
