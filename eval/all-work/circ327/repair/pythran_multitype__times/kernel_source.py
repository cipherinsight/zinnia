# pythran program multitype.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda n, m: 0 <= n <= 1024)
@requires(lambda n, m: 0 <= m <= 1024)
def times(n: Integer, m: Integer):
    return n * m

@zk_circuit
def check2(x: Float[Array, '1'], y: Boolean[Array, '1']):
    if y:
        return (x, y)
    else:
        return (x + x, y)

@zk_circuit
def check(x: Float[Array, '1'], y: Integer):
    if y:
        return (x, y)
    else:
        return (x + x, y)
