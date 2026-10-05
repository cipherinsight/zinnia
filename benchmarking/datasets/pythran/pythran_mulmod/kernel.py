# pythran program mulmod.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda x, y, m: 0 <= m <= 1024)
def gf2mulmod(x: Integer, y: Integer, m: Integer):
    z = 0
    while x > 0:
        if x & 1 != 0:
            z ^= y
        y <<= 1
        y2 = y ^ m
        if y2 < y:
            y = y2
        x >>= 1
    return z
