# pythran program crotg.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import math

@zk_circuit
def CROTG(CA: Complex, CB: Complex, C: Float=0, S: Complex=0):
    if abs(CA) == 0.0:
        C = 0.0
        S = complex(1.0, 0.0)
        CA = CB
    else:
        SCALE = abs(CA) + abs(CB)
        NORM = SCALE * math.sqrt(abs(CA / SCALE) ** 2 + abs(CB / SCALE) ** 2)
        ALPHA = CA / abs(CA)
        C = abs(CA) / NORM
        S = ALPHA * CB.conjugate() / NORM
        CA = ALPHA * NORM
    return (CA, CB, C, S)
