# pythran program check_mask.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def check_mask(db: Boolean[Array, '3 4'], out: Boolean[Array, '3'], mask: Tuple[Integer, Integer, Integer]=(1, 0, 1)):
    for idx, line in enumerate(db):
        target, vector = (line[0], line[1:])
        if (mask == np.bitwise_and(mask, vector)).all():
            if target == 1:
                out[idx] = 1
    return out
