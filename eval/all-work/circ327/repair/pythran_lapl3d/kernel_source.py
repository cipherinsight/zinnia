# pythran program lapl3d.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def laplacien(image: Float[Array, '16 16 3']):
    out_image = np.abs(4 * image[1:-1, 1:-1] - image[0:-2, 1:-1] - image[2:, 1:-1] - image[1:-1, 0:-2] - image[1:-1, 2:])
    valmax = np.max(out_image)
    valmax = max(1.0, valmax) + 1e-09
    out_image /= valmax
    return out_image
