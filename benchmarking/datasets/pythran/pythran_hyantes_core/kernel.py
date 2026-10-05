# pythran program hyantes_core.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import math

@zk_circuit
@requires(lambda xmin, ymin, xmax, ymax, step, range_, range_x, range_y, t: step != 0.0)
@requires(lambda xmin, ymin, xmax, ymax, step, range_, range_x, range_y, t: range_ != 0.0)
def run(xmin: Float, ymin: Float, xmax: Float, ymax: Float, step: Float, range_: Float, range_x: Integer, range_y: Integer, t: Float[Array, '16 3']):
    pt = [[0] * range_y for _ in range(range_x)]
    'omp parallel for'
    for i in range(range_x):
        for j in range(range_y):
            s = 0
            for k in t:
                tmp = 6368.0 * math.acos(math.cos(xmin + step * i) * math.cos(k[0]) * math.cos(ymin + step * j - k[1]) + math.sin(xmin + step * i) * math.sin(k[0]))
                if tmp < range_:
                    s += k[2] / (1 + tmp)
            pt[i][j] = s
    return pt
