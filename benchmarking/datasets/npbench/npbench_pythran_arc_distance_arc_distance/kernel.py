# npbench program pythran/arc_distance/arc_distance_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def arc_distance(theta_1: Float[Array, '16'], phi_1: Float[Array, '16'], theta_2: Float[Array, '16'], phi_2: Float[Array, '16']):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    temp = np.sin((theta_2 - theta_1) / 2) ** 2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
    distance_matrix = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))
    return distance_matrix
