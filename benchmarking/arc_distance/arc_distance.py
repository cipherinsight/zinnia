# Source: Pythran tests/cases/arc_distance.py
# Original #pythran export: arc_distance(float[:], float[], float[], float[])
from zinnia import *

N = 64


@zk_circuit
def arc_distance(theta_1: NDArray[Float, 64], phi_1: NDArray[Float, 64],
                 theta_2: NDArray[Float, 64], phi_2: NDArray[Float, 64]):
    temp = np.sin((theta_2 - theta_1) / 2) ** 2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
    distance_matrix = 2 * (np.arctan2(np.sqrt(temp), np.sqrt(1 - temp)))
    _zinnia_result = distance_matrix
