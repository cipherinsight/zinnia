# Source: Pythran tests/cases/arc_distance_list.py
# Original #pythran export: arc_distance_list((float, float) list, (float, float) list)
from zinnia import *
from math import sin, cos, atan2, sqrt, pi


@zk_circuit
def arc_distance_list(a: list, b: list):
    distance_matrix = []
    for theta_1, phi_1 in a:
        temp_matrix = [2 * (atan2(sqrt(temp), sqrt(1 - temp))) for temp in [sin((theta_2 - theta_1) / 2) ** 2 + cos(theta_1) * cos(theta_2) * sin((phi_2 - phi_1) / 2) ** 2 for theta_2, phi_2 in b]]
        distance_matrix.append(temp_matrix)

    _zinnia_result = distance_matrix
