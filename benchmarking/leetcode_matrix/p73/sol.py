# 73. Set Matrix Zeroes
# Medium
# Topics
# Companies
# Hint
#
# Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
#
# You must do it in place.
import json

import numpy as np

from zinnia import *


@zk_circuit
def verify_solution(
        matrix: NDArray[int, 8, 10],
        sol: NDArray[int, 8, 10]
):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            if matrix[i, j] == 0:
                assert sol[:, j] == 0
                assert sol[i, :] == 0


def generate_solution(
        matrix: NDArray[int, 8, 10],
):
    m, n = matrix.shape
    sol = np.copy(matrix)
    for i in range(m):
        for j in range(n):
            if matrix[i, j] == 0:
                sol[:, j] = 0
                sol[i, :] = 0
    return sol


# np.random.seed(0)
# input_matrix = np.random.randint(0, 10, (8, 10))
# sol = generate_solution(input_matrix)
#
# entries = ZKCircuit.from_method(verify_solution).argparse(
#     input_matrix, sol
# ).entries
#
# json_dict = {}
# for entry in entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
#
# json_dict = {}
# json_dict["matrix"] = [int(x) for x in input_matrix.flatten().tolist()]
# json_dict["sol"] = [int(x) for x in sol.flatten().tolist()]
# print(json.dumps(json_dict, indent=2))
#
#
