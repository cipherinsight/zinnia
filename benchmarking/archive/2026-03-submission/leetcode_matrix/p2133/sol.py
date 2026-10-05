# 2133. Check if Every Row and Column Contains All Numbers
# Easy
# Topics
# Companies
# Hint
#
# An n x n matrix is valid if every row and every column contains all the integers from 1 to n (inclusive).
#
# Given an n x n integer matrix matrix, return true if the matrix is valid. Otherwise, return false.
import json

from zinnia import *


@zk_circuit
def verify_solution(matrix: NDArray[int, 10, 10], valid: int):
    assert 0 <= valid <= 1
    n = len(matrix)
    for x in matrix.flatten():
        assert (not valid) or 1 < x <= n


# np.random.seed(0)
# input_matrix = np.random.randint(0, 10, (10, 10))
# sol = 0
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
# json_dict["valid"] = sol
# print(json.dumps(json_dict, indent=2))
