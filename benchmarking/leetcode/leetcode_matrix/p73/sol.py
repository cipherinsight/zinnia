# 73. Set Matrix Zeroes
# Medium
# Topics
# Companies
# Hint
#
# Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
#
# You must do it in place.

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
