# 2133. Check if Every Row and Column Contains All Numbers
# Easy
# Topics
# Companies
# Hint
#
# An n x n matrix is valid if every row and every column contains all the integers from 1 to n (inclusive).
#
# Given an n x n integer matrix matrix, return true if the matrix is valid. Otherwise, return false.
from zinnia import *


@zk_circuit
def verify_solution(matrix: NDArray[int, 10, 10], valid: int):
    assert 0 <= valid <= 1
    n = len(matrix)
    for i in matrix.flat:
        assert (not valid) or 1 < i <= n
