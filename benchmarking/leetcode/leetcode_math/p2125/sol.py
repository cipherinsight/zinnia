# 2125. Number of Laser Beams in a Bank
# Medium
# Topics
# Companies
# Hint
#
# Anti-theft security devices are activated inside a bank. You are given a 0-indexed binary string array bank representing the floor plan of the bank, which is an m x n 2D matrix. bank[i] represents the ith row, consisting of '0's and '1's. '0' means the cell is empty, while'1' means the cell has a security device.
#
# There is one laser beam between any two security devices if both conditions are met:
#
#     The two devices are located on two different rows: r1 and r2, where r1 < r2.
#     For each row i where r1 < i < r2, there are no security devices in the ith row.
#
# Laser beams are independent, i.e., one beam does not interfere nor join with another.
#
# Return the total number of laser beams in the bank.

from zinnia import *

@zk_circuit
def verify_solution(bank: NDArray[int, 5, 5], expected: int):
    m, n = bank.shape
    res = 0
    for si in range(n):
        for sj in range(m):
            for ti in range(n):
                for tj in range(m):
                    add_one = bank[si][sj] == 1 and bank[ti][tj] == 1 and si < ti
                    for k in range(si + 1, ti):
                        if any(bank[k][j] == 1 for j in range(sj, tj)):
                            add_one = False
                            break
                    if add_one:
                        res += 1
    assert res == expected
