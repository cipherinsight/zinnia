# 1137. N-th Tribonacci Number
# Easy
# Topics
# Companies
# Hint
#
# The Tribonacci sequence Tn is defined as follows:
#
# T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.
#
# Given n <= 100, return the value of Tn.
from zinnia import *


@zk_circuit
def verify_solution(
        n: int,
        sol: int
):
    if n == 0:
        assert sol == 0
    elif n == 1:
        assert sol == 1
    elif n == 2:
        assert sol == 1
    else:
        a, b, c = 0, 1, 1
        for i in range(3, 101):
            a, b, c = b, c, a + b + c
            if n == i:
                assert sol == c
