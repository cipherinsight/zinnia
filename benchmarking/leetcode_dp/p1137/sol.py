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
import json

from zinnia import *


@zk_circuit
def verify_solution(
        n: int,
        sol: int
):
    assert n != 0 or sol == 0
    assert n != 1 or sol == 1
    assert n != 2 or sol == 1
    a, b, c = 0, 1, 1
    for i in range(3, 101):
        a, b, c = b, c, a + b + c
        assert n != i or sol == c


# def generate_solution(
#         n: int,
# ):
#     if n == 0:
#         return 0
#     if n == 1:
#         return 1
#     if n == 2:
#         return 1
#     a, b, c = 0, 1, 1
#     for i in range(3, 101):
#         a, b, c = b, c, a + b + c
#         if n == i:
#             return c


# solution = generate_solution(100)
# # print(solution)
# circuit = ZKCircuit.from_method(verify_solution)
# print(circuit.compile().source)
# json_dict = {}
# for entry in circuit.argparse(100, solution).entries:
#     json_dict[entry.get_key()] = int(entry.value)
# print(json.dumps(json_dict))
# json_dict = {}
# json_dict['n'] = 100
# json_dict['result'] = int(solution)
# print(json.dumps(json_dict))
