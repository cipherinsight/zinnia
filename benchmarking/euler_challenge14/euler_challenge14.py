# Source: Pythran tests/cases/euler_challenge14.py
# Original #pythran export: euler14(int)
from zinnia import *


def next_num(num):
    if num & 1:
        return 3 * num + 1
    else:
        return num // 2


def series_length(num, lengths):
    if num in lengths:
        return lengths[num]
    else:
        num2 = next_num(num)
        result = 1 + series_length(num2, lengths)
        lengths[num] = result
        return result


@zk_circuit
def euler14(MAX_NUM: int):
    num_with_max_length = 1
    max_length = 0
    lengths = {1: 0}
    for ii in range(1, MAX_NUM):
        length = series_length(ii, lengths)
        if length > max_length:
            max_length = length
            num_with_max_length = ii
    _zinnia_result = num_with_max_length, max_length
