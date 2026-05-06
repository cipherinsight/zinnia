# Source: Pythran tests/cases/bubble_sort.py
# Original #pythran export: bubble_sort(int list)
from zinnia import *


@zk_circuit
def bubble_sort(list0: NDArray[Integer, 64]):
    list1 = [x for x in list0]
    for i in range(0, len(list1) - 1):
        swap_test = False
        for j in range(0, len(list1) - i - 1):
            if list1[j] > list1[j + 1]:
                list1[j], list1[j + 1] = list1[j + 1], list1[j]
            swap_test = True
        if swap_test == False:
            break
    _zinnia_result = list1
