# Source: Pythran tests/cases/insertion_sort.py
# Original #pythran export: insertion_sort(float list)
from zinnia import *


@zk_circuit
def insertion_sort(list2: NDArray[Float, 64]):
    for i in range(1, len(list2)):
        save = list2[i]
        j = i
        while j > 0 and list2[j - 1] > save:
            list2[j] = list2[j - 1]
            j -= 1
        list2[j] = save
