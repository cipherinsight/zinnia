# pythran program bubble_sort.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def bubble_sort(list0: Integer[Array, '11']):
    list1 = [x for x in list0]
    for i in range(0, len(list1) - 1):
        swap_test = False
        for j in range(0, len(list1) - i - 1):
            if list1[j] > list1[j + 1]:
                list1[j], list1[j + 1] = (list1[j + 1], list1[j])
            swap_test = True
        if swap_test == False:
            break
    return list1
