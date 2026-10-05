# pythran program another_quicksort.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def swap(l, idx1, idx2):
    if idx1 != idx2:
        tmp = l[idx1]
        l[idx1] = l[idx2]
        l[idx2] = tmp

@zk_chip
def partition(l):
    size = len(l)
    pivot_idx = size // 2
    val = l[pivot_idx]
    idx = size - 1
    if pivot_idx != idx:
        swap(l, pivot_idx, idx)
    idx = idx - 1
    i = 0
    while i <= idx:
        if l[i] > val:
            while l[idx] > val and idx > i:
                idx = idx - 1
            if idx != i:
                swap(l, i, idx)
                idx = idx - 1
            else:
                break
        i = i + 1
    assert idx == i or idx + 1 == i
    swap(l, i, size - 1)
    return i

@zk_circuit
def QuickSort(l: Integer[Array, '10']):
    size = len(l)
    if size > 1:
        idx = partition(l)
        l1 = []
        l2 = []
        for i in range(0, idx):
            l1.append(l[i])
        for i in range(idx, size):
            l2.append(l[i])
        QuickSort(l1)
        QuickSort(l2)
        for i in range(0, len(l1)):
            l[i] = l1[i]
        for i in range(0, len(l2)):
            l[len(l1) + i] = l2[i]
    return l
