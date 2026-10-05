# pythran program quicksort.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def partition(list, start, end):
    pivot = list[end]
    bottom = start - 1
    top = end
    done = 0
    while not done:
        while not done:
            bottom = bottom + 1
            if bottom == top:
                done = 1
                break
            if list[bottom] > pivot:
                list[top] = list[bottom]
                break
        while not done:
            top = top - 1
            if top == bottom:
                done = 1
                break
            if list[top] < pivot:
                list[bottom] = list[top]
                break
    list[top] = pivot
    return top

@zk_chip
def do_quicksort(list, start, end):
    if start < end:
        split = partition(list, start, end)
        do_quicksort(list, start, split - 1)
        do_quicksort(list, split + 1, end)

@zk_circuit
def quicksort(l: Integer[Array, '10'], s: Integer, e: Integer):
    do_quicksort(l, s, e)
