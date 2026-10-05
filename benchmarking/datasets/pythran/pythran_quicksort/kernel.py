from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, in-place-mutation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_chip
def partition(list: Float[Array, 'N<=1024'], start: Float, end: Float):
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
def do_quicksort(list: Float, start: Float, end: Float):
    if start < end:
        split = partition(list, start, end)
        do_quicksort(list, start, split - 1)
        do_quicksort(list, split + 1, end)

@zk_circuit
def quicksort(l: Integer[Array, 'N<=1024'], s: Integer, e: Integer):
    do_quicksort(l, s, e)
