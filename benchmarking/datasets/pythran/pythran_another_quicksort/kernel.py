from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_chip
def swap(l: Float[Array, 'N<=1024'], idx1: Float, idx2: Float):
    if idx1 != idx2:
        tmp = l[idx1]
        l[idx1] = l[idx2]
        l[idx2] = tmp

@zk_chip
def partition(l: Float[Array, 'N<=1024']):
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
def QuickSort(l: Integer[Array, 'N<=1024']):
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
