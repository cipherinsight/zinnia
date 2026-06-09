from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.

@zk_circuit
def bubble_sort(list0: Integer[Array, 'N<=1024']):
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
