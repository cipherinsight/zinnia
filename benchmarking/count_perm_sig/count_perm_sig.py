# Source: Pythran tests/cases/count_perm_sig.py
# Original #pythran export: count_perm_sig(int, int, int)
from zinnia import *
import random as rd


def signature(perm):
    i = perm.index(0)
    sig_1 = perm[i:] + perm[:i]
    sig_2 = sig_1[0:1] + sig_1[1:][::-1]

    if sig_1[1] < sig_2[1]:
        sig = sig_1
    else:
        sig = sig_2

    return tuple(sig)


@zk_circuit
def count_perm_sig(n: int, s: int, k: int):
    rd.seed(s)
    myset = set()
    count = 0
    permutation = [i for i in range(k)]
    for i in range(n):
        rd.shuffle(permutation)
        sig = signature(permutation)
        if sig not in myset:
            myset.add(sig)
            count += 1
    _zinnia_result = myset, count
