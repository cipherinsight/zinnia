# Source: Pythran tests/cases/pselect.py
# Original #pythran export: pselect(int)
from zinnia import *


def sel0(n):
    n.append(1)


def sel1(n):
    n.append(2.)


@zk_circuit
def pselect(n: int):
    l = list()
    for k in (n, not n):
        if k:
            a = sel0
        else:
            a = sel1
        a(l)
    _zinnia_result = l
