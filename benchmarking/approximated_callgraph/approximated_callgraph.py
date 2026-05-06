# Source: Pythran tests/cases/approximated_callgraph.py
# Original #pythran export: approximated_callgraph(int)
from zinnia import *


def call(i, j):
    return i + j


@zk_circuit
def approximated_callgraph(size: int):
    out = list()
    for i in range(size):
        out.append(list(map(lambda j: call(i, j), range(size))))
    _zinnia_result = out
