# Source: Pythran tests/cases/multitype.py
# Original #pythran export: times(int or str, int)
# Migration notes: chose int for first param.
from zinnia import *


@zk_circuit
def times(n: int, m: int):
    _zinnia_result = n * m
