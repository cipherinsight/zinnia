# Source: Pythran tests/cases/multi_export.py
# Original #pythran export: a(int)
from zinnia import *


@zk_circuit
def a(i: int):
    _zinnia_result = i
