# Source: Pythran tests/cases/babylonian.py
# Original #pythran export: is_square(int)
from zinnia import *


@zk_circuit
def is_square(a_positive_int: int):
    x = a_positive_int // 2
    seen = {x}
    while x * x != a_positive_int:
        x = (x + (a_positive_int // x)) // 2
        if x in seen:
            _zinnia_result = False
        seen.add(x)
    _zinnia_result = True
