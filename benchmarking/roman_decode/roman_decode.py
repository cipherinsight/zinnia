# Source: Pythran tests/cases/roman_decode.py
# Original #pythran export: decode(str)
from zinnia import *


@zk_circuit
def decode(roman: str):
    s, t = 'MDCLXVI', (1000, 500, 100, 50, 10, 5, 1)
    _rdecode = dict(zip(s, t))
    result = 0
    for r, r1 in zip(roman, roman[1:]):
        rd, rd1 = _rdecode[r], _rdecode[r1]
        result += -rd if rd < rd1 else rd
    _zinnia_result = result + _rdecode[roman[-1]]
