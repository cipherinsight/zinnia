# Source: Pythran tests/cases/scrabble.py
# Original #pythran export: scrabble_fun_score(str, str: int dict)
from zinnia import *


@zk_circuit
def scrabble_fun_score(word: str, scoretable: dict):
    _zinnia_result = sum([scoretable.get(x, 0) for x in word])
