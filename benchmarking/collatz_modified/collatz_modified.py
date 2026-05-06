# Source: Pythran tests/cases/collatz_modified.py
# Original #pythran export: collatz_modified(int)
from zinnia import *


@zk_circuit
def collatz_modified(target: int):
    start = 1
    while True:
        i = start
        steps = 0
        while True:
            if i == 1:
                break
            if i % 2 == 0:
                i = i // 2
            else:
                i = 3 * i + 1
            steps += 1
        if steps == target:
            _zinnia_result = start
        start += 1
