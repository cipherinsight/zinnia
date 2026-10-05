# pythran program collatz_modified.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def collatz_modified(target: Integer):
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
            return start
        start += 1
