# pythran program roman_decode.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def decode(roman):
    s, t = ('MDCLXVI', (1000, 500, 100, 50, 10, 5, 1))
    _rdecode = dict(zip(s, t))
    result = 0
    for r, r1 in zip(roman, roman[1:]):
        rd, rd1 = (_rdecode[r], _rdecode[r1])
        result += -rd if rd < rd1 else rd
    return result + _rdecode[roman[-1]]
