# pythran program scrabble.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def scrabble_fun_score(word, scoretable):
    return sum([scoretable.get(x, 0) for x in word])

@zk_circuit
def scrabble_imp_score(word, scoretable):
    score = 0
    for letter in word:
        if letter in scoretable:
            score += scoretable[letter]
    return score
