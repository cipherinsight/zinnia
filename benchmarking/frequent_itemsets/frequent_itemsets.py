# Source: Pythran tests/cases/frequent_itemsets.py
# Original #pythran export: frequent_itemsets(str list list)
from zinnia import *
import itertools


@zk_circuit
def frequent_itemsets(sentences: list):
    SUPP_THRESHOLD = 100
    supps = []

    supp = {}
    for sentence in sentences:
        for key in sentence:
            if key in supp:
                supp[key] += 1
            else:
                supp[key] = 1
    print("|C1| = " + str(len(supp)))
    supps.append({k: v for k, v in supp.items() if v >= SUPP_THRESHOLD})
    print("|L1| = " + str(len(supps[0])))

    supp = {}
    for sentence in sentences:
        for combination in itertools.combinations(sentence, 2):
            if combination[0] in supps[0] and combination[1] in supps[0]:
                key = ','.join(combination)
                if key in supp:
                    supp[key] += 1
                else:
                    supp[key] = 1
    print("|C2| = " + str(len(supp)))
    supps.append({k: v for k, v in supp.items() if v >= SUPP_THRESHOLD})
    print("|L2| = " + str(len(supps[1])))

    supp = {}
    for sentence in sentences:
        for combination in itertools.combinations(sentence, 3):
            if (combination[0] + ',' + combination[1] in supps[1] and
                    combination[0] + ',' + combination[2] in supps[1] and
                    combination[1] + ',' + combination[2] in supps[1]):
                key = ','.join(combination)
                if key in supp:
                    supp[key] += 1
                else:
                    supp[key] = 1
    print("|C3| = " + str(len(supp)))
    supps.append({k: v for k, v in supp.items() if v >= SUPP_THRESHOLD})
    print("|L3| = " + str(len(supps[2])))

    _zinnia_result = supps
