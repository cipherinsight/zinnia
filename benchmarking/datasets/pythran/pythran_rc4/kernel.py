# pythran program rc4.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def rc4_crypt(data, key):
    S = range(256)
    j = 0
    out = []
    for i in range(256):
        j = (j + S[i] + ord(key[i % len(key)])) % 256
        S[i], S[j] = (S[j], S[i])
    for char in data:
        i = j = 0
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = (S[j], S[i])
        out.append(chr(ord(char) ^ S[(S[i] + S[j]) % 256]))
    return ''.join(out)
