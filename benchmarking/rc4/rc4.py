# Source: Pythran tests/cases/rc4.py
# Original #pythran export: rc4_crypt(str, str)
from zinnia import *


@zk_circuit
def rc4_crypt(data: str, key: str):
    S = range(256)
    j = 0
    out = []

    for i in range(256):
        j = (j + S[i] + ord(key[i % len(key)])) % 256
        S[i], S[j] = S[j], S[i]

    for char in data:
        i = j = 0
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        out.append(chr(ord(char) ^ S[(S[i] + S[j]) % 256]))

    _zinnia_result = ''.join(out)
