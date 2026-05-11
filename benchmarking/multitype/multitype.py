# Source: Pythran tests/cases/multitype.py
# Original #pythran export: times(int or str, int)
# Migration notes: Pythran original allowed `int or str` for `n`; Zinnia has no Union
# in type annotations, so the original syntax `n: int or str` is preserved verbatim
# below — this file is intentionally uncompilable to remain faithful to upstream.
from zinnia import *


@zk_circuit
def times(n: int or str, m: int):
    _zinnia_result = n * m
