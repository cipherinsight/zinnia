"""A name bound to a static array on one path and a runtime-shaped array on
another holds the array of the path taken, with its runtime shape.

The merge promotes the static array to a runtime-shaped one; that used to
fail with an internal error (`create_ir(WriteMemory): arg has no stmt_id`)
because the promotion read the static array's cells without loading them.
Each case checks NumPy's result for every input and rejects a wrong one.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel

CASES = {
    "if_statement": ('''
@requires(lambda n, c, expected: 0 <= n <= 4)
def kernel(n: int, c: int, expected: float):
    y = np.ones(2)
    if c > 0:
        y = np.ones(n) * 3
    assert y.sum() + len(y) == expected
''', lambda n, c: (np.ones(n) * 3 if c > 0 else np.ones(2))),
    "both_branches_and_indexing": ('''
@requires(lambda n, c, expected: 1 <= n <= 4)
def kernel(n: int, c: int, expected: int):
    if c > 0:
        y = np.arange(n) * 2 + 1
    else:
        y = np.array([5, 7, 9])
    assert y[0] + y.sum() == expected
''', lambda n, c: (np.arange(n) * 2 + 1 if c > 0 else np.array([5, 7, 9]))),
    "conditional_expression": ('''
@requires(lambda n, c, expected: 0 <= n <= 4)
def kernel(n: int, c: int, expected: float):
    y = np.ones(n) if c > 0 else np.ones(2)
    assert y.sum() + len(y) == expected
''', lambda n, c: (np.ones(n) if c > 0 else np.ones(2))),
    "two_dimensional": ('''
@requires(lambda n, c, expected: 0 <= n <= 3)
def kernel(n: int, c: int, expected: float):
    y = np.ones((2, 3))
    if c > 0:
        y = np.ones((n, 3)) * 2
    assert y.sum() + y.shape[0] == expected
''', lambda n, c: (np.ones((n, 3)) * 2 if c > 0 else np.ones((2, 3)))),
}


def claim(name, array):
    if name == "both_branches_and_indexing":
        return int(array[0] + array.sum())
    if name == "two_dimensional":
        return float(array.sum() + array.shape[0])
    return float(array.sum() + len(array))


@pytest.mark.parametrize("name", sorted(CASES))
def test_merged_array_matches_numpy(name):
    source, reference = CASES[name]
    program = compile_kernel(source)
    lowest = 1 if name == "both_branches_and_indexing" else 0
    highest = 3 if name == "two_dimensional" else 4
    for n in range(lowest, highest + 1):
        for c in (0, 1):
            truth = claim(name, reference(n, c))
            assert program.prove(n, c, truth, backend="mock").proof_bytes_hex == "mock_satisfied", (n, c)
            assert program.prove(n, c, truth + 1, backend="mock").proof_bytes_hex != "mock_satisfied", (n, c)
