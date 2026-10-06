"""Loops whose trip count is a runtime value (audit findings X2, X3).

`range(n)`, `range(len(x))`, `range(x.shape[k])` and counted `while` loops
with a private bound are unrolled to a proven maximum, with each iteration
active only while Python would run it. A loop without a provable maximum is
rejected at compile time; it is never emitted with zero or a guessed number of
iterations. Each test accepts Python's result and rejects a wrong one.
"""
import numpy as np
import pytest

from zinnia import ZKCircuit
from zinnia.debug.exception import ZinniaException

UNBOUNDED = "loop bound depends on a private value with no provable maximum"


def compile_source(source):
    return ZKCircuit.from_source("kernel", source).compile()


def accepts(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


COUNT_LOOPS = {
    "range_len": "for i in range(len(x)):\n        total += i + 1",
    "range_shape": "for i in range(x.shape[0]):\n        total += i + 1",
    "range_start_stop": "for i in range(1, len(x) + 1):\n        total += i",
    "counted_while": "i = 0\n    while i < len(x):\n        total += i + 1\n        i += 1",
    "counted_while_alias": "n = len(x)\n    i = 0\n    while i < n:\n        i += 1\n        total += i",
    "counted_while_lte": "i = 1\n    while i <= len(x):\n        total += i\n        i += 1",
}


@pytest.mark.parametrize("loop", sorted(COUNT_LOOPS))
@pytest.mark.parametrize("length", [0, 1, 3, 4])
def test_private_trip_count_matches_python(loop, length):
    source = f'''def kernel(x: Integer[Array, "N<=4"], expected: Integer):
    total = 0
    {COUNT_LOOPS[loop]}
    assert total == expected
'''
    program = compile_source(source)
    x = np.arange(length, dtype=np.int64)
    expected = length * (length + 1) // 2
    assert accepts(program, x, expected)
    # A circuit that ran no iterations (or all of them) would accept these.
    assert not accepts(program, x, 0 if expected else 10)
    assert not accepts(program, x, expected + 1)


def test_range_bound_from_precondition():
    source = '''@zk_circuit(requires=[lambda n, expected: 0 <= n <= 6])
def kernel(n: Integer, expected: Integer):
    s = 0
    for i in range(n):
        s += 2
    assert s == expected
'''
    program = compile_source(source)
    for n in (0, 3, 6):
        assert accepts(program, n, 2 * n)
        assert not accepts(program, n, 2 * n + 1)
    assert not accepts(program, 0, 12)


def test_an_upper_bound_alone_bounds_range():
    """H2: `@requires(n <= K)` bounds `range(n)`; a negative `n` runs zero
    iterations, as in Python, and needs no lower bound."""
    source = '''@zk_circuit(requires=[lambda n, expected: n <= 5])
def kernel(n: Integer, expected: Integer):
    s = 0
    for i in range(n):
        s += i + 1
    for j in range(1, n, 2):
        s += 10
    assert s == expected
'''
    program = compile_source(source)

    def python(n):
        return sum(i + 1 for i in range(n)) + 10 * len(range(1, n, 2))

    for n in (-7, -1, 0, 1, 4, 5):
        assert accepts(program, n, python(n))
        assert not accepts(program, n, python(n) + 1)
    # n = 6 violates the precondition (and needs a sixth iteration).
    assert not accepts(program, 6, python(6))


def test_a_lower_bound_alone_bounds_a_descending_range():
    source = '''@zk_circuit(requires=[lambda n, expected: n >= -3])
def kernel(n: Integer, expected: Integer):
    s = 0
    for i in range(2, n, -1):
        s += i
    assert s == expected
'''
    program = compile_source(source)
    for n in (-3, 0, 2, 9):
        assert accepts(program, n, sum(range(2, n, -1)))
        assert not accepts(program, n, sum(range(2, n, -1)) - 1)


def test_loop_target_after_loop_is_the_last_value_python_produced():
    source = '''def kernel(x: Integer[Array, "N<=4"], expected: Integer[Array, "2"]):
    i = -1
    for i in range(len(x)):
        pass
    j = 0
    while j < len(x):
        j += 1
    assert np.all(np.array([i, j]) == expected)
'''
    program = compile_source(source)
    for length in (0, 1, 3, 4):
        x = np.arange(length, dtype=np.int64)
        expected = np.array([length - 1 if length else -1, length])
        assert accepts(program, x, expected)
        assert not accepts(program, x, np.array([3, 4]) if length != 4 else np.array([2, 4]))


def test_for_else_runs_after_inactive_iterations():
    source = '''def kernel(x: Integer[Array, "N<=4"], expected: Integer):
    flag = 0
    for i in range(len(x)):
        if x[0] < 0:
            break
    else:
        flag = 1
    assert flag == expected
'''
    program = compile_source(source)
    assert accepts(program, np.array([1, 2]), 1)
    assert not accepts(program, np.array([1, 2]), 0)


@pytest.mark.parametrize("loop", [
    "for i in range(n):\n        s += 1",
    "i = 0\n    while i < n:\n        s += 1\n        i += 1",
    "while n > 0:\n        n = n - 1\n        s += 1",
])
def test_unbounded_private_trip_count_is_rejected(loop):
    source = f'''def kernel(n: Integer, expected: Integer):
    s = 0
    {loop}
    assert s == expected
'''
    with pytest.raises(ZinniaException, match=UNBOUNDED):
        compile_source(source)


def test_range_with_private_bound_outside_a_loop_is_rejected():
    source = '''def kernel(x: Integer[Array, "N<=4"]):
    return np.array(list(range(len(x))))
'''
    with pytest.raises(ZinniaException, match="range\\(\\) with a runtime argument"):
        compile_source(source)


def test_comprehension_over_bounded_dynamic_array():
    # Used to iterate over nothing (the comprehension silently dropped a
    # bounded-dynamic iterable) and accept a zero sum.
    source = '''def kernel(x: Integer[Array, "N<=4"], expected: Integer):
    vals = [v * 2 for v in x]
    assert sum(vals) == expected
'''
    program = compile_source(source)
    assert accepts(program, np.array([1, 2, 3]), 12)
    assert not accepts(program, np.array([1, 2, 3]), 0)


def test_comprehension_over_private_range():
    source = '''def kernel(x: Integer[Array, "N<=4"], expected: Integer):
    vals = [i * i for i in range(len(x))]
    assert sum(vals) == expected
'''
    program = compile_source(source)
    assert accepts(program, np.array([7, 7, 7]), 5)
    assert not accepts(program, np.array([7, 7, 7]), 14)


def test_counted_while_on_full_length_reads_elements():
    # Reads `x[i]` inside the loop. Iterations beyond the runtime length run
    # under a false condition; full-length inputs need no inactive reads.
    source = '''def kernel(x: Integer[Array, "N<=4"], expected: Integer):
    total = 0
    i = 0
    while i < len(x):
        total += x[i]
        i += 1
    assert total == expected
'''
    program = compile_source(source)
    x = np.array([1, 2, 3, 4])
    assert accepts(program, x, 10)
    assert not accepts(program, x, 0)
    assert not accepts(program, x, 6)


def test_while_else_without_break_runs():
    source = '''def kernel(x: Integer[Array, "2"], expected: Integer):
    i = 0
    flag = 0
    while i < 2:
        i += 1
    else:
        flag = 7
    assert flag == expected
'''
    program = compile_source(source)
    assert accepts(program, np.array([0, 0]), 7)
    assert not accepts(program, np.array([0, 0]), 0)


def test_while_else_with_break_is_rejected():
    source = '''def kernel(x: Integer[Array, "2"]):
    i = 0
    while i < 2:
        if x[i] > 0:
            break
        i += 1
    else:
        i = 5
    return i
'''
    with pytest.raises(ZinniaException, match="while ... else"):
        compile_source(source)
