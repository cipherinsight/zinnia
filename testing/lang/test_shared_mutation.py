"""Mutation of objects reachable through more than one name.

Python lists and NumPy arrays are shared by reference: after `b = a`, after
passing `a` to a function, or after storing `a` in another list, a mutation
through one name is visible through the other. Augmented assignment
(`a += v`) on an array or list is such a mutation. Zinnia used to rebind the
mutated name only, so the other name kept the old contents and the circuit
proved a result Python does not compute. Shared arrays are now updated in
place; mutating a shared list is rejected (lists are values in Zinnia).
"""
import numpy as np
import pytest

from zinnia import ZKCircuit
from zinnia.debug.exception import ZinniaException

SHARED_LIST = "the same list is also reachable through"


def accepts(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


def compile_source(source):
    return ZKCircuit.from_source("kernel", source).compile()


@pytest.mark.parametrize("body", [
    # Passed to a helper that writes into it.
    "def put(a):\n        a[0] = 5\n    lst = [1, 2, 3]\n    put(lst)\n    total = lst[0]",
    # Aliased by assignment, then appended to.
    "a = [1, 2, 3]\n    b = a\n    b.append(4)\n    total = len(a)",
    # An element list of another list, mutated through a second name.
    "rows = [[1, 2], [3, 4]]\n    r = rows[0]\n    r[0] = 9\n    total = rows[0][0]",
    # Augmented assignment extends a list in place.
    "a = [1, 2]\n    b = a\n    b += [3]\n    total = len(a)",
])
def test_mutating_a_shared_list_is_rejected(body):
    source = f'''def kernel(x: Integer[Array, "3"], expected: Integer):
    {body}
    assert total == expected
'''
    with pytest.raises(ZinniaException, match=SHARED_LIST):
        compile_source(source)


def test_copies_of_lists_can_be_mutated():
    program = compile_source('''def kernel(x: Integer[Array, "3"], expected: Integer):
    a = [1, 2, 3]
    b = list(a)
    b.append(4)
    c = a.copy()
    c[0] = 7
    assert len(a) + a[0] + len(b) + c[0] == expected
''')
    assert accepts(program, np.array([0, 0, 0]), 3 + 1 + 4 + 7)
    assert not accepts(program, np.array([0, 0, 0]), 4 + 7 + 4 + 7)


def test_augmented_assignment_in_helper_updates_callers_array():
    program = compile_source('''def kernel(x: Integer[Array, "3"], expected: Integer[Array, "3"]):
    def bump(a):
        a += 1
    y = x.copy()
    if x[0] > 1:
        bump(y)
    assert np.all(y == expected)
''')
    assert accepts(program, np.array([5, 2, 3]), np.array([6, 3, 4]))
    assert not accepts(program, np.array([5, 2, 3]), np.array([5, 2, 3]))
    assert accepts(program, np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert not accepts(program, np.array([1, 2, 3]), np.array([2, 3, 4]))


def test_augmented_assignment_through_alias_updates_the_array():
    program = compile_source('''def kernel(x: Integer[Array, "3"], expected: Integer[Array, "3"]):
    y = x.copy()
    z = y
    z += 10
    assert np.all(y == expected)
''')
    assert accepts(program, np.array([1, 2, 3]), np.array([11, 12, 13]))
    assert not accepts(program, np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_float_result_into_integer_array_is_rejected():
    with pytest.raises(ZinniaException, match="UFuncTypeError"):
        compile_source('''def kernel(x: Integer[Array, "3"]):
    y = x.copy()
    y += 0.5
    return y
''')
