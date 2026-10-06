"""Programs Zinnia cannot compile are rejected with a message that names the
construct, not an internal error (fix-frontend's remaining internal errors)."""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from zinnia.debug.exception.base import ZinniaException

REJECTED = {
    # A list mutated inside a comprehension filter or a conditional-expression
    # arm: the change would apply on some paths only (it used to crash with
    # "IRBuilder::create_ir(WriteMemory ...): arg[1] has no stmt_id").
    "append in a filtered comprehension": (
        "acc = []\n    [acc.append(v) for v in x if v > 0]\n    return len(acc)",
        r"`acc.append\(...\)` inside a comprehension filter"),
    "append in a conditional expression": (
        "acc = [0]\n    acc.append(1) if x[0] > 0 else None\n    return len(acc)",
        r"`acc.append\(...\)` inside a comprehension filter or one arm"),
    "None on one side of a conditional expression": (
        "y = x[0] if x[1] > 0 else None\n    return x",
        "None on one side and a Integer on the other"),
    "too many indices": ("return x[1, 2]", "too many indices for array"),
    "iterating a 0-d array": ("s = 0\n    for v in np.array(x[0]):\n        s += v\n    return s",
                              "iteration over a 0-d array"),
}


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_rejected_with_a_message_naming_the_construct(name):
    body, message = REJECTED[name]
    with pytest.raises(ZinniaException, match=message):
        compile_kernel(f"def kernel(x: NDArray[int, 4]):\n    {body}\n")


def test_the_same_programs_written_with_statements_compile():
    source = '''
def kernel(x: NDArray[int, 4]):
    acc = []
    for v in x:
        if v > 0:
            acc.append(v)
    return len(acc)
'''
    x = np.array([1, -2, 3, 0])
    satisfied, values, _ = run_mock_layout(compile_kernel(source), x)
    assert satisfied and values == numpy_outputs(source, x)[0] == [2]


def test_an_unguarded_append_in_a_comprehension_still_runs():
    source = '''
def kernel(x: NDArray[int, 4]):
    acc = []
    [acc.append(v) for v in x]
    return len(acc)
'''
    satisfied, values, _ = run_mock_layout(compile_kernel(source), np.array([1, -2, 3, 0]))
    assert satisfied and values == [4]
