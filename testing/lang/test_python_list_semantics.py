"""Python list semantics: element writes, indices out of range, repetition,
and functions used as values.

- `lst[i] = v` stores `v` as it is. It used to be cast to the type of the
  element it replaced, so a float stored into `[0, 0]` was truncated (pythran
  matmul builds its result in `[[0 for ...] for ...]` and proved every element
  rounded toward zero).
- A compile-time index outside the list is Python's `IndexError` where the
  access runs. A read used to give `None` (an internal error later) and a
  write was dropped. Under a runtime condition (a loop item past a runtime
  bound) the circuit now asserts the access does not run, so the honest
  inputs prove and the inputs on which Python raises are rejected.
- `lst * n` with a runtime `n` used to multiply every element by `n`
  (`[0] * (1 + n)` became `[0]`, pythran histogram); its length is unknown,
  so it is rejected.
- A chip used as a value (`a = sel0`, pythran pselect) is rejected by name
  instead of "Variable `sel0` not found".
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from zinnia import NDArray, ZKCircuit, requires, zk_chip, zk_circuit
from zinnia.debug.exception.base import ZinniaException

MATMUL = '''
def kernel(a: NDArray[float, 2, 2]):
    m = [[0 for r in range(2)] for c in range(2)]
    for i in range(2):
        for j in range(2):
            r = 0
            for k in range(2):
                r += a[i][k] * a[k][j]
            m[i][j] = r
    return m
'''

WRITES = [
    MATMUL,
    '''
def kernel(a: NDArray[float, 2, 2]):
    m = [0, 0, 0]
    m[1] = a[0, 1]
    m[-1] = a[1, 1]
    return m
''',
    '''
def kernel(a: NDArray[float, 2, 2]):
    m = [0, 0, 0]
    if a[0, 0] > 1.0:
        m[0] = a[0, 0]
    if a[0, 0] > 9.0:
        m[1] = a[1, 1]
    return m
''',
    '''
def kernel(a: NDArray[float, 2, 2]):
    m = [0, 0]
    for i in range(2):
        m[int(a[i, 0] > 0)] = a[i, 1]
    return m
''',
]


@pytest.mark.parametrize("source", WRITES)
def test_a_list_element_takes_the_written_value(source):
    a = np.array([[1.5, 2.25], [0.25, -1.0]])
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), a)
    expected, expected_shapes = numpy_outputs(source, a)
    assert satisfied and shapes == expected_shapes
    np.testing.assert_allclose(values, expected, atol=1e-6)


@zk_circuit
@requires(lambda a, n: 0 <= n <= 8)
def loop_past_the_list(a: NDArray[float, 3], n: int):
    l = [1.0, 2.0, 3.0]
    s = 0.0
    for i in range(1, n + 1):
        s = s + l[i]
        l[i] = a[0]
    return s, l


def test_constant_index_past_the_list_runs_only_where_python_does_not_raise():
    program = ZKCircuit.from_method(loop_past_the_list).compile()
    a = np.array([7.0, 8.0, 9.0])
    satisfied, values, _ = run_mock_layout(program, a, 2)
    assert satisfied and values == [5.0, 1.0, 7.0, 7.0]
    for n in (3, 5, 8):  # Python raises IndexError at l[3]
        assert not run_mock_layout(program, a, n)[0]


@pytest.mark.parametrize("body, message", [
    ("l = [1.0, 2.0]\n    return l[2]", "IndexError: index 2 is out of range"),
    ("l = [1.0, 2.0]\n    l[-3] = x[0]\n    return l", "IndexError: assignment index -3"),
    ("return [0] * (1 + n)", r"sequence repetition `seq \* n` with a runtime count"),
    ("return n * [x[0]]", r"sequence repetition `seq \* n` with a runtime count"),
])
def test_rejected_with_the_python_error(body, message):
    with pytest.raises(ZinniaException, match=message):
        compile_kernel(f"def kernel(x: NDArray[float, 2], n: Integer):\n    {body}\n")


def test_constant_repetition_still_repeats():
    source = "def kernel(x: NDArray[float, 2]):\n    return [x[0]] * 3 + 2 * [x[1]]\n"
    x = np.array([1.5, -2.0])
    satisfied, values, _ = run_mock_layout(compile_kernel(source), x)
    assert satisfied and values == numpy_outputs(source, x)[0] == [1.5, 1.5, 1.5, -2.0, -2.0]


@zk_chip
def add_one(n):
    return n + 1


@zk_circuit
def chip_as_value(n: int):
    a = add_one
    return a(n)


def test_a_chip_used_as_a_value_is_rejected_by_name():
    with pytest.raises(ZinniaException, match="functions as values are not supported: `add_one`"):
        ZKCircuit.from_method(chip_as_value).compile()
