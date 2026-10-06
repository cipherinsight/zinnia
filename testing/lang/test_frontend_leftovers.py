"""Programs that used to reach internal errors or a wrong rewrite."""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel
from zinnia.debug.exception.base import ZinniaException


def accepts(program, *args) -> bool:
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


def test_for_over_np_repeat_visits_the_repeated_elements():
    # A rewrite meant for `itertools.repeat` matched any call named `repeat`
    # and iterated k times over the whole array.
    program = compile_kernel('''
def kernel(a: NDArray[int, 2], expected: NDArray[int, 6]):
    out = []
    for v in np.repeat(a, 3):
        out.append(v)
    assert np.all(np.array(out) == expected)
''')
    a = np.array([4, 7])
    assert accepts(program, a, np.repeat(a, 3))
    assert not accepts(program, a, np.array([4, 7, 4, 7, 4, 7]))


def test_full_median_is_a_scalar():
    # `np.median` without an axis returns a float scalar in NumPy; it was a
    # 0-d array here, and `int(2 * np.median(x))` failed with an internal
    # "cannot materialize a composite" error.
    program = compile_kernel('''
def kernel(x: NDArray[int, 4], expected: int):
    assert int(2 * np.median(x)) == expected
''')
    x = np.array([7, 1, 4, 10])
    truth = int(2 * np.median(x))
    assert accepts(program, x, truth)
    assert not accepts(program, x, truth + 1)


def test_median_without_arguments_is_a_clear_error():
    with pytest.raises(ZinniaException, match="np.median requires an array argument"):
        compile_kernel('''
def kernel(x: NDArray[int, 4]):
    return np.median()
''')
