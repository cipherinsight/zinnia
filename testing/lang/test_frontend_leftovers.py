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


def test_astype_of_a_scalar_is_a_scalar():
    # `np.floor(s / 2).astype(int)` (DS-1000 0467/0468) was a 0-d array, and
    # `max(0, i - r)` with it failed with "cannot materialize a composite".
    program = compile_kernel('''
def kernel(s: int, i: int, expected: int):
    r = np.floor(s / 2).astype(int)
    assert max(0, i - r) + r == expected
''')
    for s, i in [(3, 0), (3, 4), (7, 2)]:
        r = int(np.floor(s / 2).astype(int))
        truth = max(0, i - r) + r
        assert accepts(program, s, i, truth)
        assert not accepts(program, s, i, truth + 1)


def test_numpy_scalar_types_called_as_constructors_convert():
    # `np.float32(x)` returned the type itself, which then reached an
    # arithmetic op as an internal error.
    program = compile_kernel('''
def kernel(x: float, n: int, expected: float):
    assert np.float32(x) * 2 + np.float64(n) + np.int64(x) == expected
''')
    for x, n in [(2.75, 3), (-1.5, 0)]:
        truth = float(np.float64(x) * 2 + np.float64(n) + np.int64(x))
        assert accepts(program, x, n, truth)
        assert not accepts(program, x, n, truth + 1)


@pytest.mark.parametrize("call", ["np.uint8(x)", "np.int32(x)"])
def test_arithmetic_on_narrow_numpy_scalars_is_rejected(call):
    # The conversion follows NumPy (test_dtype_conversions.py); NumPy then
    # computes `+ 1` in the narrow type, Zinnia would compute it in int64.
    with pytest.raises(ZinniaException, match=r"NumPy computes it modulo 2\^(8|32)"):
        compile_kernel(f'''
def kernel(x: int):
    return {call} + 1
''')


def test_int_of_a_complex_number_is_pythons_type_error():
    with pytest.raises(ZinniaException, match=r"int\(\) argument must be a real number, not 'complex'"):
        compile_kernel('''
def kernel(x: float):
    return int(x * 1j)
''')
