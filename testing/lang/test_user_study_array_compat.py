"""Behavioral regressions exposed by ordinary study-task implementations.

Inputs remain private at compilation; NumPy supplies independent expected values.
"""
import numpy as np
import pytest

from zinnia import ZKCircuit
from studies.user_study.validate import checked_source, python_function


def check(source, *inputs):
    expected = np.asarray(python_function(source)(*[x.copy() for x in inputs]))
    # A rank-zero result has a scalar annotation in the small scalar wrapper.
    if expected.ndim == 0:
        source = source.replace("return ", "return np.array([", 1).rstrip() + "])\n"
        expected = expected.reshape(1)
    circuit = ZKCircuit.from_source("kernel", checked_source(source, expected.shape))
    circuit.compile()
    result = circuit.prove(*inputs, expected, backend="mock")
    assert result.proof_bytes_hex == "mock_satisfied"
    wrong = expected.copy()
    wrong.flat[0] += 1
    assert circuit.prove(*inputs, wrong, backend="mock").proof_bytes_hex != "mock_satisfied"


@pytest.mark.parametrize("expr", [
    "np.clip(x, a_min=0, a_max=9)",
    "np.clip(x, min=0, max=9)",
    "x.clip(0, 9)",
    "(x + 0).clip(min=0, max=9)",
    "np.clip(x, None, 9)",
    "np.clip(x, 0, None)",
    "np.clip(x, 9, 0)",
    "np.clip(x, np.array([0, 1, 2]), np.array([3, 4, 5]))",
])
def test_clip_spellings_and_bounds(expr):
    check(f'def kernel(x: Integer[Array, "2 3"]):\n    return {expr}\n',
          np.array([[-3, 0, 4], [9, 10, 21]]))


def test_clip_out_writes_through_transposed_view():
    check('''def kernel(x: Integer[Array, "2 3"]):
    y = x.T
    np.clip(y, 0, 9, out=y)
    return x
''', np.array([[-3, 0, 4], [9, 10, 21]]))


@pytest.mark.parametrize("copy_expr", ["np.copy(x)", "x.copy()", "(x + 0).copy()"])
def test_copy_detaches_storage(copy_expr):
    check(f'''def kernel(x: Integer[Array, "2 3"]):
    y = {copy_expr}
    y[0, 0] = 100
    return x + y
''', np.arange(6).reshape(2, 3))


EINSUM_CASES = [
    ("ii->i", [(3, 3)]),
    ("ii->", [(3, 3)]),
    ("ij->ji", [(2, 3)]),
    ("ij->j", [(2, 3)]),
    ("ij", [(2, 3)]),
    ("ij,jk->ik", [(2, 3), (3, 4)]),
    ("ij,jk", [(2, 3), (3, 4)]),
    ("i,i->", [(4,), (4,)]),
    ("i,j->ij", [(2,), (3,)]),
    ("...i,...i->...", [(2, 3, 4), (1, 3, 4)]),
    ("...ij,...jk->...ik", [(2, 3, 4), (4, 2)]),
    ("...ij,...jk", [(2, 3, 4), (1, 4, 2)]),
    ("ijk,jil->kl", [(2, 3, 2), (3, 2, 4)]),
    ("i,i,i->i", [(4,), (4,), (4,)]),
]


@pytest.mark.parametrize("subscripts,shapes", EINSUM_CASES)
def test_einsum_against_numpy(subscripts, shapes):
    names = [f"a{i}" for i in range(len(shapes))]
    params = ", ".join(f'{name}: Integer[Array, "{" ".join(map(str, shape))}"]' for name, shape in zip(names, shapes))
    source = f"def kernel({params}):\n    return np.einsum({subscripts!r}, {', '.join(names)}, optimize=True)\n"
    inputs = [np.arange(np.prod(shape)).reshape(shape) % 7 - i for i, shape in enumerate(shapes)]
    check(source, *inputs)


@pytest.mark.parametrize("subscripts,shapes", [
    ("ii->i", [(2, 3)]),
    ("ij,jk->ik", [(2, 3), (2, 4)]),
    ("ij->kk", [(2, 3)]),
    ("ij->k", [(2, 3)]),
])
def test_einsum_rejects_invalid_contracts(subscripts, shapes):
    names = [f"a{i}" for i in range(len(shapes))]
    params = ", ".join(f'{name}: Integer[Array, "{" ".join(map(str, shape))}"]' for name, shape in zip(names, shapes))
    source = f"def kernel({params}):\n    return np.einsum({subscripts!r}, {', '.join(names)})\n"
    with pytest.raises(Exception, match="einsum"):
        ZKCircuit.from_source("kernel", source).compile()


@pytest.mark.parametrize("values", [[-1, -2, -3, -4], [1, 2, 3, 4], [-1, 2, -3, 4]])
def test_conditional_append_preserves_length_and_sum(values):
    check('''def kernel(x: Integer[Array, "4"]):
    active = []
    for v in x:
        if v > 0:
            active.append(v)
    return np.array([len(active), sum(active)])
''', np.array(values))


def test_nested_filtered_zip_comprehension():
    check('''def kernel(x: Integer[Array, "4"], y: Integer[Array, "3"]):
    return np.array([sum(a*b for a,b in zip(x,y) if a > 0 if b < 5)])
''', np.array([-1, 2, 3, 4]), np.array([2, 3, 7]))


def test_signed_integer_boundaries_survive_field_conversion():
    check('''def kernel(x: Integer[Array, "4"]):
    return np.where(x < 0, x, 0)
''', np.array([-(2**63), -1, 0, 2**63 - 1], dtype=np.int64))
