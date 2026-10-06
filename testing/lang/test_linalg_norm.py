"""`np.linalg.norm(x, ord, axis, keepdims)` follows NumPy (accept-gaps, 2026-10-06).

Before, every call with a keyword argument was rejected and no call was
lowered at all (DS-1000 0456-0458 use `axis=-1`). The lowering covers the
orders that need no singular values: vector `None`/2/1/0/inf/-inf, matrix
`None`/'fro'/1/-1/inf/-inf, over any axis or axis pair, with `keepdims`.
Each case is compared with NumPy on the mock, and halo2-base and Noir must
prove the mock's values exactly. Orders that need singular values or a
fractional power are rejected with an error naming them.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy
from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel
from zinnia.debug.exception.base import ZinniaException

RNG = np.random.default_rng(456)
A43 = RNG.normal(size=(4, 3))
A232 = RNG.normal(size=(2, 3, 2))
INTS = np.array([0, 3, -4, 0, 1])


def kernel(params: str, body: str) -> str:
    return f"def kernel({params}):\n    return {body}\n"


F43 = 'a: Float[Array, "4 3"]'

CASES = [
    # DS-1000 0456/0457: pairwise distances.
    (kernel(F43, "np.linalg.norm(a - a[:, None], axis = -1)"), (A43,)),
    # DS-1000 0458.
    (kernel(F43, "np.triu(np.linalg.norm(a - a[:, None], axis=-1))"), (A43,)),
    (kernel(F43, "np.linalg.norm(a)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, axis=0)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, axis=1, keepdims=True)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, keepdims=True)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord=2, axis=1)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord=1, axis=0)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord=np.inf, axis=1)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, -np.inf, 1)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, 0, 1)"), (np.where(A43 > 0, A43, 0.0),)),
    # Matrix norms.
    (kernel(F43, "np.linalg.norm(a, 'fro')"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord=1)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord=-1)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord=np.inf)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord=-np.inf)"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord=1, axis=(1, 0))"), (A43,)),
    (kernel(F43, "np.linalg.norm(a, ord='fro', axis=(1, 0), keepdims=True)"), (A43,)),
    # Rank 3: a vector axis and an axis pair.
    (kernel('a: Float[Array, "2 3 2"]', "np.linalg.norm(a, axis=1)"), (A232,)),
    (kernel('a: Float[Array, "2 3 2"]', "np.linalg.norm(a, ord=np.inf, axis=(2, 0))"), (A232,)),
    (kernel('a: Float[Array, "2 3 2"]', "np.linalg.norm(a)"), (A232,)),
    # Integer input gives a float result.
    (kernel('a: Integer[Array, "5"]', "np.linalg.norm(a)"), (INTS,)),
    (kernel('a: Integer[Array, "5"]', "np.linalg.norm(a, 0)"), (INTS,)),
    (kernel('a: Integer[Array, "5"]', "np.linalg.norm(a, 1)"), (INTS,)),
    # Edge shapes: one element, a length-1 axis.
    (kernel('a: Float[Array, "1"]', "np.linalg.norm(a)"), (np.array([-2.5]),)),
    (kernel('a: Float[Array, "3 1"]', "np.linalg.norm(a, axis=1)"), (np.array([[1.5], [-0.25], [0.0]]),)),
    # A Python list argument.
    (kernel('x: Float, y: Float', "np.linalg.norm([x, y])"), (3.0, -4.0)),
]


@pytest.mark.parametrize("source, args", CASES)
def test_norm_matches_numpy_on_the_mock(source, args):
    assert_matches_numpy(source, *args, atol=1e-8)


# halo2-base and Noir runs are slower; a sample covering every reduction kind.
BACKEND_SAMPLE = [CASES[i] for i in (0, 4, 9, 10, 13, 15, 19, 21, 25)]


@pytest.mark.parametrize("source, args", BACKEND_SAMPLE)
def test_norm_is_the_mock_value_on_every_backend(source, args):
    assert_all_backends_match_numpy(source, *args, atol=1e-8)


def test_a_wrong_norm_is_rejected():
    source = ('def kernel(a: Float[Array, "4 3"], expected: Float):\n'
              '    assert np.linalg.norm(a, axis=1)[2] == expected\n')
    program = compile_kernel(source)
    honest = program.prove(A43, 0.0, backend="mock")
    assert honest.proof_bytes_hex != "mock_satisfied"
    value = np.linalg.norm(A43, axis=1)[2]
    # The mock's value is within one Q32 unit of NumPy's; find it among neighbours.
    accepted = [d for d in range(-2, 3)
                if program.prove(A43, round(value * 2**32 + d) / 2**32, backend="mock").proof_bytes_hex
                == "mock_satisfied"]
    assert len(accepted) == 1


@pytest.mark.parametrize("body, message", [
    ("np.linalg.norm(a, 'nuc')", "needs singular values"),
    ("np.linalg.norm(a, 2)", "needs singular values"),
    ("np.linalg.norm(a, 3, axis=1)", "fractional power"),
    ("np.linalg.norm(a, 'fro', axis=1)", "Invalid norm order 'fro' for vectors"),
    ("np.linalg.norm(a, 3)", "Invalid norm order 3 for matrices"),
    ("np.linalg.norm(a, axis=(0, 0))", "Duplicate axes"),
    ("np.linalg.norm(a, axis=2)", "out of bounds"),
    ("np.linalg.norm(a, unknown=1)", "unexpected keyword argument `unknown`"),
])
def test_unsupported_orders_are_rejected_by_name(body, message):
    with pytest.raises(ZinniaException, match=message):
        compile_kernel(kernel(F43, body))


def test_rank_three_needs_an_axis_for_a_non_default_order():
    with pytest.raises(ZinniaException, match="Improper number of dimensions"):
        compile_kernel(kernel('a: Float[Array, "2 3 2"]', "np.linalg.norm(a, 1)"))


# ── np.triu / np.tril (DS-1000 0458 applies triu to a norm) ──────────────

TRIANGLE_CASES = [
    (kernel(F43, "np.triu(a)"), (A43,)),
    (kernel(F43, "np.tril(a)"), (A43,)),
    (kernel(F43, "np.triu(a, 1)"), (A43,)),
    (kernel(F43, "np.tril(a, k=-1)"), (A43,)),
    (kernel(F43, "np.triu(a, k=-2)"), (A43,)),
    (kernel('a: Float[Array, "2 3 2"]', "np.tril(a, 1)"), (A232,)),
    (kernel('a: Integer[Array, "5"]', "np.triu(a)"), (INTS,)),
    (kernel('a: Integer[Array, "5"]', "np.tril(a, 2)"), (INTS,)),
]


@pytest.mark.parametrize("source, args", TRIANGLE_CASES)
def test_triangles_match_numpy(source, args):
    assert_matches_numpy(source, *args, atol=1e-9)


@pytest.mark.parametrize("source, args", [TRIANGLE_CASES[i] for i in (2, 5, 6)])
def test_triangles_on_every_backend(source, args):
    assert_all_backends_match_numpy(source, *args, atol=1e-9)
