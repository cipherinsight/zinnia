"""Partial Q32 identities retain their windows; wide numerators stay exact."""
from pathlib import Path

import numpy as np
import pytest

from benchmarking.harness.kernels import compile_kernel as compile_file
from testing.lang.mock_outputs import compile_kernel, run_mock
from testing.lang.all_backends import BN254_P, noir_field_publics
from testing.lang.test_dtype_conversions import noir_accepts


def test_division_identity_preserves_intermediate_window():
    expression = "(x + x) / 1.0"
    # The final division brings the output back into range; bounding only
    # the final output would accept the invalid intermediate.
    program = compile_kernel(f"def kernel(x: Float, expected: Float):\n    assert ({expression}) / 2.0 == expected\n")
    for x, accepted in [(1.25, True), (-1.25, True), (1.5e9, False), (-1.5e9, False)]:
        assert run_mock(program, x, x)[0] == accepted
        assert noir_accepts(program, x, x) == accepted
    assert not run_mock(program, 1.25, 1.5)[0]
    assert not noir_accepts(program, 1.25, 1.5)


@pytest.fixture(scope="module", params=["halo2", "noir"])
def simplex(request):
    root = Path(__file__).resolve().parents[2]
    return compile_file(root / "benchmarking/datasets/pythran/pythran_projection_simplex/kernel.py",
                        "projection_simplex", backend=request.param), request.param


def test_simplex_rejects_out_of_window_identity(simplex):
    simplex, backend = simplex
    v = np.full(10, -2147483647.5)
    assert not run_mock(simplex, v, 1)[0]
    if backend == "noir":
        assert not noir_accepts(simplex, v, 1)
    else:
        # This synthesizes and checks concrete range constraints, without keys.
        with pytest.raises(Exception, match="Witness does not satisfy circuit constraints"):
            simplex.emit_structural(v, 1, backend="halo2-base")


@pytest.mark.parametrize("x, expected", [(0.5, 0.10000000009313226), (1e9, 0.10000000009313226)])
def test_simplex_wide_prefix_sum(simplex, x, expected):
    simplex, backend = simplex
    v = np.full(10, x)
    satisfied, outputs = run_mock(simplex, v, 1)
    assert satisfied
    assert outputs == [expected] * 10
    if backend == "noir":
        assert noir_field_publics(simplex, v, 1) == [round(expected * 2**32)] * 10
    else:
        simplex.emit_structural(v, 1, backend="halo2-base")


@pytest.mark.parametrize("expression, x, expected", [
    ("x + x", 1.5e9, 3e9),
    ("x + x", -1.5e9, -3e9),
    ("(x + x) * 0.25", 1.5e9, 0.75e9),
    ("int(x + x)", 1.5e9, 3e9),
    ("(x + x) // 2.0", 1.5e9, 1.5e9),
    ("(x + x) % 2.0", 1.5e9, 0.0),
])
def test_wide_multiply_cast_and_floor_division(expression, x, expected):
    program = compile_kernel(f"def kernel(x: Float):\n    return {expression}\n")
    assert run_mock(program, x) == (True, [expected])
    program.emit_structural(x, backend="halo2-base")
    scale = 1 if expression.startswith("int(") else 2**32
    assert noir_field_publics(program, x) == [round(expected * scale) % BN254_P]


@pytest.mark.parametrize("expression", ["(x + x) * 1.0", "1.0 * (x + x)", "x * 2.0"])
def test_exact_integer_factor_keeps_its_wide_domain(expression):
    # These lower to exact field multiplication on both real backends;
    # unlike DivF, they do not introduce a bounded quotient hint.
    program = compile_kernel(f"def kernel(x: Float, expected: Float):\n    assert ({expression}) / 2.0 == expected\n")
    for x in [1.5e9, -1.5e9]:
        assert run_mock(program, x, x)[0]
        assert noir_accepts(program, x, x)
