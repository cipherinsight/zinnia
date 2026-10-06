"""Statistics, fixtures, the scope rule and the rejection classifier."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from benchmarking.harness.acceptance.classify import classify_compile_error, resource_rejection
from benchmarking.harness.analysis import stats
from benchmarking.harness.fixtures import FixtureError, load_fixture
from benchmarking.harness.kernels import function_roles, out_of_scope_imports


def test_median_quartiles_geomean():
    assert stats.median([3.0, 1.0, 2.0]) == 2.0
    assert stats.quartiles([1.0, 2.0, 3.0, 4.0, 5.0]) == (2.0, 4.0)
    assert stats.geometric_mean([0.5, 2.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        stats.geometric_mean([1.0, 0.0])


def test_bootstrap_interval_is_deterministic_and_brackets_the_geomean():
    ratios = [0.5, 0.6, 0.8, 1.1, 0.7, 0.9]
    low, high = stats.bootstrap_geomean_interval(ratios)
    assert (low, high) == stats.bootstrap_geomean_interval(ratios)
    assert low < stats.geometric_mean(ratios) < high
    assert stats.bootstrap_geomean_interval([2.0, 2.0]) == pytest.approx((2.0, 2.0))


def write_fixture(path, a_value, b_value, shape=(2,)):
    path.write_text(json.dumps({
        "case": "c", "function": "kernel",
        "parameters": [{"name": "x", "kind": "array", "dtype": "float", "shape": list(shape)},
                       {"name": "n", "kind": "scalar", "dtype": "int"}],
        "inputs": {"A": {"x": a_value, "n": 3}, "B": {"x": b_value, "n": 3}}}))
    return path


def test_fixture_arguments_follow_the_signature(tmp_path):
    fixture = load_fixture(write_fixture(tmp_path / "f.json", [1, 2], [3.5, 4]))
    x, n = fixture.arguments("B")
    assert x.dtype == np.float64 and x.tolist() == [3.5, 4.0]
    assert n == 3 and isinstance(n, int)
    assert len(fixture.sha256) == 64


def test_fixture_inputs_must_differ_and_match_shapes(tmp_path):
    with pytest.raises(FixtureError, match="must differ"):
        load_fixture(write_fixture(tmp_path / "same.json", [1, 2], [1, 2]))
    with pytest.raises(FixtureError, match="shape"):
        load_fixture(write_fixture(tmp_path / "shape.json", [1, 2, 3], [1, 2, 3]))


def test_complex_fixture_values_are_real_imaginary_pairs(tmp_path):
    path = tmp_path / "complex.json"
    path.write_text(json.dumps({
        "case": "c", "function": "kernel",
        "parameters": [{"name": "z", "kind": "array", "dtype": "complex", "shape": [2]},
                       {"name": "c", "kind": "scalar", "dtype": "complex"}],
        "inputs": {"A": {"z": [[1.0, 2.0], [0.0, -1.0]], "c": [0.5, 0.0]},
                   "B": {"z": [[3.0, 0.0], [1.5, 2.5]], "c": [0.0, 1.0]}}}))
    z, c = load_fixture(path).arguments("B")
    assert z.dtype == np.complex128 and z.tolist() == [3 + 0j, 1.5 + 2.5j]
    assert c == 1j
    path.write_text(path.read_text().replace("[3.0, 0.0]", "[3.0, 0.0, 1.0]"))
    with pytest.raises(FixtureError, match=r"not a \[real, imag\] pair"):
        load_fixture(path)


def test_a_kernel_without_parameters_has_one_input(tmp_path):
    path = tmp_path / "none.json"
    path.write_text(json.dumps({"case": "c", "function": "kernel", "parameters": [],
                                "inputs": {"A": {}, "B": {}}}))
    assert load_fixture(path).arguments("B") == []


def test_scope_rule(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("from zinnia import *\nimport numpy as np\nimport numpy.linalg\n"
                      "import math\n\n@zk_circuit\ndef kernel(x: Integer):\n    return x\n")
    assert out_of_scope_imports(kernel) == []
    assert function_roles(kernel) == {"kernel": "circuit"}
    kernel.write_text("import numpy as np\nfrom scipy import special\nimport os.path\n")
    assert out_of_scope_imports(kernel) == ["os", "scipy"]


@pytest.mark.parametrize("message, reason_class, construct", [
    ("ZinniaException: Only supported NumPy imports are allowed in circuit source",
     "out_of_scope", "import of a module other than numpy and math"),
    ("IR generation failed: shape element at axis 0 must be a compile-time constant int",
     "no_provable_bound", "array shape from a runtime value"),
    ("IR generation failed: loop bound depends on a private value with no provable maximum: "
     "cannot bound the number of iterations of range() because its start or stop is a runtime "
     "value", "no_provable_bound", "range() trip count"),
    ("IR generation failed: linspace `num` must be a compile-time constant int",
     "no_provable_bound", "np.linspace num from a runtime value"),
    ("IR generation failed: shape element at axis 0 must be a compile-time constant int (it is "
     "a runtime value in [0, 2]; a runtime size is not supported here)", "compiler_limitation",
     "shape element from a bounded runtime value where a constant is needed"),
    ("IR generation failed: op `pow_f` requires precondition `BoolComb {..}` could not be "
     "lowered to a witness constraint", "compiler_limitation",
     "precondition of `pow_f` neither proved nor checkable in the circuit"),
    ("IR generation failed: np.mgrid is not implemented", "unsupported_construct", "np.mgrid"),
    ("IR generation failed: np.linalg.matrix_rank is not implemented; supported verification "
     "helpers are ...", "unsupported_construct", "np.linalg.matrix_rank"),
    ("UnsupportedOperatorException: `is` / `is not` compare object identity",
     "unsupported_construct", "`is` / `is not`"),
    ("IR generation failed: Unknown function `zip` in @zk_circuit.", "unsupported_construct",
     "call of `zip` (no builtin, chip or numpy function of that name in scope)"),
    ("compile thread panicked: index out of bounds: the len is 3 but the index is 3",
     "compiler_error", "internal error"),
    ("IR generation failed: Variable `x_min` not found", "compiler_error", "internal error"),
    ("IR generation failed: IRBuilder::ensure_ptr: cannot materialize a composite",
     "compiler_error", "internal error"),
    ("something nobody anticipated", "unclassified", ""),
])
def test_rejection_classes(message, reason_class, construct):
    rejection = classify_compile_error(message)
    assert (rejection.reason_class, rejection.construct) == (reason_class, construct)


def test_resource_rejections_name_the_limit():
    assert resource_rejection("timeout after 301s", 300.0, "8G").construct == \
        "compile time over 300 s"
    assert resource_rejection("out of memory after 12s", 300.0, "8G").construct == \
        "compile memory over 8G"
