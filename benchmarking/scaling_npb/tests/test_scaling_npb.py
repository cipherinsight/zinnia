"""The size rule, the kernel rewrite and the zkVM diff check of the NPBench/Pythran scaling study.

    .venv/bin/python -m pytest benchmarking/scaling_npb/tests
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from benchmarking.scaling_npb import generate, rule, zkvm_check


def test_scaled_knob_is_npbench_preset_t_formula():
    assert rule.scaled_knob(1000, 63) == 16
    assert rule.scaled_knob(100, 7) == 15
    assert rule.scaled_knob(3, 1875) == 2          # never below min(p, 2)
    assert rule.scaled_knob(1, 50) == 1


def test_choose_takes_the_smallest_divisor_that_fits():
    choice = rule.choose({"N": 100, "alpha": 1.5}, 32,
                         lambda k: (k, None if k["N"] <= 32 else "too big"))
    assert choice.divisor == 4 and choice.knobs == {"N": 25, "alpha": 1.5}


def test_choose_reports_why_no_divisor_fits():
    with pytest.raises(rule.NotScalable, match="initialize"):
        rule.choose({"N": 40}, 32, lambda k: (None, "initialize() fails"))


def test_within_cap_checks_integer_scalars_and_dimensions():
    assert rule.within_cap([np.zeros((4, 32)), 32, 2.5], 32) is None
    assert "dimension 33" in rule.within_cap([np.zeros((33,))], 32)
    assert "integer input 40" in rule.within_cap([40], 32)


def test_render_evaluates_knob_expressions():
    assert rule.render("x = np.linspace(0,10,{m*n}).reshape({m},{n})", {"m": 16, "n": 8}) == \
        "x = np.linspace(0,10,128).reshape(16,8)"
    assert rule.render("r = np.arange(0.,10., {10 / c})", {"c": 32}) == "r = np.arange(0.,10., 0.3125)"


def test_prepare_drops_uninitialised_buffers_and_self_imports():
    line = "import numpy as np; from lap import laplacian; b = np.empty_like(a); c = np.empty((3,))"
    assert rule.prepare(line, "lap") == "import numpy as np; b = np.zeros_like(a); c = np.zeros((3,))"


def test_every_template_uses_each_knob_and_renders_completely():
    for key, line in rule.PYTHRAN_LINES.items():
        assert "{" not in line.upstream, key
        for name in line.knobs:
            assert any(name in expression for expression in
                       re.findall(r"\{([^{}]+)\}", line.template)), (key, name)
        assert "{" not in rule.render(line.template, line.knobs), key


KERNEL = '''# header comment kept
from zinnia import *

@zk_circuit
@requires(lambda n, A: 0 <= n <= 1024)
def kernel(n: Integer, A: Float[Array, '14 15'], b: Float[Array, "15"]):
    return A @ b * n

@zk_circuit
def other(x: Float[Array, '4']):
    return x
'''


def test_rewrite_kernel_changes_only_the_entry_shapes():
    new = generate.rewrite_kernel(KERNEL, "kernel", {"n": "Integer", "A": 'Float[Array, "27 29"]',
                                                     "b": 'Float[Array, "29"]'})
    assert "A: Float[Array, '27 29'], b: Float[Array, \"29\"]" in new
    assert "x: Float[Array, '4']" in new and new.startswith("# header comment kept")
    assert new.replace("27 29", "14 15").replace('"29"', '"15"') == KERNEL
    assert generate.body_unchanged(KERNEL, new)


def test_size_key_reads_complex_pairs_as_one_axis():
    parameters = [{"name": "z", "kind": "array", "dtype": "complex", "shape": [3]},
                  {"name": "n", "kind": "scalar", "dtype": "int"}]
    assert generate.size_key(parameters, {"z": [[1, 0], [2, 0], [3, 0]], "n": 7}) == [["z", [3]], ["n", 7]]


def test_zkvm_compare_allows_only_size_constants():
    original = "const N: usize = 16;\nfn main() { let x = N; }\n"
    assert zkvm_check.compare(original, original)[0] == "identical"
    assert zkvm_check.compare(original, original.replace("16", "64"))[0] == "size constants"
    assert zkvm_check.compare(original, original.replace("x = N", "x = N + 1"))[0] == "not scalable"


def test_proportional_formula_keeps_ratios_and_never_grows():
    assert rule.proportional_knob(200, 16, 200) == 16 and rule.proportional_knob(100, 16, 200) == 8
    assert rule.proportional_knob(3, 16, 1000) == 2          # never below min(p, 2)
    choice = rule.choose_proportional({"m": 200, "n": 100}, 32, lambda k: (k, None))
    assert choice.knobs == {"m": 32, "n": 16} and choice.largest == 32 and choice.divisor is None
    assert rule.choose_proportional({"N": 30}, 32, lambda k: (k, None)).knobs == {"N": 30}


def test_proportional_lowers_the_target_until_the_line_fits():
    choice = rule.choose_proportional({"c": 200}, 32, lambda k: (k, None if k["c"] <= 30 else "33 points"))
    assert choice.largest == 30 and len(choice.tried) == 2
