"""Sort-family constructions of the array lowering layer (layer-sort, 2026-10-06).

NumPy equality on the mock for both compile targets, ties and edge values
included, plus the shape of the lowering each target picks. The hints' tamper
tests are Rust tests (`lower::sort::search::tests`,
`prove::halo2_base::synthesizer::tests::searchsorted_position_is_pinned`,
`prove::noir::soundness_tests::searchsorted_position_is_pinned_in_noir`).
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import run_mock
from testing.lang.test_sort_hints import TARGETS, assert_halo2_base_agrees, compile_for, ir_classes

# ── argmax over a constant sorted table (np.searchsorted) ───────────────

ECDF = ('def kernel(t: Float):\n'
        '    ys = np.arange(1, 21) / 20.0\n'
        '    return (ys {cmp} t).argmax()\n')
TIED = ('def kernel(t: Float):\n'
        '    ys = np.array([0.1, 0.25, 0.25, 0.7, 2.0])\n'
        '    return (ys {cmp} t).argmax()\n')
MIRRORED = ('def kernel(t: Float):\n'
            '    ys = np.arange(1, 21) / 20.0\n'
            '    return (t {cmp} ys).argmax()\n')
INTS = ('def kernel(t: Integer):\n'
        '    ys = np.arange(10) * 3 - 4\n'
        '    return (ys {cmp} t).argmax()\n')

FLOAT_QUERIES = [-1.0, 0.0, 0.05, 0.05 + 2**-32, 0.1, 0.25, 0.26, 0.5, 0.7, 0.95, 1.0, 2.0, 7.5]
INT_QUERIES = list(range(-6, 30))


def numpy_argmax(source, t):
    env = {"np": np}
    body = source.replace("Float", "float").replace("Integer", "int")
    exec(body, env)
    return int(env["kernel"](t))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("template, queries", [
    (ECDF, FLOAT_QUERIES), (TIED, FLOAT_QUERIES), (MIRRORED, FLOAT_QUERIES), (INTS, INT_QUERIES),
])
@pytest.mark.parametrize("cmp", [">", ">=", "<", "<="])
def test_argmax_over_a_constant_table_matches_numpy(template, queries, cmp, target):
    source = template.format(cmp=cmp)
    program = compile_for(source, target)
    for t in queries:
        satisfied, outputs = run_mock(program, t)
        assert satisfied, (source, t)
        assert outputs[0] == numpy_argmax(source, t), (source, t)


@pytest.mark.parametrize("target", TARGETS)
def test_constant_table_search_is_a_hinted_position(target):
    # Neither 20 comparisons nor a scan: one hinted position, two pins.
    classes = ir_classes(compile_for(ECDF.format(cmp=">"), target))
    assert classes.count("HintIR") == 1
    assert classes.count("LessThanOrEqualFIR") == 2
    assert "GreaterThanFIR" not in classes and "LogicalOrIR" not in classes


def test_constant_table_search_proves_on_halo2_base():
    for template in (ECDF, TIED):
        for t in (0.25, 0.7, -3.0, 9.0):
            assert_halo2_base_agrees(template.format(cmp=">"), "halo2", t)
