"""The routed Waksman sort on halo2-base (sort-route, 2026-10-07).

`lower/sort/routed.rs`: packed keys routed into ascending order by a Waksman
network whose switch bits the prover supplies, checked increasing. The halo2-base
cost rule picks it for mid-sized sorts (Noir never; large ones take the grand
product of `lower/sort/grand_product.rs`, `test_challenge_constructions.py`). These
tests force it (`ZINNIA_FORCE_CONSTRUCTION`), compile for the halo2-base target,
check that the routed plan is the one lowered, compare with NumPy
on the mock (ties, all keys equal, already sorted, reversed, sizes that are not
powers of two) and check that halo2-base proofs agree with the mock. The tamper
tests (every switch setting of small networks rejected or giving NumPy's outputs)
are Rust tests: `lower::sort::routed_tests`,
`prove::halo2_base::synthesizer::tests::routing_settings_are_rejected_or_give_numpys_outputs`.
"""
import numpy as np

from testing.lang.mock_outputs import numpy_argument, run_mock
import pytest

from testing.lang.test_sort_hints import assert_halo2_base_agrees, compile_for, ir_classes, reference


@pytest.fixture(autouse=True)
def force_routing(monkeypatch):
    """Every sort that can be routed is routed (other groups as usual)."""
    monkeypatch.setenv("ZINNIA_FORCE_CONSTRUCTION", "sort: routed permutation")


def switch_count(n):
    """Switches of the Waksman network on n wires (`lower::sort::waksman`)."""
    if n < 2:
        return 0
    if n == 2:
        return 1
    return switch_count(n // 2) + switch_count(n - n // 2) + n - 1


def inputs(n, kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "ties":
        return np.round(rng.normal(size=n) * 2) / 2
    if kind == "equal":
        return np.full(n, 1.25)
    if kind == "sorted":
        return np.arange(n) / 4.0 - 3
    if kind == "reversed":
        return (np.arange(n) / 4.0 - 3)[::-1].copy()
    return rng.normal(size=n)


KINDS = ["ties", "equal", "sorted", "reversed", "distinct"]
SIZES = [16, 33, 100]


def argsort_kernel(n):
    return f'def kernel(a: Float[Array, "{n}"]):\n    return np.argsort(a)\n'


def sort_kernel(n):
    return f'def kernel(a: Float[Array, "{n}"]):\n    return np.sort(a)\n'


def gather_kernel(n):
    return (f'def kernel(a: Float[Array, "{n}"], b: Integer[Array, "{n}"]):\n'
            '    return b[np.argsort(a)]\n')


def assert_routed(program, n):
    classes = ir_classes(program)
    assert classes.count("HintIR") == switch_count(n), "the routed plan was not chosen"
    assert classes.count("AssertBitIR") == switch_count(n)


def assert_matches_numpy(source, program, *args):
    expected = np.asarray(reference(source)(*[numpy_argument(a) for a in args]), dtype=float).ravel()
    satisfied, outputs = run_mock(program, *args)
    assert satisfied, "honest witness rejected"
    np.testing.assert_allclose(np.asarray(outputs, dtype=float), expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("make", [argsort_kernel, sort_kernel, gather_kernel])
def test_routed_sorts_match_numpy(make, n):
    source = make(n)
    program = compile_for(source, "halo2")
    assert_routed(program, n)
    for kind in KINDS:
        a = inputs(n, kind)
        args = (a, np.arange(n) * 7 - 50) if "b:" in source else (a,)
        assert_matches_numpy(source, program, *args)


def test_routed_argsort_keeps_ties_in_original_order():
    n = 33
    program = compile_for(argsort_kernel(n), "halo2")
    a = np.zeros(n)
    a[::3] = -1.0
    satisfied, outputs = run_mock(program, a)
    assert satisfied
    assert outputs == list(np.argsort(a, kind="stable"))


def test_noir_never_routes():
    classes = ir_classes(compile_for(argsort_kernel(100), "noir"))
    assert "AssertBitIR" not in classes and classes.count("HintIR") == 100


@pytest.mark.parametrize("source, args", [
    (argsort_kernel(16), (inputs(16, "ties", 3),)),
    (sort_kernel(33), (inputs(33, "equal"),)),
    (gather_kernel(16), (inputs(16, "reversed"), np.arange(16) * 3 - 20)),
])
def test_routed_sort_proves_on_halo2_base(source, args):
    assert_routed(compile_for(source, "halo2"), len(args[0]))
    assert_halo2_base_agrees(source, "halo2", *args)
