"""Constructions that check an identity at a verifier challenge (layer-challenge, 2026-10-07).

On halo2-base the array lowering layer may check an identity among committed values at a
random point instead of computing it (`lower/challenge.rs`): a sort by hinted keys and a
grand product (`lower/sort/grand_product.rs`), `np.convolve` / `np.correlate` by a
polynomial identity and a 2-D float matrix product by Freivalds' check (`lower/linear`)
(`is_permutation` by a grand product: `lower::permutation::tests`). These tests check that the
selection picks them where they are cheaper and never on Noir, that the outputs equal
NumPy (sorts exactly; float sums with Q32 rounding) on the mock, and that halo2-base proofs
agree with the mock. The tamper tests are Rust tests (`lower::sort::grand_product_tests`,
`lower::linear::tests`, `lower::permutation::tests`, `prove::halo2_base::synthesizer::
two_phase_tests`).
"""
import numpy as np

from testing.lang.mock_outputs import numpy_argument, run_mock
import pytest

from testing.lang.test_sort_hints import assert_halo2_base_agrees, compile_for, ir_classes, reference

GRAND_PRODUCT = "sort: grand-product permutation"
POLYNOMIAL_IDENTITY = "linear: polynomial identity"
FREIVALDS = "linear: freivalds"
DIRECT = "linear: direct product sums"


def assert_matches_numpy(source, program, *args, atol=1e-9):
    expected = np.asarray(reference(source)(*[numpy_argument(a) for a in args]), dtype=float).ravel()
    satisfied, outputs = run_mock(program, *args)
    assert satisfied, "honest witness rejected"
    np.testing.assert_allclose(np.asarray(outputs, dtype=float), expected, atol=atol, rtol=0)


def identity_checks(program):
    return ir_classes(program).count("IdentityCheckIR")


# ── Sorts ────────────────────────────────────────────────────────────────

SORTS = {
    "argsort": 'def kernel(a: Float[Array, "{n}"]):\n    return np.argsort(a)\n',
    "sort": 'def kernel(a: Float[Array, "{n}"]):\n    return np.sort(a)\n',
    "gather": ('def kernel(a: Float[Array, "{n}"], b: Integer[Array, "{n}"]):\n'
               '    return b[np.argsort(a)]\n'),
    "top": 'def kernel(a: Float[Array, "{n}"]):\n    return np.argsort(a)[::-1][:3]\n',
}


def sort_inputs(n, kind, seed=0):
    rng = np.random.default_rng(seed)
    a = {
        "ties": np.round(rng.normal(size=n) * 2) / 2,
        "equal": np.full(n, -1.5),
        "sorted": np.arange(n) / 4.0 - 3,
        "reversed": (np.arange(n) / 4.0 - 3)[::-1].copy(),
        "distinct": rng.normal(size=n) * 100,
    }[kind]
    return a


@pytest.mark.parametrize("shape", list(SORTS))
@pytest.mark.parametrize("n", [2, 7, 64])
def test_grand_product_sorts_match_numpy(monkeypatch, shape, n):
    monkeypatch.setenv("ZINNIA_FORCE_CONSTRUCTION", GRAND_PRODUCT)
    source = SORTS[shape].format(n=n)
    program = compile_for(source, "halo2")
    assert identity_checks(program) == 1
    for kind in ["ties", "equal", "sorted", "reversed", "distinct"]:
        a = sort_inputs(n, kind)
        args = (a, np.arange(n) * 7 - 50) if "b:" in source else (a,)
        assert_matches_numpy(source, program, *args)


@pytest.mark.parametrize("shape, n, chosen", [
    # 0381's shape: hinted keys read from a table (11,958 cells) cost less
    # than the grand product (13,965) and the routed network (21,826).
    ("argsort", 100, False),
    # The grand product: 7,578 cells at k = 8 (6,558 at lookup width 8)
    # against 7,638 at k = 9 for hinted keys read from a table; committed
    # polynomials times 2^k 19,456 against 25,088 as the circuits configure
    # them (eval-redo/layer/cost-model-misses.md).
    ("sort", 64, True),
    ("argsort", 16, False),   # saves less than the challenge's one-time cost
])
def test_the_grand_product_is_chosen_where_it_pays(shape, n, chosen):
    source = SORTS[shape].format(n=n)
    assert (identity_checks(compile_for(source, "halo2")) == 1) == chosen
    assert identity_checks(compile_for(source, "noir")) == 0


# ── Convolutions and correlations ───────────────────────────────────────

CONVOLVE = 'def kernel(a: Float[Array, "{n}"], v: Float[Array, "{m}"]):\n    return np.convolve(a, v{mode})\n'
CORRELATE = 'def kernel(a: Float[Array, "{n}"], v: Float[Array, "{m}"]):\n    return np.correlate(a, v{mode})\n'


@pytest.mark.parametrize("template", [CONVOLVE, CORRELATE])
@pytest.mark.parametrize("mode", ["", ", 'full'", ", 'same'", ", 'valid'"])
@pytest.mark.parametrize("n, m", [(1, 1), (5, 2), (2, 5), (6, 6), (7, 3)])
def test_polynomial_identity_matches_numpy(monkeypatch, template, mode, n, m):
    monkeypatch.setenv("ZINNIA_FORCE_CONSTRUCTION", POLYNOMIAL_IDENTITY)
    source = template.format(n=n, m=m, mode=mode)
    program = compile_for(source, "halo2")
    assert identity_checks(program) == 1
    rng = np.random.default_rng(n * 10 + m)
    a, v = rng.normal(size=n) * 5, rng.normal(size=m) * 5
    # Each output is the exact sum of the Q32 products rounded once (as the
    # direct sums): within the inputs' own quantization of NumPy's float64.
    assert_matches_numpy(source, program, a, v, atol=1e-6)
    monkeypatch.setenv("ZINNIA_FORCE_CONSTRUCTION", DIRECT)
    direct = compile_for(source, "halo2")
    assert identity_checks(direct) == 0
    assert run_mock(direct, a, v) == run_mock(program, a, v), "the identity and the direct sums differ"


@pytest.mark.parametrize("n, chosen", [(64, True), (8, False)])
def test_the_polynomial_identity_is_chosen_where_it_pays(n, chosen):
    source = CONVOLVE.format(n=n, m=n, mode="")
    assert (identity_checks(compile_for(source, "halo2")) == 1) == chosen
    assert identity_checks(compile_for(source, "noir")) == 0


def test_integer_convolutions_keep_their_exact_sums():
    source = 'def kernel(a: Integer[Array, "64"], v: Integer[Array, "64"]):\n    return np.convolve(a, v)\n'
    assert identity_checks(compile_for(source, "halo2")) == 0


# ── Matrix products ─────────────────────────────────────────────────────

MATMULS = {
    "at": 'def kernel(a: Float[Array, "{r} {k}"], b: Float[Array, "{k} {c}"]):\n    return a @ b\n',
    "dot": 'def kernel(a: Float[Array, "{r} {k}"], b: Float[Array, "{k} {c}"]):\n    return np.dot(a, b)\n',
    "matmul": 'def kernel(a: Float[Array, "{r} {k}"], b: Float[Array, "{k} {c}"]):\n    return np.matmul(a, b)\n',
}


@pytest.mark.parametrize("spelling", list(MATMULS))
@pytest.mark.parametrize("r, k, c", [(1, 1, 1), (2, 3, 4), (4, 1, 3), (3, 5, 1), (6, 6, 6)])
def test_freivalds_matches_numpy(monkeypatch, spelling, r, k, c):
    monkeypatch.setenv("ZINNIA_FORCE_CONSTRUCTION", FREIVALDS)
    source = MATMULS[spelling].format(r=r, k=k, c=c)
    program = compile_for(source, "halo2")
    assert identity_checks(program) == 1
    rng = np.random.default_rng(r * 100 + k * 10 + c)
    a, b = rng.normal(size=(r, k)) * 5, rng.normal(size=(k, c)) * 5
    assert_matches_numpy(source, program, a, b, atol=1e-6)
    monkeypatch.setenv("ZINNIA_FORCE_CONSTRUCTION", DIRECT)
    assert run_mock(compile_for(source, "halo2"), a, b) == run_mock(program, a, b)


@pytest.mark.parametrize("n, chosen", [(16, True), (4, False)])
def test_freivalds_is_chosen_where_it_pays(n, chosen):
    source = MATMULS["at"].format(r=n, k=n, c=n)
    assert (identity_checks(compile_for(source, "halo2")) == 1) == chosen
    assert identity_checks(compile_for(source, "noir")) == 0


# ── Proofs agree with the mock ──────────────────────────────────────────

@pytest.mark.parametrize("force, source, args", [
    (GRAND_PRODUCT, SORTS["gather"].format(n=9), (sort_inputs(9, "ties", 4), np.arange(9) * 3 - 10)),
    (GRAND_PRODUCT, SORTS["sort"].format(n=6), (sort_inputs(6, "equal"),)),
    (POLYNOMIAL_IDENTITY, CONVOLVE.format(n=5, m=3, mode=""), (np.linspace(-2, 3, 5), np.array([0.5, -1.25, 2.0]))),
    (POLYNOMIAL_IDENTITY, CORRELATE.format(n=6, m=4, mode=", 'same'"), (np.linspace(-1, 4, 6), np.linspace(2, -2, 4))),
    (FREIVALDS, MATMULS["at"].format(r=3, k=4, c=2), (np.linspace(-3, 3, 12).reshape(3, 4), np.linspace(1, -2, 8).reshape(4, 2))),
])
def test_halo2_base_proofs_agree_with_the_mock(monkeypatch, force, source, args):
    monkeypatch.setenv("ZINNIA_FORCE_CONSTRUCTION", force)
    assert identity_checks(compile_for(source, "halo2")) == 1
    assert_halo2_base_agrees(source, "halo2", *args)

