"""Sorts lowered by the cost rule of `optim::sort_lowering` (hint-op, 2026-10-06).

`np.sort`, `np.argsort`, `np.median` and `np.percentile` lower each sort's live
outputs with a sorting network or with prover hints pinned in the IR, picked per
compile target (`ZinniaConfig(backend=...)`). Every lowering must compute NumPy's
stable sort, ties included, on both targets; these tests compare with NumPy on the
mock backend, check that the circuit backends agree with the mock, and pin which
lowering each target picks (so a cost-rule regression fails here). The pins' tamper
tests (a wrong hint is rejected on mock, halo2-base and Noir) are Rust tests:
`optim::sort_lowering::tests`, `prove::halo2_base::synthesizer::tests::
sort_hints_are_pinned`, `prove::noir::soundness_tests::sort_hints_are_pinned_in_noir`.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import run_mock
from zinnia import ZKCircuit
from zinnia.config.zinnia_config import ZinniaConfig

TARGETS = ["halo2", "noir"]
BN254_P = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
HALO2_PARAMS = {"k": 12, "precision_bits": 32, "lookup_bits": 8}


def compile_for(source, target):
    return ZKCircuit.from_source("kernel", source, config=ZinniaConfig(backend=target)).compile()


def ir_classes(program):
    return [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]


class StableNumPy:
    """NumPy with `np.argsort` defaulting to `kind="stable"`, the order Zinnia
    proves (NumPy's default introsort leaves the order of ties unspecified)."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def argsort(a, axis=-1, kind=None, order=None):
        return np.argsort(a, axis=axis, kind="stable", order=order)


def reference(source):
    """The kernel source run as plain NumPy with stable argsorts."""
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
    env = {"np": StableNumPy()}
    exec(compile(ast.fix_missing_locations(tree), "<kernel>", "exec"), env)
    return env["kernel"]


def assert_numpy(source, target, *args, atol=1e-6):
    expected = np.asarray(reference(source)(*[np.copy(a) for a in args]), dtype=float).ravel()
    satisfied, outputs = run_mock(compile_for(source, target), *args)
    assert satisfied, "honest witness rejected"
    np.testing.assert_allclose(np.asarray(outputs, dtype=float), expected, atol=atol, rtol=0)


def with_ties(n, seed):
    rng = np.random.default_rng(seed)
    return np.round(rng.normal(size=n) * 2) / 2  # many equal values


INTS_WITH_TIES = np.array([3, -1, 3, 0, -1, 7, 3, 0, -9, 2, 2, 7])


# ── NumPy equality, ties included, on both targets ──────────────────────

SORT_KERNELS = [
    # (source, arguments)
    ('def kernel(a: Float[Array, "40"]):\n    return np.sort(a)\n', (with_ties(40, 1),)),
    ('def kernel(a: Float[Array, "40"]):\n    return np.argsort(a)\n', (with_ties(40, 2),)),
    ('def kernel(a: Integer[Array, "12"]):\n    return np.argsort(a)\n', (INTS_WITH_TIES,)),
    ('def kernel(a: Integer[Array, "12"]):\n    return np.argsort(a)[::-1][:4]\n', (INTS_WITH_TIES,)),
    ('def kernel(a: Float[Array, "40"]):\n    return np.argsort(a)[::-1][:5]\n', (with_ties(40, 3),)),
    ('def kernel(a: Float[Array, "40"]):\n    return np.argsort(a)[:3]\n', (with_ties(40, 4),)),
    ('def kernel(a: Float[Array, "40"]):\n    return np.argsort(a)[10:13]\n', (with_ties(40, 5),)),
    ('def kernel(a: Float[Array, "40"]):\n    s = np.sort(a)\n    return [s[0], s[-1], s[7]]\n',
     (with_ties(40, 6),)),
    ('def kernel(a: Integer[Array, "12"]):\n    return [np.sort(a)[3], np.argsort(a)[3]]\n',
     (INTS_WITH_TIES,)),
    ('def kernel(a: Float[Array, "9"]):\n    return np.median(a)\n', (with_ties(9, 7),)),
    ('def kernel(a: Float[Array, "10"]):\n    return np.median(a)\n', (with_ties(10, 8),)),
    ('def kernel(a: Integer[Array, "12"]):\n    return np.median(a)\n', (INTS_WITH_TIES,)),
    ('def kernel(a: Float[Array, "20"]):\n    return np.percentile(a, 37)\n', (with_ties(20, 9),)),
    ('def kernel(a: Float[Array, "20"]):\n    return np.percentile(a, 62.5)\n', (with_ties(20, 10),)),
    ('def kernel(a: Float[Array, "3 4"], b: Integer[Array, "3 4"]):\n'
     '    i = np.argsort(a, axis=0)\n    return np.take_along_axis(b, i, axis=0)\n',
     (with_ties(12, 11).reshape(3, 4), np.arange(12).reshape(3, 4))),
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("source, args", SORT_KERNELS)
def test_sort_outputs_match_numpy(source, args, target):
    assert_numpy(source, target, *args)


RUNTIME_RANK = 'def kernel(a: Float[Array, "30"], t: Integer):\n    return np.sort(a)[t]\n'
REVERSED_RUNTIME_RANK = 'def kernel(a: Float[Array, "30"], t: Integer):\n    return np.sort(a)[::-1][t]\n'
RUNTIME_RANK_INDEX = 'def kernel(a: Integer[Array, "12"], t: Integer):\n    return np.argsort(a)[t]\n'


@pytest.mark.parametrize("target", TARGETS)
def test_reads_at_runtime_ranks_match_numpy(target):
    a = with_ties(30, 12)
    for source in (RUNTIME_RANK, REVERSED_RUNTIME_RANK):
        program = compile_for(source, target)
        for t in (0, 1, 14, 29, -1, -30):
            satisfied, outputs = run_mock(program, a, t)
            assert satisfied
            assert outputs[0] == pytest.approx(reference(source)(a, t), abs=1e-9), (source, t)
        for t in (30, -31):
            assert not run_mock(program, a, t)[0], (source, t)  # NumPy raises IndexError
    program = compile_for(RUNTIME_RANK_INDEX, target)
    for t in range(12):
        satisfied, outputs = run_mock(program, INTS_WITH_TIES, t)
        assert satisfied and outputs[0] == np.argsort(INTS_WITH_TIES, kind="stable")[t]


PERCENTILE = 'def kernel(a: Float[Array, "20"], p: Integer):\n    return np.percentile(a, p)\n'


@pytest.mark.parametrize("target", TARGETS)
def test_runtime_percentile_matches_numpy_with_ties(target):
    a = with_ties(20, 13)
    program = compile_for(PERCENTILE, target)
    for p in range(0, 101, 3):
        satisfied, outputs = run_mock(program, a, p)
        assert satisfied
        assert outputs[0] == pytest.approx(np.percentile(a, p), abs=1e-6), p
    for p in (-1, 101):
        assert not run_mock(program, a, p)[0]


# ── Which lowering each target picks ────────────────────────────────────

def test_full_argsort_is_a_network_on_halo2_and_hinted_on_noir():
    source = 'def kernel(a: Float[Array, "100"]):\n    return np.argsort(a)\n'
    halo2 = ir_classes(compile_for(source, "halo2"))
    noir = ir_classes(compile_for(source, "noir"))
    assert "HintIR" not in halo2 and halo2.count("LessThanFIR") == 1077
    # One hinted index per rank, 99 key comparisons, one native read each.
    assert noir.count("HintIR") == 100 and noir.count("LessThanFIR") == 99
    assert noir.count("ReadMemoryIR") == 100


@pytest.mark.parametrize("target", TARGETS)
def test_few_live_outputs_are_pinned_alone(target):
    # The indices of the 7 largest of 100: 7 hinted keys, a chain of 6
    # comparisons and one count of the 100 keys below the 7th largest.
    classes = ir_classes(compile_for('def kernel(a: Float[Array, "100"]):\n'
                                     '    return np.argsort(a)[::-1][:7]\n', target))
    assert classes.count("HintIR") == 7 and classes.count("LessThanFIR") == 106
    assert "SortedIR" not in classes


@pytest.mark.parametrize("target", TARGETS)
def test_percentile_is_two_counts(target):
    # Two order statistics at runtime ranks, each pinned by n comparisons
    # and n equality tests: no sort at all.
    classes = ir_classes(compile_for(PERCENTILE, target))
    assert classes.count("HintIR") == 2
    assert classes.count("LessThanFIR") == 40 and classes.count("EqualFIR") == 40
    assert "SelectFIR" not in classes


# ── The circuit backends agree with the mock ────────────────────────────

def assert_halo2_base_agrees(source, target, *args):
    program = compile_for(source, target)
    mock = program.prove(*args, backend="mock")
    circuit = program.prove(*args, backend="halo2-base", params=HALO2_PARAMS)
    by_statement = {int(e.split("=")[0].rsplit("_", 1)[1]): int(e.split("=")[1]) % BN254_P
                    for e in mock.public_values}
    expected = [by_statement[k] for k in sorted(by_statement)]
    got = [int.from_bytes(bytes.fromhex(h), "little") for h in circuit.public_values]
    assert got == expected


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("source, args", [
    ('def kernel(a: Float[Array, "24"]):\n    return np.argsort(a)\n', (with_ties(24, 14),)),
    ('def kernel(a: Float[Array, "40"]):\n    return np.argsort(a)[::-1][:5]\n', (with_ties(40, 15),)),
    (PERCENTILE, (with_ties(20, 16), 41)),
    (RUNTIME_RANK_INDEX, (INTS_WITH_TIES, 5)),
])
def test_halo2_base_proof_matches_mock(source, args, target):
    assert_halo2_base_agrees(source, target, *args)
