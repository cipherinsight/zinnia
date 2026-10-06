"""Cheaper Noir forms and output layout changes (resid-noir).

* A comparison of a value with a constant, once an asserted comparison has
  checked the value into an offset window (`t = x - c` in `[0, 2^W)`, e.g. the
  int64 input-domain check `x >= i64::MIN`), is Noir's native comparison of
  `t as u64` with `k - c`, instead of a hinted `qlt_bits` or `sign_abs`.
* `(p & q) | (r & s)`: both ANDs get a witness before the OR.
* A sum of three or more terms read by several sums that get a witness (the
  shared Horner coefficients of `np.in1d`) gets a witness of its own first.
* The extreme pin of `max`/`min` multiplies the gaps its bound checks
  range-check (`m - a_j`), not `a_j - m`.
* A rank-1 runtime-shaped output exposes its length once: its only dim
  repeats the length's public slot.

Each test: NumPy's result is accepted on the mock, halo2-base and Noir; every
single wrong claim is rejected on all three.
"""
import re

import numpy as np
import pytest

from benchmarking.harness.outputs import decode_with_layout
from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from testing.lang.test_dtype_conversions import NOIR_PARAMS, halo2_accepts, mock_accepts, noir_accepts
from zinnia import ZinniaConfig, ZKCircuit

BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617
HALO2 = {"k": 12, "lookup_bits": 8, "precision_bits": 32}
I64_MIN, I64_MAX = -(2**63), 2**63 - 1


class Programs:
    """One kernel compiled for each target (the compile target picks among
    equivalent constructions, e.g. the extreme pin on Noir)."""

    def __init__(self, source: str):
        self.halo2 = ZKCircuit.from_source("kernel", source, config=ZinniaConfig(backend="halo2")).compile()
        self.noir = ZKCircuit.from_source("kernel", source, config=ZinniaConfig(backend="noir")).compile()

    def accepted(self, *args) -> bool:
        return (mock_accepts(self.halo2, *args) and halo2_accepts(self.halo2, *args)
                and mock_accepts(self.noir, *args) and noir_accepts(self.noir, *args))

    def rejected(self, *args) -> bool:
        return not (mock_accepts(self.halo2, *args) or halo2_accepts(self.halo2, *args)
                    or mock_accepts(self.noir, *args) or noir_accepts(self.noir, *args))

    def noir_text(self, *args) -> str:
        return self.noir.emit_structural(*args, backend="noir", params=NOIR_PARAMS)


def assert_claims_pinned(programs, args, claims):
    """`claims` (one array per claimed output) accepted everywhere; flipping
    any single entry of any claim rejected everywhere."""
    assert programs.accepted(*args, *claims)
    for which, claim in enumerate(claims):
        for i in np.ndindex(claim.shape):
            wrong = [c.copy() for c in claims]
            wrong[which][i] = 1 - wrong[which][i]
            assert programs.rejected(*args, *wrong), (which, i)


# ── comparisons in a checked window ─────────────────────────────────────

WINDOW = (
    'def kernel(x: Integer[Array, "8"], lt: Integer[Array, "8"], le: Integer[Array, "8"],\n'
    '           gt: Integer[Array, "8"], ge: Integer[Array, "8"]):\n'
    '    assert np.all(np.where(x < {k}, 1, 0) == lt)\n'
    '    assert np.all(np.where(x <= {k}, 1, 0) == le)\n'
    '    assert np.all(np.where({k} < x, 1, 0) == gt)\n'
    '    assert np.all(np.where({k} <= x, 1, 0) == ge)\n'
)


@pytest.mark.parametrize("k", [0, 5, -3])
def test_comparisons_with_a_constant_read_the_checked_window(k):
    programs = Programs(WINDOW.format(k=k))
    x = np.array([I64_MIN, k - 1, k, k + 1, -1, 0, 7, I64_MAX])
    claims = [(x < k).astype(int), (x <= k).astype(int), (x > k).astype(int), (x >= k).astype(int)]
    text = programs.noir_text(x, *claims)
    # Every comparison is the native one on the window of the input check.
    assert text.count("as u64) <") >= 2 * 8, text
    assert "qlt_bits" not in text and "sign_abs" not in text, text
    assert_claims_pinned(programs, [x], claims)


def test_float_sign_tests_read_the_checked_window():
    """`x[x >= 0]` on floats (DS-1000 0412): the sign test of a Q32 input."""
    source = 'def kernel(x: Float[Array, "6"], expected: Integer[Array, "6"]):\n' \
             '    assert np.all(np.where(x >= 0, 1, 0) == expected)\n'
    programs = Programs(source)
    x = np.array([-1.5, 0.0, 2.25, -2.0**-32, 2.0**-32, -1000.0])
    expected = (x >= 0).astype(int)
    assert "as u64) <" in programs.noir_text(x, expected)
    assert_claims_pinned(programs, [x], [expected])


# ── OR of ANDs, shared Horner coefficients ──────────────────────────────

def test_or_of_two_ands():
    """`(A > b0) & (A < b1) | (A > b1) & (A < b2)` (DS-1000 0444)."""
    source = ('def kernel(A: Integer[Array, "8"], B: Integer[Array, "3"], expected: Integer[Array, "8"]):\n'
              '    m = np.logical_and(A > B[0], A < B[1]) | np.logical_and(A > B[1], A < B[2])\n'
              '    assert np.all(np.where(m, 1, 0) == expected)\n')
    programs = Programs(source)
    A = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    B = np.array([1, 4, 7])
    expected = (((A > B[0]) & (A < B[1])) | ((A > B[1]) & (A < B[2]))).astype(int)
    assert "std::as_witness" in programs.noir_text(A, B, expected)
    assert_claims_pinned(programs, [A, B], [expected])


def test_membership_with_shared_coefficients():
    """`~np.in1d(A, B)` (DS-1000 0442): Horner's rule on coefficients shared by every element."""
    source = ('def kernel(A: Integer[Array, "8"], B: Integer[Array, "3"], expected: Integer[Array, "8"]):\n'
              '    assert np.all(np.where(~np.in1d(A, B), 1, 0) == expected)\n')
    programs = Programs(source)
    A = np.array([1, 2, 3, 4, 5, 6, -7, 8])
    B = np.array([3, -7, 8])
    expected = (~np.isin(A, B)).astype(int)
    text = programs.noir_text(A, B, expected)
    # The coefficient sum shared by every element has one witness, which
    # each element's first Horner step `A[i] + s` reads.
    witnessed = re.findall(r"std::as_witness\((v\d+)\);", text)
    assert any(text.count(f" + {v};") == len(A) for v in witnessed), text
    assert_claims_pinned(programs, [A, B], [expected])


# ── extreme pin ─────────────────────────────────────────────────────────

def test_row_maximum_mask():
    """`a.max(axis=1, keepdims=1) == a` (DS-1000 0436), ties included."""
    source = ('def kernel(a: Float[Array, "3 5"], expected: Integer[Array, "3 5"]):\n'
              '    assert np.all(np.where(a.max(axis=1, keepdims=1) == a, 1, 0) == expected)\n')
    programs = Programs(source)
    a = np.array([[1.5, -2.0, 1.5, 0.25, -7.0], [-3.0, -1.0, -2.0, -1.5, -1.25], [0.0, 0.0, 0.0, 0.0, 0.0]])
    expected = (a.max(axis=1, keepdims=True) == a).astype(int)
    text = programs.noir_text(a, expected)
    # Each bound check's gap `m - a_j` is the product's factor: no `a_j - m`.
    maxima = set(re.findall(r"let (v\d+): Field = v\d+\[\d+\];", text))
    gaps = [g for g in re.findall(r"let (v\d+): Field = (v\d+) - (v\d+);\n    \1\.assert_max_bit_size::<64>\(\);", text)
            if g[1] in maxima]
    assert len(gaps) == a.size, text
    for _, m, x in gaps:
        assert f"= {x} - {m};" not in text, text
    assert_claims_pinned(programs, [a], [expected])


# ── rank-1 runtime-shaped output ────────────────────────────────────────

FILTER = 'def kernel(x: Integer[Array, "6"]):\n    return x[x >= 0]\n'


def test_rank_one_dynamic_output_exposes_its_length_once():
    program = compile_kernel(FILTER)
    dynamic = program.get_output_layout()[0]["dynamic"]
    assert dynamic["dims"] == [dynamic["length"]] and "public" in dynamic["length"]
    publics = [s["public"] for s in [dynamic["length"], *dynamic["cells"]] if "public" in s]
    assert sorted(publics) == list(range(len(publics)))
    assert program.describe_publics()[dynamic["length"]["public"]]["metadata"] == "length"
    for x in ([3, -1, 0, -7, 9, 2], [-1] * 6, [4, 5, 6, 7, 8, 9]):
        satisfied, values, shapes = run_mock_layout(program, np.array(x))
        expected, expected_shapes = numpy_outputs(FILTER, np.array(x))
        assert satisfied and values == expected and shapes == expected_shapes


def test_rank_one_dynamic_output_length_is_bound_on_halo2():
    program = ZKCircuit.from_source("kernel", FILTER).compile()
    keys = program.setup(params=HALO2)
    x = np.array([3, -1, 0, -7, 9, 2])
    proof = program.prove_with_keys(keys, x)
    assert program.verify_with_keys(keys, proof)
    fields = [int.from_bytes(bytes.fromhex(v), "little") for v in proof["public_values"]]
    values, shapes = decode_with_layout(fields, program.get_output_layout(), program.get_public_inputs())
    assert (values, shapes) == ([3, 0, 9, 2], [[4]])
    length = program.get_output_layout()[0]["dynamic"]["length"]["public"]
    for delta in (1, BN254 - 1):
        tampered = dict(proof, public_values=list(proof["public_values"]))
        changed = (fields[length] + delta) % BN254
        tampered["public_values"][length] = changed.to_bytes(32, "little").hex()
        assert not program.verify_with_keys(keys, tampered)


def test_decoder_accepts_shared_shape_entries_only():
    i = lambda index: {"public": index, "kind": "i"}  # noqa: E731
    shared = [{"output": {"return": []},
               "dynamic": {"length": i(0), "rank": {"constant": 1}, "dims": [i(0)], "cells": [i(1), i(2)]}}]
    assert decode_with_layout([1, 5, 0], shared) == ([5], [[1]])
    # A cell may not repeat another public value.
    repeated_cell = [{"output": {"return": []},
                      "dynamic": {"length": i(0), "rank": {"constant": 1}, "dims": [i(0)], "cells": [i(1), i(1)]}}]
    with pytest.raises(ValueError, match="exactly once"):
        decode_with_layout([1, 5], repeated_cell)


# ── long product sums (nargo's elaboration recursion limit) ─────────────

@pytest.mark.parametrize("source, args, good", [
    ('def kernel(a: Float[Array, "256"], b: Float[Array, "256"], expected: Float):\n'
     '    assert np.dot(a, b) == expected\n',
     (np.arange(256) % 7 - 3.0, np.arange(256) % 5 - 2.0), None),
    ('def kernel(a: Float[Array, "256"], expected: Float):\n'
     '    assert np.var(a) == expected\n',
     (np.where(np.arange(256) % 2 == 0, 1.0, -1.0),), 1.0),
])
def test_long_product_sums_compile_on_noir(source, args, good):
    """`np.dot` of length 256 and `np.var` of 256 values used to be one
    256-term expression that nargo rejects ("Reached the recursion limit")."""
    programs = Programs(source)
    good = float(np.dot(*args)) if good is None else good
    assert "qscale_down_raw" in programs.noir_text(*args, good)
    assert mock_accepts(programs.noir, *args, good) and noir_accepts(programs.noir, *args, good)
    assert not noir_accepts(programs.noir, *args, good + 1.0)
