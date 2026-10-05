"""The ordered output check and the encodings it relies on."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarking.harness.case_sets import Tolerance
from benchmarking.harness.outputs import (BN254_MODULUS, FIELD_ELEMENTS, LOGICAL, ZINNIA_EXPOSURE,
                                          DecodedPublics, Leaf, check_outputs, decode_exposed,
                                          field_from_le_hex, field_to_le_hex, field_to_signed,
                                          fields_from_be_bytes, fields_to_be_bytes, float_to_q32,
                                          reference_leaves)

TIGHT = Tolerance(abs=1e-9, rel=0.0, source="test")


def test_quantizer_rounds_half_away_from_zero_like_rust():
    assert float_to_q32(0.1) == 429496730  # 0.1 * 2^32 = 429496729.6
    assert float_to_q32(0.5 / 2**32) == 1  # Python's round() would give 0
    assert float_to_q32(-0.5 / 2**32) == -1
    assert float_to_q32(-1.0) == -(2**32)


def test_field_encodings_round_trip():
    assert field_to_signed(BN254_MODULUS - 5) == -5
    assert field_to_signed(7) == 7
    assert field_from_le_hex(field_to_le_hex(-3)) == BN254_MODULUS - 3
    values = [1, BN254_MODULUS - 1, 2**200]
    assert fields_from_be_bytes(fields_to_be_bytes(values)) == values
    with pytest.raises(ValueError):
        fields_from_be_bytes(b"\x00" * 31)


def test_decode_exposed_uses_the_expose_kind():
    fields = [BN254_MODULUS - 2**31, 42]  # -0.5 in Q32, then an integer
    assert decode_exposed(fields, ["f", "i"]) == [-0.5, 42]
    with pytest.raises(ValueError):
        decode_exposed(fields, ["f"])


def test_reference_leaves_flatten_in_output_order():
    leaves, shapes = reference_leaves((np.array([[1, 2], [3, 4]]), np.float64(0.5), True))
    assert [leaf.value for leaf in leaves] == [1, 2, 3, 4, 0.5, 1]
    assert [leaf.kind for leaf in leaves] == ["int"] * 4 + ["float", "int"]
    assert shapes == [[2, 2], [], []]
    complex_leaves, _ = reference_leaves(np.array([1 + 2j]))
    assert [(l.value, l.kind) for l in complex_leaves] == [(1.0, "float"), (2.0, "float")]


def test_a_permuted_output_is_a_mismatch():
    """The old audit compared sorted multisets and passed permutations (audit H-09, M9)."""
    expected = [Leaf(v, "float") for v in (1.0, 2.0, 3.0)]
    check = check_outputs(expected, DecodedPublics(LOGICAL, [3.0, 2.0, 1.0]), TIGHT)
    assert check.status == "mismatch"
    assert [m[0] for m in check.mismatches] == [0, 2]


def test_counts_must_agree_and_integers_are_exact():
    ints = [Leaf(3, "int"), Leaf(4, "int")]
    assert check_outputs(ints, DecodedPublics(LOGICAL, [3, 4]), TIGHT).matched
    assert check_outputs(ints, DecodedPublics(LOGICAL, [3.0, 4.0]), TIGHT).matched
    assert not check_outputs(ints, DecodedPublics(LOGICAL, [3, 4.0000001]), TIGHT).matched
    assert not check_outputs(ints, DecodedPublics(LOGICAL, [3, 4, 0]), TIGHT).matched
    assert not check_outputs(ints, DecodedPublics(LOGICAL, [True, 4]), TIGHT).matched


def test_float_tolerance_is_absolute_plus_relative():
    expected = [Leaf(100.0, "float")]
    loose = Tolerance(abs=1e-6, rel=1e-3, source="test")
    assert check_outputs(expected, DecodedPublics(LOGICAL, [100.05]), loose).matched
    assert not check_outputs(expected, DecodedPublics(LOGICAL, [100.2]), loose).matched
    nan = [Leaf(float("nan"), "float")]
    assert not check_outputs(nan, DecodedPublics(LOGICAL, [0.0]), loose).matched


def test_zinnia_exposure_is_exact_or_unchecked():
    expected = [Leaf(1.5, "float"), Leaf(2, "int")]
    exposed = DecodedPublics(ZINNIA_EXPOSURE, [1.5, 2], [])
    assert check_outputs(expected, exposed, TIGHT).matched
    all_constant = DecodedPublics(ZINNIA_EXPOSURE, [], [1.5, 2])
    assert check_outputs(expected, all_constant, TIGHT).matched
    mixed = DecodedPublics(ZINNIA_EXPOSURE, [1.5], [2])
    assert check_outputs(expected, mixed, TIGHT).status == "unchecked"
    with_metadata = DecodedPublics(ZINNIA_EXPOSURE, [2, 1.5, 2], [])
    assert check_outputs(expected, with_metadata, TIGHT).status == "unchecked"


def test_field_elements_are_decoded_by_reference_type():
    expected = [Leaf(-0.5, "float"), Leaf(7, "int")]
    publics = DecodedPublics(FIELD_ELEMENTS, [-(2**31), 7])
    assert check_outputs(expected, publics, TIGHT).matched
    assert check_outputs(expected, DecodedPublics(FIELD_ELEMENTS, [7]), TIGHT).status \
        == "unchecked"


def test_compiler_layout_decodes_constants_metadata_and_dynamic_lengths():
    from benchmarking.harness.outputs import decode_with_layout

    q = lambda v: float_to_q32(v) % BN254_MODULUS  # noqa: E731
    fields = [q(0.25), 2, 1, q(1.5), q(-3.0), q(9.0), 2]  # cell 9.0 is padding
    i = lambda index: {"public": index, "kind": "i"}  # noqa: E731
    f = lambda index: {"public": index, "kind": "f"}  # noqa: E731
    layout = [
        {"shape": [2], "cells": [f(0), {"constant": 7}]},
        {"dynamic": {"length": i(1), "dims": [i(6), i(2)], "cells": [f(3), f(4), f(5)]}},
    ]
    assert decode_with_layout(fields, layout) == ([0.25, 7, 1.5, -3.0], [[2], [2, 1]])
    with pytest.raises(ValueError, match="exactly once"):
        decode_with_layout(fields + [5], layout)
    with pytest.raises(ValueError, match="outside"):
        decode_with_layout([q(0.25), 4, 1, q(1.5), q(-3.0), q(9.0), 4], layout)
    with pytest.raises(ValueError, match="multiply"):
        decode_with_layout([q(0.25), 2, 1, q(1.5), q(-3.0), q(9.0), 1], layout)


def test_reported_shapes_must_match_the_reference():
    expected = [Leaf(v, "float") for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    same = DecodedPublics(LOGICAL, values, shapes=[[2, 3]])
    assert check_outputs(expected, same, TIGHT, expected_shapes=[[2, 3]]).matched
    reshaped = DecodedPublics(LOGICAL, values, shapes=[[3, 2]])
    assert check_outputs(expected, reshaped, TIGHT, expected_shapes=[[2, 3]]).status == "mismatch"
