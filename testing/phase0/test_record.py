"""The record invariants catch a record that breaks them.

`record.violations` runs on every accepted circuit of the parity set
(`corpus._analyze_worker`), where it is expected to answer "nothing wrong".
A checker that answered that whatever it was given would be invisible, so
each invariant of `implementation.md` §5a is broken here in turn and the
answer is asserted.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from testing.phase0 import record


def key(text: str) -> dict[str, Any]:
    """A value key as an *entry* carries it: the full object, whose `"key"`
    field is the canonical string. A reference to a value — an operand, an
    alias, a subject — is that string alone (`export.rs`'s schema).
    """
    return {"key": text, "stmt_path": ["0"], "iters": [], "calls": [],
            "expr_index": 0}


def producer(kind: str, **fields: Any) -> dict[str, Any]:
    payload = {"kind": kind, "alias": None, "rules": None, "operands": None,
               "condition": None}
    payload.update(fields)
    return payload


def entry(name: str, prod: dict[str, Any], kind: str = "array") -> dict[str, Any]:
    desc = {"array": {"kind": "array", "dtype": "I64", "rank": 1, "dims": [],
                      "dims_pretty": [], "defcond": 0, "psi": 0},
            "none": {"kind": "none"}}[kind]
    return {"key": key(name), "name": None, "desc": desc, "producer": prod}


def clean() -> dict[str, Any]:
    """A two-entry record that breaks nothing: an input and a value read off it."""
    return {
        "descriptors": [
            entry("a", producer("input")),
            entry("b", producer("applied", rules=["E-Slice"],
                                 operands=["a"])),
        ],
        "constraints": [
            {"kind": "check", "provenance": "E-Slice", "subject": "b"},
            {"kind": "var_range", "provenance": "range of n"},
        ],
        "obligations": [
            {"kind": "CANFAIL", "provenance": "E-Slice", "subject": "b"},
        ],
    }


def broken(mutate: Callable[[dict[str, Any]], None]) -> list[str]:
    payload = clean()
    mutate(payload)
    return record.violations(payload)


def test_a_clean_record_breaks_nothing() -> None:
    assert record.violations(clean()) == []


def test_i1_an_applied_producer_names_a_rule() -> None:
    def drop(payload: dict[str, Any]) -> None:
        payload["descriptors"][1]["producer"]["rules"] = []

    assert broken(drop) == ["I1: b names no rule"]


def test_i1_only_a_value_with_no_content_is_opaque() -> None:
    def opaque(payload: dict[str, Any]) -> None:
        payload["descriptors"][1]["producer"] = producer("opaque")

    assert broken(opaque) == ["I1: b is an entry of kind array and may not be opaque"]


def test_i2_an_operand_is_recorded_earlier() -> None:
    def forward(payload: dict[str, Any]) -> None:
        payload["descriptors"][1]["producer"]["operands"] = ["c"]

    assert broken(forward) == [
        "I2: the producer of b names c, which Δ does not record before it"
    ]


def test_i3_a_check_and_an_obligation_name_an_entry() -> None:
    def elsewhere(payload: dict[str, Any]) -> None:
        payload["constraints"][0]["subject"] = "c"
        payload["obligations"][0]["subject"] = "c"

    assert broken(elsewhere) == [
        "I3: the check from E-Slice points at c, which Δ does not record",
        "I3: the CANFAIL obligation from E-Slice points at c, "
        "which Δ does not record",
    ]


def test_i4_keys_are_unique() -> None:
    def twice(payload: dict[str, Any]) -> None:
        payload["descriptors"].append(copy.deepcopy(payload["descriptors"][0]))

    assert broken(twice) == ["I4: two values share the key a"]


def test_a_composite_needs_no_rule_of_its_own() -> None:
    """§3.3 makes a composite a list of descriptors, so building one applies
    no transfer rule; its elements carry the rules that ran."""
    payload = clean()
    payload["descriptors"][1]["desc"] = {"kind": "composite", "composite": "list",
                                         "elems": []}
    payload["descriptors"][1]["producer"]["rules"] = []
    assert record.violations(payload) == []
