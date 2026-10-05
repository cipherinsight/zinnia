"""Decoding public values and the ordered output check (replaces `audit_bindings.py`).

The check compares a proof's public outputs with the NumPy reference element by
element, in the kernel's output order (tuple parts in order, each array
row-major): integers and booleans exactly, floats within the case tolerance.
Unlike the old audit (findings H-09 and M9) it never matches multisets, never
accepts unexplained extra values, and fails on any count difference.

How a system's public values become the kernel's output depends on the system,
so adapters report one of three layouts:

- `logical`: the values are already the kernel's flattened output (the
  per-system program contract requires this of every baseline program);
- `zinnia_exposure`: Zinnia's public vector in exposure order, decoded by the
  kind of each `ExposePublicI/F` statement, plus the constant outputs the
  compiler left out of the public vector (`elided`). The compiler does not yet
  export where the elided outputs and dynamic-shape metadata sit, so the
  check is exact for two layouts only: every output public (no elisions, as
  many publics as output elements) or every output constant (no publics).
  Anything else is reported as `unchecked`, never as a match. Once the
  compiler exports its output layout (`decode_with_layout`), the Zinnia
  adapters decode with it and report `logical` values with shapes instead;
- `field_elements`: signed field integers whose types are not known to the
  program; they are decoded by the type of the reference output at the same
  position when the counts agree, else `unchecked`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from benchmarking.harness.case_sets import Tolerance

#: The BN254 scalar field modulus (the field of halo2-base and of Noir/Barretenberg).
BN254_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617
Q32_SCALE = 1 << 32
MAX_REPORTED_MISMATCHES = 10

LOGICAL = "logical"
ZINNIA_EXPOSURE = "zinnia_exposure"
FIELD_ELEMENTS = "field_elements"


# ---------------------------------------------------------------------------
# Field and fixed-point encoding
# ---------------------------------------------------------------------------


def field_to_signed(value: int) -> int:
    """A field element as a signed integer: values above p/2 are negative."""
    value %= BN254_MODULUS
    return value - BN254_MODULUS if value > BN254_MODULUS // 2 else value


def signed_to_field(value: int) -> int:
    return value % BN254_MODULUS


def q32_to_float(quantized: int) -> float:
    return quantized / Q32_SCALE


def float_to_q32(value: float) -> int:
    """Zinnia's quantizer (`src/prove/field.rs`): `(v * 2^32).round() as i128`.

    Rust's `f64::round` rounds halves away from zero, unlike Python's `round`.
    """
    scaled = value * Q32_SCALE
    magnitude = math.floor(abs(scaled) + 0.5)
    return int(magnitude if scaled >= 0 else -magnitude)


def field_from_le_hex(text: str) -> int:
    """halo2-base public values: hex of the canonical 32-byte little-endian repr."""
    return int.from_bytes(bytes.fromhex(text), "little")


def field_to_le_hex(value: int) -> str:
    return (value % BN254_MODULUS).to_bytes(32, "little").hex()


def fields_from_be_bytes(data: bytes) -> list[int]:
    """Barretenberg's `public_inputs` file: 32-byte big-endian field elements."""
    if len(data) % 32:
        raise ValueError(f"public inputs file of {len(data)} bytes is not a multiple of 32")
    return [int.from_bytes(data[i:i + 32], "big") for i in range(0, len(data), 32)]


def fields_to_be_bytes(values: list[int]) -> bytes:
    return b"".join((v % BN254_MODULUS).to_bytes(32, "big") for v in values)


def decode_with_layout(fields: list[int],
                       layout: list[dict]) -> tuple[list[int | float], list[list[int]]]:
    """The kernel's output (flattened values, shape per part) from a public vector.

    `layout` is the output contract the compiler is asked to export (see the
    harness report): one entry per top-level output part (tuple element, or the
    single return value), in output order:

        slot    = {"public": index, "kind": "i" | "f"} | {"constant": value}
        static  = {"shape": [d, ...], "cells": [slot, ...]}            # prod(shape) cells
        dynamic = {"dynamic": {"length": slot, "dims": [slot, ...],    # runtime extents
                               "cells": [slot, ...]}}                  # max_length cells

    Only the first `length` cells of a dynamic part are output elements, and its
    dims must multiply to `length`. Every public index must be used exactly
    once, so no public value goes unexplained.
    """
    used: list[int] = []

    def value(slot: dict) -> int | float:
        if "constant" in slot:
            return slot["constant"]
        used.append(slot["public"])
        signed = field_to_signed(fields[slot["public"]])
        return q32_to_float(signed) if slot["kind"] == "f" else signed

    values: list[int | float] = []
    shapes: list[list[int]] = []
    for part in layout:
        if "dynamic" not in part:
            cells = [value(cell) for cell in part["cells"]]
            if math.prod(part["shape"]) != len(cells):
                raise ValueError(f"static part of shape {part['shape']} has {len(cells)} cells")
            values.extend(cells)
            shapes.append(list(part["shape"]))
            continue
        dynamic = part["dynamic"]
        length = value(dynamic["length"])
        dims = [value(d) for d in dynamic["dims"]]
        cells = [value(cell) for cell in dynamic["cells"]]
        if not (isinstance(length, int) and 0 <= length <= len(cells)):
            raise ValueError(f"dynamic length {length} outside [0, {len(cells)}]")
        if math.prod(dims) != length:
            raise ValueError(f"dynamic dims {dims} do not multiply to the length {length}")
        values.extend(cells[:length])
        shapes.append(dims)
    if sorted(used) != list(range(len(fields))):
        raise ValueError("the layout does not account for every public value exactly once")
    return values, shapes


def decode_exposed(fields: list[int], kinds: list[str]) -> list[int | float]:
    """Decode field elements by Zinnia expose kind: `i` signed integer, `f` Q32 float."""
    if len(fields) != len(kinds):
        raise ValueError(f"{len(fields)} public values but {len(kinds)} exposed outputs")
    decoded: list[int | float] = []
    for value, kind in zip(fields, kinds):
        signed = field_to_signed(value)
        decoded.append(q32_to_float(signed) if kind == "f" else signed)
    return decoded


# ---------------------------------------------------------------------------
# The reference side
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    """One element of the kernel's flattened output."""

    value: int | float
    kind: str  # "int" | "float"


def _parts(result: Any) -> tuple:
    """Top-level output parts, flattened the way Zinnia flattens a return value."""
    if isinstance(result, tuple):
        return result
    if isinstance(result, list) and any(isinstance(p, (np.ndarray, list, tuple)) for p in result):
        return tuple(result)
    return (result,)


def reference_leaves(result: Any) -> tuple[list[Leaf], list[list[int]]]:
    """The reference output as typed leaves (row-major) and the shape of each part."""
    leaves: list[Leaf] = []
    shapes: list[list[int]] = []
    for part in _parts(result):
        if isinstance(part, (list, tuple)) and any(isinstance(p, (np.ndarray, list, tuple))
                                                  for p in part):
            nested, nested_shapes = reference_leaves(tuple(part))
            leaves.extend(nested)
            shapes.extend(nested_shapes)
            continue
        array = np.asarray(part)
        shapes.append(list(array.shape))
        for value in array.ravel():
            leaves.extend(_leaves_of_scalar(value, array.dtype.kind))
    return leaves, shapes


def _leaves_of_scalar(value: Any, dtype_kind: str) -> list[Leaf]:
    if dtype_kind == "c":  # Zinnia exposes a complex number as (real, imag)
        return [Leaf(float(value.real), "float"), Leaf(float(value.imag), "float")]
    if dtype_kind == "f":
        return [Leaf(float(value), "float")]
    if dtype_kind in "iub":
        return [Leaf(int(value), "int")]
    raise ValueError(f"output of NumPy kind {dtype_kind!r} cannot be checked")


def leaves_to_json(leaves: list[Leaf]) -> list[list]:
    return [[leaf.value, leaf.kind] for leaf in leaves]


def leaves_from_json(raw: list[list]) -> list[Leaf]:
    return [Leaf(value, kind) for value, kind in raw]


# ---------------------------------------------------------------------------
# The proof side
# ---------------------------------------------------------------------------


@dataclass
class DecodedPublics:
    """A proof's public outputs as an adapter reports them (see module docstring)."""

    layout: str
    values: list[Any]
    elided: list[Any] = field(default_factory=list)
    #: Shape of each output part, when the system reports it (`logical` only).
    shapes: list[list[int]] | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(raw: dict) -> "DecodedPublics":
        return DecodedPublics(layout=raw["layout"], values=list(raw["values"]),
                              elided=list(raw.get("elided", [])), shapes=raw.get("shapes"))


@dataclass
class OutputCheck:
    status: str  # "match" | "mismatch" | "unchecked"
    detail: str
    expected_count: int
    actual_count: int
    mismatches: list[list] = field(default_factory=list)  # [index, expected, actual]
    tolerance: dict | None = None

    @property
    def matched(self) -> bool:
        return self.status == "match"

    def to_json(self) -> dict:
        return asdict(self)


def _normalize(value: Any) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):  # some programs print big integers as strings
        return int(value) if value.lstrip("-").isdigit() else float(value)
    raise ValueError(f"public value {value!r} is not a number")


def _leaf_matches(leaf: Leaf, actual: int | float, tolerance: Tolerance) -> bool:
    if leaf.kind == "int":
        # Exact: an integer output decoded as a fixed-point float must be integral.
        if isinstance(actual, int):
            return actual == leaf.value
        return float(actual).is_integer() and int(actual) == leaf.value
    if not math.isfinite(leaf.value):
        return False  # not representable in fixed point; never a silent pass
    return tolerance.allows(float(leaf.value), float(actual))


def compare_positional(expected: list[Leaf], actual: list[Any], tolerance: Tolerance,
                       how: str) -> OutputCheck:
    """Element-by-element comparison in output order."""
    values = [_normalize(v) for v in actual]
    tol = asdict(tolerance)
    if len(values) != len(expected):
        return OutputCheck("mismatch", f"{how}: {len(values)} values for {len(expected)} "
                           "output elements", len(expected), len(values), tolerance=tol)
    mismatches = [[i, leaf.value, value] for i, (leaf, value) in enumerate(zip(expected, values))
                  if not _leaf_matches(leaf, value, tolerance)]
    status = "mismatch" if mismatches else "match"
    detail = f"{how}: {len(mismatches)} of {len(expected)} elements differ"
    return OutputCheck(status, detail, len(expected), len(values),
                       mismatches[:MAX_REPORTED_MISMATCHES], tol)


def _unchecked(reason: str, expected: list[Leaf], publics: DecodedPublics) -> OutputCheck:
    return OutputCheck("unchecked", reason, len(expected), len(publics.values))


def check_outputs(expected: list[Leaf], publics: DecodedPublics, tolerance: Tolerance,
                  expected_shapes: list[list[int]] | None = None) -> OutputCheck:
    """Compare a proof's public outputs with the reference (see module docstring).

    When the system reports the shape of each output part, it must equal the
    reference's (`expected_shapes`): a runtime reshape can keep the flattened
    values and still be wrong.
    """
    if publics.layout == LOGICAL:
        if publics.shapes is not None and expected_shapes is not None \
                and [list(s) for s in publics.shapes] != expected_shapes:
            return OutputCheck("mismatch", f"output shapes {publics.shapes}, reference "
                               f"{expected_shapes}", len(expected), len(publics.values))
        return compare_positional(expected, publics.values, tolerance, "logical outputs")
    if publics.layout == ZINNIA_EXPOSURE:
        return _check_zinnia_exposure(expected, publics, tolerance)
    if publics.layout == FIELD_ELEMENTS:
        if len(publics.values) != len(expected):
            return _unchecked(f"{len(publics.values)} field elements for {len(expected)} "
                              "outputs and no layout to decode them", expected, publics)
        decoded = [q32_to_float(int(v)) if leaf.kind == "float" else int(v)
                   for v, leaf in zip(publics.values, expected)]
        return compare_positional(expected, decoded, tolerance,
                                  "field elements decoded by reference type")
    raise ValueError(f"unknown public layout {publics.layout!r}")


def _check_zinnia_exposure(expected: list[Leaf], publics: DecodedPublics,
                           tolerance: Tolerance) -> OutputCheck:
    exposed, elided = publics.values, publics.elided
    if not elided and len(exposed) == len(expected):
        return compare_positional(expected, exposed, tolerance, "every output exposed")
    if not exposed and len(elided) == len(expected):
        return compare_positional(expected, elided, tolerance,
                                  "every output a compile-time constant")
    return _unchecked(
        f"{len(exposed)} exposed and {len(elided)} elided values for {len(expected)} output "
        "elements; the compiler does not export where elided values and dynamic-shape "
        "metadata sit in the output, so the order cannot be checked", expected, publics)
