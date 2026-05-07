from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ExpectedStatus(str, Enum):
    PASS = "pass"
    XFAIL = "xfail"
    REJECT = "reject"
    UNCLEAR = "unclear"


class ArrayMode(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    MIXED = "mixed"


@dataclass(frozen=True)
class ShapeCase:
    id: str
    data: dict[str, Any]

    def expected_for_mode(self, mode: ArrayMode | None = None) -> ExpectedStatus | None:
        if mode is not None and mode.value in self.data.get("xfail_modes", []):
            return ExpectedStatus.XFAIL
        if "expected" in self.data:
            return ExpectedStatus(self.data["expected"])
        return None

    @property
    def reason(self) -> str | None:
        return self.data.get("reason")

    @property
    def replaces(self) -> str | None:
        return self.data.get("replaces")

    @property
    def spellings(self) -> tuple[str, ...] | None:
        values = self.data.get("spellings")
        if values is None:
            return None
        return tuple(values)

    @property
    def modes(self) -> tuple[ArrayMode, ...] | None:
        values = self.data.get("modes")
        if values is None:
            return None
        return tuple(ArrayMode(value) for value in values)

    @property
    def rank(self) -> int | None:
        shape = self.data.get("input") or self.data.get("lhs") or self.data.get("rank_from")
        if shape == "scalar" or shape is None:
            return 0 if shape == "scalar" else None
        return len(shape)

    @property
    def pattern(self) -> str:
        parts = []
        for key in ("input", "lhs", "rhs", "newshape", "axis", "index", "target", "rank_from"):
            if key in self.data:
                parts.append(f"{key}={self.data[key]}")
        return ";".join(parts)


@dataclass(frozen=True)
class OperatorCase:
    name: str
    kind: str
    spellings: tuple[str, ...]
    modes: tuple[ArrayMode, ...]
    dtypes: tuple[str, ...]
    shapes: tuple[ShapeCase, ...]
    expected: ExpectedStatus
    tags: tuple[str, ...] = field(default_factory=tuple)
    reason: str | None = None


def load_manifest(path: Path) -> list[OperatorCase]:
    raw = yaml.safe_load(path.read_text()) or {}
    operators = raw.get("operators", {})
    cases: list[OperatorCase] = []
    for name, entry in operators.items():
        shapes = []
        for idx, shape_data in enumerate(entry.get("shapes", [])):
            shape_id = shape_data.get("id", str(idx))
            shapes.append(ShapeCase(shape_id, dict(shape_data)))
        cases.append(
            OperatorCase(
                name=name,
                kind=entry["kind"],
                spellings=tuple(entry.get("spellings") or ("default",)),
                modes=tuple(ArrayMode(mode) for mode in entry.get("modes", ["static"])),
                dtypes=tuple(entry.get("dtypes", ["int"])),
                shapes=tuple(shapes),
                expected=ExpectedStatus(entry.get("expected", "pass")),
                tags=tuple(entry.get("tags", [])),
                reason=entry.get("reason"),
            )
        )
    return cases
