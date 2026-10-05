"""What every proving-system adapter provides.

An adapter runs inside a step child process (`benchmarking.harness.step`) and
implements three steps of the protocol:

- `setup` with input A: compile, generate keys; state goes to `ctx.work_dir`;
- `prove` with input B: one proof, written under `ctx.proof_dir`, with the two
  clocks of the protocol (`prove_core_s`, `end_to_end_s`) and the decoded
  public outputs;
- `verify`: verify `ctx.proof_path`; with `ctx.tamper`, first change one
  public value (the harness hands the adapter a copy of the proof directory).

`windows` documents, per clock, exactly what the adapter times; it is copied
into every raw record. Failures the adapter recognizes are raised as
`StepFailure` with a class from `benchmarking.harness.protocol.FAILURE_CLASSES`.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.fixtures import Fixture, load_fixture
from benchmarking.harness.outputs import DecodedPublics


class StepFailure(Exception):
    """A classified failure of one step."""

    def __init__(self, failure_class: str, detail: str):
        super().__init__(f"{failure_class}: {detail}")
        self.failure_class = failure_class
        self.detail = detail


@dataclass
class StepContext:
    case: PerfCase
    input_name: str
    work_dir: Path
    proof_dir: Path | None = None
    proof_path: Path | None = None
    tamper: bool = False
    options: dict = field(default_factory=dict)

    @cached_property
    def fixture(self) -> Fixture:
        return load_fixture(self.case.fixture_path)

    def arguments(self) -> list[Any]:
        return self.fixture.arguments(self.input_name)

    def state_path(self) -> Path:
        return self.work_dir / "setup_state.json"

    def save_state(self, state: dict) -> None:
        self.state_path().write_text(json.dumps(state, indent=1))

    def load_state(self) -> dict:
        if not self.state_path().exists():
            raise StepFailure("harness_error", "setup state missing (setup did not run?)")
        return json.loads(self.state_path().read_text())


@dataclass
class SetupResult:
    timings_s: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class ProveResult:
    #: The prover call alone, witness or trace already computed where the API allows.
    prove_core_s: float
    #: Witness generation or execution plus proving.
    end_to_end_s: float
    proof_path: str
    proof_bytes: int
    publics: DecodedPublics
    timings_s: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        data = asdict(self)
        data["publics"] = self.publics.to_json()
        data["proof_sha256"] = sha256_path(Path(self.proof_path))
        return data


@dataclass
class VerifyResult:
    verified: bool
    verify_s: float | None
    #: Set when a tamper check cannot apply (the proof has no public value).
    not_applicable: str | None = None
    timings_s: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def sha256_path(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Adapter(ABC):
    name: str = ""
    description: str = ""
    windows: dict[str, str] = {}

    def missing_program(self, case: PerfCase) -> str | None:
        """Why this system has no program for `case`, or `None` if it has one."""
        return None

    def program_files(self, case: PerfCase) -> list[Path]:
        """Files whose hashes identify this system's program for `case`."""
        return []

    @abstractmethod
    def setup(self, ctx: StepContext) -> SetupResult: ...

    @abstractmethod
    def prove(self, ctx: StepContext) -> ProveResult: ...

    @abstractmethod
    def verify(self, ctx: StepContext) -> VerifyResult: ...
