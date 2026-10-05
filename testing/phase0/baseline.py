"""The baseline artifacts of `revision/phase0/acceptance.md` §3.1.

Consistency compares phase 0's record against what the baseline compiler
answered for the same program. This module produces the baseline half: it
compiles one parity circuit with `ZKCircuit.from_method(fn).compile()` and
reduces the result to the four artifacts §3.1 names.

| Artifact | Where it comes from |
|---|---|
| input descriptors | `program_inputs[i].export()`, plus the *declared* annotation |
| static allocations | `AllocateMemoryIR` statements (`segment_id`, `size`) |
| dynamic envelopes | `AllocateDynamicNDArrayMetaIR` (`max_length`, `max_shape`) |
| return exposure | `ExposePublicI/F` statement counts and the elided publics |
| enforced checks | the `Assert*` / `StructuralPredicate` / `ScalarPrecondition` statements |
| `allowed_meta` | the oracle of the retired `benchmarking/audit_bindings.py` (kept below), for a DS-1000 case with an input fixture |

Two facts about the baseline shape this reduction, and `consistency/`
relies on both:

* **The baseline allocates no static array.** An `NDArray` / `StaticArray`
  value is a vector of SSA cells, so the only static allocation in the IR is
  an `AllocateMemoryIR` *RAM segment*, which carries a flat `size` and no
  shape and exists only for an array the program indexes dynamically. The
  static-shape comparison is therefore a comparison of flat sizes.
* **A fact the baseline discharged leaves no statement.** `visit_while`'s
  guard resolution, `resolve_shape_type`'s exact/upper SMT queries and the
  constant folding of an index all run inside `IRGenerator` and are consumed
  there; the IR records only the *outcome* (a shorter statement list, a
  concrete shape) and the checks the baseline could **not** discharge, which
  it emitted as runtime assertions. See `consistency/`, "what cannot be
  extracted".

Compiling is slow — seconds for DS-1000, minutes for NPBench and Pythran —
so every record is cached under
`revision/phase0/results/baseline_cache/<program>__<function>.json` and a
re-run recompiles nothing unless `refresh=True`.

Isolation is `corpus.run_in_child`, shared with the coverage and adequacy
harnesses: the compile runs in a spawned child with a hard address-space
limit and a wall-clock timeout, and this process never compiles. A child that
dies or outlives its budget becomes a `memory_exceeded` / `timeout` record
rather than taking the run with it.

This module also produces the *other* half, phase 0's record, under the same
isolation (:func:`phase0_record_isolated`). It is not cached: a record of an
unrolled program runs to tens of megabytes of JSON, which is not something to
keep 187 copies of on disk.

Run as a script to fill the cache:

```sh
MEMRUN_MAX=12G .../memrun.sh .venv/bin/python testing/phase0/baseline.py
MEMRUN_MAX=12G .../memrun.sh .venv/bin/python testing/phase0/baseline.py --refresh
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# The package's `__init__` puts the repository root on `sys.path`; that
# cannot run before the package itself is importable, so a module that is
# also a script does it first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from testing.phase0 import REPO_ROOT, RESULTS  # noqa: E402
from testing.phase0.corpus import (  # noqa: E402
    ChildLimits,
    Circuit,
    first_line,
    limit_address_space,
    load_module,
    parity_set,
    run_in_child,
)

BASELINE_CACHE = RESULTS / "baseline_cache"

#: Address space one baseline compile or one phase 0 analysis may use.
ADDRESS_SPACE_LIMIT = 8 * 1024**3

#: Wall-clock seconds one baseline compile may take. The slowest parity
#: program recorded in `coverage.md` §9 took 117 s; this leaves headroom
#: without letting one program hold the sweep for an hour.
COMPILE_TIMEOUT_SECONDS = 600.0

#: Wall-clock seconds one phase 0 analysis may take, as in `test_adequacy.py`.
ANALYZE_TIMEOUT_SECONDS = 300.0

#: This module's isolation budget (`corpus.run_in_child`): the compile and
#: the analysis get the same address space and their own timeouts.
COMPILE_LIMITS = ChildLimits(ADDRESS_SPACE_LIMIT, COMPILE_TIMEOUT_SECONDS)

#: IR classes that are a check the baseline could not discharge statically.
ENFORCED_CHECK_CLASSES = (
    "AssertIR",
    "AssertBitIR",
    "AssertNonNegIIR",
    "AssertU128LtIR",
    "AssertU128LtBitIR",
    "AssertDynamicNDArrayMetaIR",
    "StructuralPredicateIR",
    "ScalarPreconditionIR",
)

#: `ds1000_case0303` -> `0303`, the key `INPUT_BUILDERS` uses.
_DS1000_CASE = re.compile(r"^ds1000_case(\d+)$")


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineInput:
    """One circuit parameter, as the baseline resolved it.

    `declared` is the annotation the front end parsed (a `SymbolicNDArray`
    when the shape names symbols); `resolved` is what
    `ZKProgramInput.export()["dt"]` carries after
    `zinnia_compiler._resolved_program_input_dtypes` has applied the
    `scalar_requires` facts, which is the envelope the circuit was built for.
    """

    name: str
    kind: str
    declared: Any
    resolved: Any


@dataclass(frozen=True)
class StaticAllocation:
    """An `AllocateMemoryIR`: a RAM segment of `size` flat cells."""

    stmt_id: int
    segment_id: int
    size: int
    init_value: int


@dataclass(frozen=True)
class DynamicAllocation:
    """An `AllocateDynamicNDArrayMetaIR`, with the parameter it witnesses.

    `param_name` is the `WitnessDynamicNDArrayMetaIR` for the same
    `array_id`, which is how a dynamic array is tied back to a circuit input;
    it is `None` for an array the program produced itself.
    """

    stmt_id: int
    array_id: int
    dtype_name: str
    max_length: int
    max_rank: int
    max_shape: list[int]
    param_name: str | None


@dataclass(frozen=True)
class ReadFootprint:
    """How much of one input the circuit reads, from the read statements.

    `shape` is the per-axis maximum index plus one over every
    `ReadInteger` / `ReadFloat` / `ReadHash` path naming the parameter — the
    rectangle the baseline addressed, which is at most the declared shape.
    """

    param: str
    paths: int
    shape: list[int]


@dataclass(frozen=True)
class ReturnExposure:
    """The return payload the circuit binds, in cells.

    A return cell is either exposed as a public (`ExposePublicI` /
    `ExposePublicF`) or elided because it is a compile-time constant, and
    `get_elided_publics()` reports the elided ones with their type. The two
    are split by type because the *float* cells are pure payload while the
    *integer* cells mix payload with the runtime shape metadata a dynamic
    return exposes (`expose_dynamic_ndarray_return`); only the float count is
    an envelope the harness can compare against.
    """

    public_int: int
    public_float: int
    elided_int: int
    elided_float: int

    @property
    def float_cells(self) -> int:
        return self.public_float + self.elided_float


@dataclass
class BaselineRecord:
    """Everything `consistency/` reads about one baseline compile.

    `status` is `compiled`, `compile_error`, `timeout` or `memory_exceeded`;
    every field below `status` is empty unless the status is `compiled`.
    """

    program: str
    function: str
    status: str
    error_type: str | None = None
    error_message: str | None = None
    compile_seconds: float | None = None
    ir_statements: int = 0
    inputs: list[BaselineInput] = field(default_factory=list)
    static_allocations: list[StaticAllocation] = field(default_factory=list)
    dynamic_allocations: list[DynamicAllocation] = field(default_factory=list)
    read_footprints: list[ReadFootprint] = field(default_factory=list)
    exposure: ReturnExposure | None = None
    enforced_checks: dict[str, int] = field(default_factory=dict)
    ir_classes: dict[str, int] = field(default_factory=dict)
    #: The `allowed_meta` oracle of `audit_bindings.py`, sorted; `None` when
    #: the oracle is unavailable, with `allowed_meta_reason` saying why.
    allowed_meta: list[int] | None = None
    allowed_meta_reason: str | None = None

    @property
    def label(self) -> str:
        return f"{self.program}::{self.function}"

    @property
    def ok(self) -> bool:
        return self.status == "compiled"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(payload: dict[str, Any]) -> "BaselineRecord":
        """Rebuild a record from its cached JSON."""
        return BaselineRecord(
            program=payload["program"],
            function=payload["function"],
            status=payload["status"],
            error_type=payload.get("error_type"),
            error_message=payload.get("error_message"),
            compile_seconds=payload.get("compile_seconds"),
            ir_statements=payload.get("ir_statements", 0),
            inputs=[BaselineInput(**row) for row in payload.get("inputs", [])],
            static_allocations=[
                StaticAllocation(**row) for row in payload.get("static_allocations", [])
            ],
            dynamic_allocations=[
                DynamicAllocation(**row)
                for row in payload.get("dynamic_allocations", [])
            ],
            read_footprints=[
                ReadFootprint(**row) for row in payload.get("read_footprints", [])
            ],
            exposure=(
                ReturnExposure(**payload["exposure"])
                if payload.get("exposure") is not None
                else None
            ),
            enforced_checks=payload.get("enforced_checks", {}),
            ir_classes=payload.get("ir_classes", {}),
            allowed_meta=payload.get("allowed_meta"),
            allowed_meta_reason=payload.get("allowed_meta_reason"),
        )


# ---------------------------------------------------------------------------
# Reducing the IR to the artifacts of §3.1
# ---------------------------------------------------------------------------


def _ir_data(stmt: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    instance = stmt["ir_instance"]
    return instance["__class__"], instance["ir_data"]


def static_allocations(stmts: list[dict[str, Any]]) -> list[StaticAllocation]:
    """Every `AllocateMemoryIR`, in statement order."""
    out = []
    for stmt in stmts:
        name, data = _ir_data(stmt)
        if name == "AllocateMemoryIR":
            out.append(
                StaticAllocation(
                    stmt_id=int(stmt["stmt_id"]),
                    segment_id=int(data["segment_id"]),
                    size=int(data["size"]),
                    init_value=int(data["init_value"]),
                )
            )
    return out


def dynamic_allocations(stmts: list[dict[str, Any]]) -> list[DynamicAllocation]:
    """Every `AllocateDynamicNDArrayMetaIR`, tied to its witnessed parameter."""
    witnessed: dict[int, str] = {}
    for stmt in stmts:
        name, data = _ir_data(stmt)
        if name == "WitnessDynamicNDArrayMetaIR":
            witnessed[int(data["array_id"])] = data["param_name"]
    out = []
    for stmt in stmts:
        name, data = _ir_data(stmt)
        if name != "AllocateDynamicNDArrayMetaIR":
            continue
        array_id = int(data["array_id"])
        out.append(
            DynamicAllocation(
                stmt_id=int(stmt["stmt_id"]),
                array_id=array_id,
                dtype_name=data["dtype_name"],
                max_length=int(data["max_length"]),
                max_rank=int(data["max_rank"]),
                max_shape=[int(d) for d in data.get("max_shape", [])],
                param_name=witnessed.get(array_id),
            )
        )
    return out


def read_footprints(stmts: list[dict[str, Any]]) -> list[ReadFootprint]:
    """The rectangle of each input the read statements address."""
    counts: Counter[str] = Counter()
    extents: dict[str, list[int]] = {}
    for stmt in stmts:
        name, data = _ir_data(stmt)
        if name not in ("ReadIntegerIR", "ReadFloatIR", "ReadHashIR"):
            continue
        path = data["path"]
        param = path["param"]
        counts[param] += 1
        shape = extents.setdefault(param, [])
        for axis, segment in enumerate(path.get("segments", [])):
            index = segment.get("Index")
            if index is None:  # a field or a slice segment carries no extent
                continue
            while len(shape) <= axis:
                shape.append(0)
            shape[axis] = max(shape[axis], int(index) + 1)
    return [
        ReadFootprint(param=param, paths=counts[param], shape=extents.get(param, []))
        for param in sorted(counts)
    ]


def return_exposure(
    stmts: list[dict[str, Any]], elided: list[dict[str, Any]]
) -> ReturnExposure:
    """The return payload the circuit binds.

    `elided` is `get_elided_publics()`: a list of `{"Int": v}` / `{"Float": v}`
    dicts in return order, the compiler's claim of which output cells the
    kernel text itself pins.
    """
    classes = Counter(_ir_data(stmt)[0] for stmt in stmts)
    kinds = Counter(key for entry in elided for key in entry)
    return ReturnExposure(
        public_int=classes.get("ExposePublicIIR", 0),
        public_float=classes.get("ExposePublicFIR", 0),
        elided_int=kinds.get("Int", 0),
        elided_float=kinds.get("Float", 0),
    )


def enforced_checks(stmts: list[dict[str, Any]]) -> dict[str, int]:
    """The checks the baseline emitted because it could not discharge them."""
    classes = Counter(_ir_data(stmt)[0] for stmt in stmts)
    return {name: classes[name] for name in ENFORCED_CHECK_CLASSES if classes[name]}


# ---------------------------------------------------------------------------
# The `allowed_meta` oracle
# ---------------------------------------------------------------------------


def ds1000_case(program: str) -> str | None:
    """`ds1000_case0303` -> `0303`; `None` for any other program."""
    match = _DS1000_CASE.match(program)
    return match.group(1) if match else None


#: The June 2026 hand-written DS-1000 inputs (`INPUT_BUILDERS`) live in the
#: archived runner that defined them; the oracle below keeps using them so its
#: answers do not change now that `audit_bindings.py` is retired.
INPUT_BUILDERS_MODULE = (REPO_ROOT / "benchmarking" / "archive" / "2026-06-agent-sweeps"
                         / "runners" / "run_halo2_baselines.py")

#: The oracle's ground truth for kernels whose body is not plain NumPy, carried
#: over unchanged from `audit_bindings.GROUND_TRUTH_OVERRIDES`: 0344 calls
#: `list(Integer)`, which Zinnia defines as a one-element list.
def _ground_truth_0344(mystr: Any) -> Any:
    import numpy as np

    return np.array([int(mystr)], dtype=int)


_GROUND_TRUTH_OVERRIDES = {"0344": _ground_truth_0344}


def _input_builders() -> dict[str, Any]:
    """`INPUT_BUILDERS` of the archived runner, imported by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("zinnia_june_input_builders",
                                                  INPUT_BUILDERS_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {INPUT_BUILDERS_MODULE}")
    module = importlib.util.module_from_spec(spec)
    # Registered before executing: the module defines dataclasses, which look
    # their module up in `sys.modules`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.INPUT_BUILDERS


def _coerce_builder_value(value: Any, dt_spec: Any) -> Any:
    """An `INPUT_BUILDERS` value as the kernel argument `audit_bindings` passed
    (its `_coerce_arg`, unchanged): the builders spell integers as strings and
    arrays as flat lists; the resolved annotation gives the shape and dtype."""
    import numpy as np

    target_shape, target_dtype = None, None
    if isinstance(dt_spec, dict) and "NDArray" in dt_spec:
        target_shape = tuple(dt_spec["NDArray"].get("shape", []))
        target_dtype = dt_spec["NDArray"].get("dtype", None)
    elif isinstance(dt_spec, dict) and "Scalar" in dt_spec:
        target_dtype = dt_spec["Scalar"].get("dtype", None)
    if isinstance(value, list):
        if not value:
            arr = np.array([], dtype=np.float64)
        elif isinstance(value[0], (str, bool, int)):
            arr = np.array([int(v) for v in value], dtype=np.int64)
        elif isinstance(value[0], float):
            arr = np.array(value, dtype=np.float64)
        else:
            arr = np.asarray(value)
        if target_shape:
            arr = arr.reshape(target_shape)
        if target_dtype == "Float":
            arr = arr.astype(np.float64)
        elif target_dtype in ("Integer", "Boolean"):
            arr = arr.astype(np.int64)
        return arr
    if isinstance(value, str):
        return float(value) if target_dtype == "Float" else int(value)
    return float(value) if target_dtype == "Float" else value


def _shape_metadata(result: Any) -> set[int]:
    """Integers that may appear as shape metadata of `result`: 0, and every
    part's size, rank, per-axis extents and product of extents."""
    import numpy as np

    if isinstance(result, tuple):
        parts = result
    elif isinstance(result, list) and any(isinstance(p, np.ndarray) for p in result):
        parts = tuple(result)
    else:
        parts = (result,)
    meta = {0}
    for part in parts:
        array = np.asarray(part)
        meta.update({int(array.size), int(array.ndim), *(int(d) for d in array.shape)})
        if array.ndim >= 1:
            meta.add(int(np.prod(array.shape)))
    return meta


def _fixture_argument(fixture: dict[str, Any], name: str) -> Any:
    """The fixture entry for parameter `name`, with `audit_one`'s fallbacks.

    The DS-1000 input builders were written for the halo2 baselines and do
    not always spell a parameter the way the kernel does; `audit_one` accepted
    four spellings and this must accept the same four, or the oracle would be
    unavailable for cases the audit covered.
    """
    for key in (name, name.lower(), name.upper(), f"{name}_val", f"{name.lower()}_val"):
        if key in fixture:
            return fixture[key]
    raise KeyError(name)


def allowed_meta_for(program: str, function: str, program_inputs: list) -> tuple[
    list[int] | None, str | None
]:
    """The `allowed_meta` oracle for one circuit, or the reason it is absent.

    The oracle runs the kernel body as plain NumPy on the DS-1000 input
    fixture and collects the integers that may legitimately appear as shape
    metadata of the return value: the size, the rank and every per-axis
    extent. It needs an input fixture, so it exists only for a DS-1000 case
    with an `INPUT_BUILDERS` entry and only for the `kernel` entry point.
    """
    from benchmarking.harness.reference import run_kernel_as_numpy
    from testing.phase0.corpus import DATASET_DIRS

    case = ds1000_case(program)
    if case is None:
        return None, f"no NumPy ground-truth fixture: {program} is not a DS-1000 case"
    if function != "kernel":
        return None, f"the oracle builds inputs for `kernel`, not `{function}`"
    builders = _input_builders()
    if case not in builders:
        return None, f"the June 2026 input builders have no entry for case {case}"
    fixture = builders[case]()
    try:
        args = [
            _coerce_builder_value(_fixture_argument(fixture, pi.name), pi.export().get("dt"))
            for pi in program_inputs
        ]
    except KeyError as exc:
        return None, f"the input fixture for case {case} has no entry for {exc}"
    override = _GROUND_TRUTH_OVERRIDES.get(case)
    kernel_path = DATASET_DIRS["ds1000"] / program / "kernel.py"
    expected = override(*args) if override else run_kernel_as_numpy(kernel_path, "kernel", args)
    return sorted(int(value) for value in _shape_metadata(expected)), None


# ---------------------------------------------------------------------------
# Compiling one circuit in a child process
# ---------------------------------------------------------------------------


def _declared_inputs(zk_circuit: Any) -> dict[str, Any]:
    """The annotation of each parameter, before the baseline resolves it."""
    from zinnia.compile.zinnia_compiler import ZinniaCompiler

    ast_dict = ZinniaCompiler.circuit_ast_parse(zk_circuit.source, zk_circuit.name)
    return {
        inp["name"]: inp.get("annotation", {}).get("dt")
        for inp in ast_dict.get("inputs", [])
    }


def reduce_compiled(
    program: str,
    function: str,
    compiled: Any,
    declared: dict[str, Any],
    seconds: float,
) -> BaselineRecord:
    """One compiled program reduced to the artifacts of §3.1."""
    stmts = compiled.get_ir_stmts()
    return BaselineRecord(
        program=program,
        function=function,
        status="compiled",
        compile_seconds=seconds,
        ir_statements=len(stmts),
        inputs=[
            BaselineInput(
                name=pi.name,
                kind=pi.kind,
                declared=declared.get(pi.name),
                resolved=pi.export()["dt"],
            )
            for pi in compiled.get_program_inputs()
        ],
        static_allocations=static_allocations(stmts),
        dynamic_allocations=dynamic_allocations(stmts),
        read_footprints=read_footprints(stmts),
        exposure=return_exposure(stmts, compiled.get_elided_publics()),
        enforced_checks=enforced_checks(stmts),
        ir_classes=dict(Counter(_ir_data(stmt)[0] for stmt in stmts)),
    )


def _compile_worker(kernel_path: str, program: str, function: str, queue) -> None:
    """Compile one circuit and put its :class:`BaselineRecord` JSON on `queue`."""
    limit_address_space(ADDRESS_SPACE_LIMIT)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from zinnia import ZKCircuit

        module = load_module(Path(kernel_path))
        zk_circuit = ZKCircuit.from_method(getattr(module, function))
        declared = _declared_inputs(zk_circuit)
        started = time.time()
        compiled = zk_circuit.compile()
        record = reduce_compiled(
            program, function, compiled, declared, time.time() - started
        )
    except BaseException as exc:  # PyO3 panics inherit from BaseException
        queue.put(
            asdict(
                BaselineRecord(
                    program=program,
                    function=function,
                    status="compile_error",
                    error_type=type(exc).__name__,
                    error_message=first_line(exc, 500),
                )
            )
        )
        return
    try:
        record.allowed_meta, record.allowed_meta_reason = allowed_meta_for(
            program, function, compiled.get_program_inputs()
        )
    except BaseException as exc:
        record.allowed_meta_reason = (
            f"the oracle raised {type(exc).__name__}: {first_line(exc, 200)}"
        )
    queue.put(asdict(record))


def compile_isolated(circuit: Circuit) -> BaselineRecord:
    """Compile one parity circuit with the baseline, in its own process."""
    outcome = run_in_child(
        _compile_worker,
        (str(circuit.kernel_path), circuit.program, circuit.function),
        COMPILE_LIMITS,
    )
    if outcome.answer is not None:
        return BaselineRecord.from_json(outcome.answer)
    # A child that died without answering ran out of address space, whether
    # the kernel or the Rust allocator noticed first; either way the record
    # says the compile did not fit.
    return BaselineRecord(
        program=circuit.program,
        function=circuit.function,
        status="timeout" if outcome.status == "timeout" else "memory_exceeded",
        error_message=outcome.detail,
    )


def cache_path(program: str, function: str) -> Path:
    return BASELINE_CACHE / f"{program}__{function}.json"


def baseline_record(circuit: Circuit, refresh: bool = False) -> BaselineRecord:
    """The baseline record for one circuit, from the cache when possible."""
    path = cache_path(circuit.program, circuit.function)
    if not refresh and path.exists():
        return BaselineRecord.from_json(json.loads(path.read_text(encoding="utf-8")))
    record = compile_isolated(circuit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_json(), indent=2) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# Phase 0's record, under the same isolation
# ---------------------------------------------------------------------------


def _analyze_worker(kernel_path: str, function: str, out_path: str, queue) -> None:
    """Write phase 0's record for one circuit to `out_path` as JSON.

    The record crosses the process boundary as a file rather than through the
    queue: an unrolled program's record runs to tens of megabytes, and
    pickling that through a pipe is both slower and harder to bound than a
    write the parent reads back.
    """
    limit_address_space(ADDRESS_SPACE_LIMIT)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from zinnia import ZKCircuit

        module = load_module(Path(kernel_path))
        record = ZKCircuit.from_method(getattr(module, function)).analyze_shapes()
    except BaseException as exc:
        queue.put(
            {
                "status": "rejected",
                "reason": "harness_" + type(exc).__name__,
                "detail": first_line(exc, 300),
            }
        )
        return
    rejection = record.get("rejected")
    if rejection is not None:
        queue.put({"status": "rejected", **rejection})
        return
    Path(out_path).write_text(json.dumps(record), encoding="utf-8")
    queue.put({"status": "ok", "path": out_path})


def phase0_record_isolated(
    circuit: Circuit, timeout_seconds: float = ANALYZE_TIMEOUT_SECONDS
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Phase 0's record for one circuit, plus a status dict.

    Returns `(record, status)`. `record` is `None` unless
    `status["status"]` is `ok`; the other statuses are `rejected` (with
    `reason` and `detail`), `timeout` and `memory_exceeded`, exactly as
    `corpus.analyze_isolated` reports them.
    """
    limits = ChildLimits(ADDRESS_SPACE_LIMIT, timeout_seconds)
    with tempfile.TemporaryDirectory(prefix="phase0-record-") as tmp:
        out_path = Path(tmp) / f"{circuit.program}__{circuit.function}.json"
        outcome = run_in_child(
            _analyze_worker,
            (str(circuit.kernel_path), circuit.function, str(out_path)),
            limits,
        )
        if outcome.answer is None:
            status = "memory_exceeded" if outcome.status == "crashed" else outcome.status
            return None, {
                "status": status,
                "reason": status,
                "detail": outcome.detail,
            }
        if outcome.answer["status"] != "ok":
            return None, outcome.answer
        return json.loads(out_path.read_text(encoding="utf-8")), {"status": "ok"}


# ---------------------------------------------------------------------------
# Filling the cache
# ---------------------------------------------------------------------------


def fill_cache(circuits: list[Circuit], refresh: bool) -> list[BaselineRecord]:
    """Compile every circuit that is not cached and report as it goes."""
    records = []
    started = time.time()
    for index, circuit in enumerate(circuits, start=1):
        cached = cache_path(circuit.program, circuit.function).exists() and not refresh
        at = time.time()
        record = baseline_record(circuit, refresh=refresh)
        note = "cached" if cached else f"{time.time() - at:6.1f}s"
        print(
            f"[{index:3d}/{len(circuits)}] {record.status:16s} {note:>8s}  "
            f"{record.label}",
            flush=True,
        )
        records.append(record)
    print(f"total {time.time() - started:.1f}s")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="recompile every circuit, ignoring the cache",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="PROGRAM::FUNCTION",
        help="restrict to one circuit; repeatable",
    )
    args = parser.parse_args(argv)

    circuits = parity_set()
    if args.only:
        wanted = set(args.only)
        circuits = [c for c in circuits if c.label in wanted]
        missing = wanted - {c.label for c in circuits}
        if missing:
            parser.error(f"not in the parity set: {', '.join(sorted(missing))}")

    records = fill_cache(circuits, refresh=args.refresh)
    counts = Counter(record.status for record in records)
    print("baseline cache:", ", ".join(f"{n} {s}" for s, n in counts.most_common()))
    print(f"wrote {BASELINE_CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
