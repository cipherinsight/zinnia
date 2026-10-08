"""Scaled copies of the NPBench/Pythran performance entries, one set per cap.

For every `perf` entry of `datasets/case_sets/npbench_pythran_perf.json` and
every cap of `rule.CAPS`, the rule (`rule.py`) gives the entry's sizes; the
entry at that cap is one of

* `scaled`: new sizes. The kernel (the CAP-16 kernel with only the entry
  function's array annotations changed; bodies byte-identical), its fixture
  (inputs A and B from the same generators as the CAP-16 fixtures:
  `acceptance_fixtures.build`, NPBench's `initialize()` or the Pythran line,
  the same seeds) and the program's `zkvm.rs` (copied) are written to
  `DATA/cap<C>/<program>/`;
* `unchanged`: the rule gives the sizes of an earlier point (the performance
  set at CAP 16, or a smaller cap): the same program, not measured again;
* `fixed_size`: the rule cannot scale it (no size knob, `initialize()` or the
  example line fails at every divisor, the NumPy reference fails); the reason
  is recorded.

Outputs under DATA (a directory outside the checkout's dataset):

* `cap<C>/<program>/`: kernel.py, zkvm.rs, fixture[.<function>].json;
* `case_sets/npbench_pythran_cap<C>.json`: the harness case set of the cap,
  in the format of npbench_pythran_perf.json (tolerances and flags copied
  from the original entry; case ids `<case>_cap<C>`; absolute folder paths,
  see `relocate`); unchanged and fixed-size entries are listed with their
  status and reason, so the harness records them as excluded;
* `points.json`: every (entry, cap) with its status, sizes, divisor, input
  element count and, for CAP 16, how the rule's sizes compare with the
  performance set's.

The zkVM folder name is the program's name, so the SP1 and RISC Zero hosts
resolve the same guest (`<program>__<function>`) as for the original entry;
`zkvm_check.py` confirms the copied programs equal the originals.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarking.scaling_npb import rule

BENCHMARKING = Path(__file__).resolve().parents[1]
DATASETS = BENCHMARKING / "datasets"
PERF_SET = DATASETS / "case_sets" / "npbench_pythran_perf.json"
ACCEPTANCE_SET = DATASETS / "case_sets" / "acceptance.json"
SOURCES = DATASETS / "upstream_sources.json"

sys.path.insert(0, str(DATASETS / "tools"))
import acceptance_case_set as acs  # noqa: E402
import acceptance_fixtures as fixtures  # noqa: E402

SCALED, UNCHANGED, FIXED = "scaled", "unchanged", "fixed_size"


def case_set_path(data: Path, cap: int) -> Path:
    return data / "case_sets" / f"npbench_pythran_cap{cap}.json"


def scaled_case_id(case: str, cap: int) -> str:
    return f"{case}_cap{cap}"


# ---------------------------------------------------------------------------
# What the rule gives for one entry at one cap
# ---------------------------------------------------------------------------


@dataclass
class Sizing:
    """The entry's input A at one cap, by parameter name (B is the seeded perturbation of A,
    drawn when the fixture is written, as for the CAP-16 fixtures)."""

    knobs: dict[str, Any]
    divisor: int | None
    a: dict[str, Any]
    note: str
    largest: int | None = None


@dataclass
class Entry:
    """One performance entry with what its sizing needs."""

    perf: dict                      # the npbench_pythran_perf.json case
    acceptance: dict                # its acceptance.json entry
    fixture: dict                   # its CAP-16 fixture document
    upstream_source: str
    module_globals: dict | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.perf["program"], self.perf["function"]

    @property
    def folder(self) -> Path:
        return BENCHMARKING / self.perf["path"]

    @property
    def parameters(self) -> list[dict]:
        return self.fixture["parameters"]


class Sizer:
    """How the rule sizes one entry."""

    #: A reason when the rule keeps the entry at its CAP-16 size at every cap.
    fixed: str | None = None

    def at(self, cap: int) -> Sizing:
        raise NotImplementedError


class NpbenchSizer(Sizer):
    def __init__(self, entry: Entry, root: Path):
        self.entry, self.root = entry, root
        self.bench = acs.npbench_info(root, entry.acceptance["upstream"])
        if self.bench["short_name"] in acs.sizes.NPBENCH_T_OVERRIDES:
            raise ValueError(f"{self.bench['short_name']}: the acceptance override is CAP-16 specific")
        self.function = acs.function_defs(ast.parse(entry.upstream_source))[entry.perf["function"]]

    def at(self, cap: int) -> Sizing:
        def probe(candidate: dict) -> tuple[Any, str | None]:
            try:
                values = acs.npbench_values(self.root, self.bench, candidate)
            except Exception as exc:  # noqa: BLE001 - initialize() rejects these parameters
                return None, f"initialize() fails: {type(exc).__name__}: {exc}"
            return values, rule.within_cap([values[a] for a in self.bench["input_args"]], cap)

        choice = rule.choose(self.bench["parameters"]["S"], cap, probe)
        signature = acs.npbench_signature(self.bench, self.function, choice.values, f"cap {cap}")
        a = dict(signature.example)
        note = (f"A is NPBench's initialize() data at {choice.knobs} (preset S {self.bench['parameters']['S']} "
                f"through the scaling rule at CAP {cap}: divisor {choice.divisor}).")
        factors = fixtures.NPBENCH_Q32_INPUTS.get(self.bench["short_name"])
        if factors:
            for name, factor in factors[0].items():
                a[name] = a[name] * factor
            note += f" Changed for Q32 as at CAP 16: {factors[1]}."
        return Sizing(choice.knobs, choice.divisor, a, note)


def example_arguments(code: str, function: str, module_globals: dict) -> list[Any] | None:
    """The arguments of the first call of `function` in an example line, NumPy's global
    generator seeded with 0 first; None when the line does not reach the call."""
    np.random.seed(0)
    calls = acs.call_arguments(code, function, module_globals)
    return calls[0] if calls else None


def flatten_arguments(args: list[Any]) -> list[Any]:
    return [np.asarray(a) if isinstance(a, (list, tuple)) else a for a in args]


class PythranSizer(Sizer):
    def __init__(self, entry: Entry, line: rule.ExampleLine):
        self.entry, self.line = entry, line
        self.module = Path(entry.acceptance["upstream"]).stem

    def arguments(self, knobs: dict) -> list[Any] | None:
        return example_arguments(rule.render(self.line.template, knobs), self.entry.perf["function"],
                                 self.entry.module_globals)

    def at(self, cap: int) -> Sizing:
        def probe(candidate: dict) -> tuple[Any, str | None]:
            args = self.arguments(candidate)
            if args is None:
                return None, "the example line fails"
            return args, rule.within_cap(flatten_arguments(args), cap)

        choice = rule.choose_proportional(self.line.knobs, cap, probe)
        names = [p["name"] for p in self.entry.parameters]
        if len(choice.values) != len(names):
            raise rule.NotScalable(f"the line passes {len(choice.values)} arguments, the kernel takes {names}")
        code = rule.render(self.line.template, choice.knobs)
        note = (f"A is the arguments of the example call `{code}` (upstream "
                f"`{self.line.upstream}` with its size knobs {self.line.knobs} scaled in proportion at "
                f"CAP {cap}: largest knob {choice.largest}).")
        if self.line.q32_values:
            note += " The values come from the CAP-16 fixture's Q32 generator (acceptance_fixtures.py)."
        return Sizing(choice.knobs, None, dict(zip(names, choice.values)), note, choice.largest)

    def check_template(self) -> list[str]:
        """Problems with the template: it must give the upstream line's arguments at the
        upstream knob values (only their shapes when the values come from elsewhere or
        the upstream line draws unseeded)."""
        problems = []
        if not re.search(rf"^#\s*(runas|bench):?\s+{re.escape(self.line.upstream)}\s*$",
                         self.entry.upstream_source, re.M):
            problems.append("the upstream line is not a #runas/#bench line of the upstream file")
        recorded = (self.entry.acceptance.get("sizing") or {}).get("upstream_example")
        if recorded and recorded != acs.reproducible(self.line.upstream):
            problems.append(f"acceptance.json records another upstream example: {recorded}")
        function = self.entry.perf["function"]
        upstream = example_arguments(rule.prepare(self.line.upstream, self.module), function,
                                     self.entry.module_globals)
        mine = self.arguments(self.line.knobs)
        if upstream is None or mine is None:
            return problems + ["the upstream line or the template does not reach the call"]
        again = example_arguments(rule.prepare(self.line.upstream, self.module), function,
                                  self.entry.module_globals)
        compare_values = not self.line.q32_values and all(
            np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(upstream, again))
        for index, (x, y) in enumerate(zip(flatten_arguments(upstream), flatten_arguments(mine))):
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape or (compare_values and not np.array_equal(x, y)):
                problems.append(f"argument {index}: upstream {x.shape}, template {y.shape}"
                                + (" (values differ)" if x.shape == y.shape else ""))
        return problems


class KeptSizer(Sizer):
    """A Pythran entry the rule leaves at its CAP-16 sizes at every cap: its upstream
    example passes no array dimension above 16 (`reason` None), or a clause of the rule
    fixes its size (`reason`)."""

    def __init__(self, entry: Entry, reason: str | None = None):
        self.entry, self.reason = entry, reason


class FixedSizer(Sizer):
    def __init__(self, reason: str):
        self.fixed = reason


def sizer_for(entry: Entry, roots: dict[str, Path]) -> Sizer:
    if entry.perf["group"] == "npbench":
        return NpbenchSizer(entry, roots["npbench"])
    if entry.key in rule.PYTHRAN_FIXED:
        return FixedSizer(rule.PYTHRAN_FIXED[entry.key])
    if entry.key in rule.PYTHRAN_KEPT_BY_RULE:
        return KeptSizer(entry, rule.PYTHRAN_KEPT_BY_RULE[entry.key])
    if entry.key in rule.PYTHRAN_LINES:
        return PythranSizer(entry, rule.PYTHRAN_LINES[entry.key])
    sizing = entry.acceptance["sizing"]
    if not sizing["rule"].startswith("upstream example used as it is"):
        raise ValueError(f"{entry.key}: no rule for its CAP-16 sizing {sizing['rule']!r}")
    return KeptSizer(entry)


# ---------------------------------------------------------------------------
# Kernels and fixtures
# ---------------------------------------------------------------------------


def shape_of(value: Any) -> list[int]:
    return list(np.asarray(value).shape)


def annotations_for(entry: Entry, a: dict[str, Any]) -> dict[str, str]:
    """The entry function's annotations with the array shapes of input A; dtype and
    rank must stay those of the CAP-16 signature."""
    current = entry.acceptance["signature"]
    out = {}
    for parameter in entry.parameters:
        name = parameter["name"]
        annotation = current[name]
        if parameter["kind"] == "array":
            shape = shape_of(a[name])
            if len(shape) != len(parameter["shape"]):
                raise rule.NotScalable(f"{name}: rank {len(shape)}, the kernel has rank {len(parameter['shape'])}")
            annotation = re.sub(r'"[\d ]+"', '"' + " ".join(map(str, shape)) + '"', annotation)
        elif parameter["kind"] != "scalar":
            raise rule.NotScalable(f"{name}: {parameter['kind']} parameters are not supported")
        out[name] = annotation
    return out


def rewrite_kernel(source: str, function: str, annotations: dict[str, str]) -> str:
    """`source` with the shape strings of `function`'s array annotations replaced; every
    other byte stays."""
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for arg in node.args.args:
        if arg.annotation is None or "Array" not in ast.unparse(arg.annotation):
            continue
        start = offsets[arg.annotation.lineno - 1] + arg.annotation.col_offset
        end = offsets[arg.annotation.end_lineno - 1] + arg.annotation.end_col_offset
        shape = re.search(r'"([\d ]+)"', annotations[arg.arg]).group(1)
        edits.append((start, end, re.sub(r"(['\"])[\d ]+\1", lambda m: m.group(1) + shape + m.group(1),
                                         source[start:end])))
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source


def body_unchanged(old: str, new: str) -> bool:
    """Every function body (and everything but the entry annotations) is the same."""
    def key(text: str) -> list[str]:
        tree = acs.retained_code(ast.parse(text))
        return [acs.body_key(fn) for fn in acs.function_defs(tree).values()]
    return key(old) == key(new)


def input_elements(fixture: dict, input_name: str = "A") -> int:
    """How many numbers input `input_name` holds (a complex number counts once)."""
    total = 0
    for parameter in fixture["parameters"]:
        value = np.asarray(fixture["inputs"][input_name][parameter["name"]])
        total += value.size // (2 if parameter["dtype"] == "complex" else 1)
    return total


def size_key(parameters: list[dict], a: dict[str, Any]) -> list:
    """What makes two points the same program: the array shapes and integer scalars of A."""
    key = []
    for parameter in parameters:
        name = parameter["name"]
        if parameter["kind"] == "array":
            shape = shape_of(a[name])
            if parameter["dtype"] == "complex" and shape[-1:] == [2] and len(shape) > len(parameter["shape"]):
                shape = shape[:-1]          # fixture.json writes a complex number as a pair
            key.append([name, shape])
        elif parameter["dtype"] == "int":
            key.append([name, int(a[name])])
    return key


def fixture_key(fixture: dict) -> list:
    return size_key(fixture["parameters"], fixture["inputs"]["A"])


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


@dataclass
class Point:
    case: str
    program: str
    function: str
    group: str
    cap: int
    status: str
    reason: str | None = None
    same_as: int | None = None          # unchanged: the cap whose program it is (16 = perf set)
    knobs: dict | None = None
    divisor: int | None = None          # NPBench: d of the divisor formula
    largest: int | None = None          # Pythran: the largest knob's value (proportional formula)
    signature: dict | None = None
    input_elements: int | None = None
    fixture_notes: str | None = None
    folder: str | None = None           # scaled: relative to DATA
    fixture: str | None = None


def load_entries() -> list[Entry]:
    perf = json.loads(PERF_SET.read_text())
    acceptance = {(e["program"], e["function"]): e for e in json.loads(ACCEPTANCE_SET.read_text())["entries"]}
    entries = []
    for case in perf["cases"]:
        if case["status"] != "perf":
            continue
        accepted = acceptance[(case["program"], case["function"])]
        fixture = json.loads((BENCHMARKING / case["path"] / case["fixture"]).read_text())
        entries.append(Entry(case, accepted, fixture, ""))
    return entries


def attach_upstream(entries: list[Entry], roots: dict[str, Path]) -> None:
    globals_by_program: dict[str, dict] = {}
    for entry in entries:
        upstream = entry.acceptance["upstream"]
        if entry.perf["group"] == "npbench":
            entry.upstream_source = (roots["npbench"] / "npbench" / "benchmarks" / upstream).read_text()
        else:
            entry.upstream_source = (roots["pythran"] / "pythran" / "tests" / "cases" / upstream).read_text()
            if entry.perf["program"] not in globals_by_program:
                globals_by_program[entry.perf["program"]] = acs.pythran_module_globals(entry.upstream_source)
            entry.module_globals = globals_by_program[entry.perf["program"]]


def write_point(entry: Entry, cap: int, sizing: Sizing, data: Path) -> tuple[Point, dict]:
    """Write the scaled kernel, fixture and zkvm.rs; the point and its fixture."""
    program, function = entry.key
    folder = data / f"cap{cap}" / program
    kernel = folder / "kernel.py"
    created = not kernel.exists()
    if created:
        folder.mkdir(parents=True)
        kernel.write_text((entry.folder / "kernel.py").read_text())
        shutil.copyfile(entry.folder / "zkvm.rs", folder / "zkvm.rs")
    annotations = annotations_for(entry, sizing.a)
    before = kernel.read_text()
    after = rewrite_kernel(before, function, annotations)
    if not body_unchanged(before, after):
        raise RuntimeError(f"{entry.key}: rewriting the annotations changed the kernel")
    kernel.write_text(after)
    try:
        document = fixtures.build(program, function, kernel, annotations, sizing.a,
                                  fixtures.seed_for(program, function), sizing.note)
    except fixtures.FixtureError as exc:
        if created:
            shutil.rmtree(folder)
        else:
            kernel.write_text(before)
        raise rule.NotScalable(f"NumPy reference fails: {exc}") from exc
    fixtures.dump(folder / entry.perf["fixture"], document)
    point = Point(scaled_case_id(entry.perf["case"], cap), program, function, entry.perf["group"], cap, SCALED,
                  knobs=sizing.knobs, divisor=sizing.divisor, largest=sizing.largest, signature=annotations,
                  input_elements=input_elements(document), fixture_notes=document["notes"],
                  folder=str(folder.relative_to(data)), fixture=entry.perf["fixture"])
    return point, document


def reference_relation(entry: Entry, sizer: Sizer, scratch: Path) -> dict:
    """How the performance set's sizes (CAP 16) relate to the rule at CAP 16."""
    current = fixture_key(entry.fixture)
    sizing = entry.acceptance.get("sizing", {})
    if sizer.fixed:
        return {"relation": "fixed", "detail": sizer.fixed, "acceptance_rule": sizing.get("rule")}
    if isinstance(sizer, KeptSizer):
        if sizer.reason:
            return {"relation": "rule", "detail": sizer.reason, "acceptance_rule": sizing.get("rule")}
        line = sizing.get("example_before_q32_change", sizing["example"])
        args = example_arguments(rule.prepare(line, Path(entry.acceptance["upstream"]).stem),
                                 entry.perf["function"], entry.module_globals)
        arrays = [a for a in flatten_arguments(args or []) if isinstance(a, np.ndarray)]
        scalars = [int(a) for a in args or [] if rule.is_int(a)]
        fits = args is not None and rule.within_cap(arrays, rule.REFERENCE_CAP) is None
        detail = (f"the upstream example passes no array dimension above {rule.REFERENCE_CAP} (shapes "
                  f"{[shape_of(a) for a in arrays]}, integer scalars {scalars}, values rather than sizes); "
                  "the rule leaves it as it is at every cap")
        return {"relation": "rule" if fits else "differs",
                "detail": detail if fits else "the upstream example passes a dimension above the cap",
                "acceptance_rule": sizing.get("rule")}
    try:
        at16 = sizer.at(rule.REFERENCE_CAP)
        _, document = write_point(entry, rule.REFERENCE_CAP, at16, scratch)
    except rule.NotScalable as exc:
        return {"relation": "rule fails at 16", "detail": str(exc)}
    same_sizes = fixture_key(document) == current
    same_inputs = same_sizes and document["inputs"] == entry.fixture["inputs"]
    relation = "rule (identical fixture)" if same_inputs else "rule sizes" if same_sizes else "hand"
    return {"relation": relation, "rule_knobs": at16.knobs, "rule_divisor": at16.divisor,
            "rule_largest": at16.largest,
            "rule_signature": annotations_for(entry, at16.a), "current_signature": entry.acceptance["signature"],
            "acceptance_rule": sizing.get("rule"), "acceptance_change": sizing.get("change")}


def generate(data: Path, roots: dict[str, Path], caps: tuple[int, ...] = rule.CAPS) -> dict:
    entries = load_entries()
    attach_upstream(entries, roots)
    template_problems = {}
    points: list[Point] = []
    relations = {}
    with tempfile.TemporaryDirectory(prefix="scaling-npb-16-") as scratch:
        for entry in entries:
            case = entry.perf["case"]
            sizer = sizer_for(entry, roots)
            if isinstance(sizer, PythranSizer):
                problems = sizer.check_template()
                if problems:
                    template_problems[case] = problems
                    continue
            relations[case] = reference_relation(entry, sizer, Path(scratch))
            earlier = {rule.REFERENCE_CAP: fixture_key(entry.fixture)}
            for cap in caps:
                point = size_point(entry, sizer, cap, data, earlier)
                points.append(point)
                print(f"{point.case:<70} {point.status:<10} {point.reason or point.same_as or ''}", flush=True)
    if template_problems:
        raise SystemExit(f"templates that do not reproduce the upstream line: {template_problems}")
    write_case_sets(entries, points, data, caps)
    document = {"rule": rule.__doc__, "caps": list(caps),
                "upstream_roots": {k: str(v) for k, v in roots.items()},
                "upstream_commits": {k: acs.git_head(v) for k, v in roots.items()},
                "reference_cap": relations, "points": [asdict(p) for p in points]}
    (data / "points.json").write_text(json.dumps(document, indent=1, default=str) + "\n")
    return document


def size_point(entry: Entry, sizer: Sizer, cap: int, data: Path, earlier: dict[int, list]) -> Point:
    program, function = entry.key
    base = Point(scaled_case_id(entry.perf["case"], cap), program, function, entry.perf["group"], cap, FIXED)
    if sizer.fixed:
        base.reason = sizer.fixed
        return base
    if isinstance(sizer, KeptSizer):
        base.status, base.same_as, base.reason = UNCHANGED, rule.REFERENCE_CAP, sizer.reason
        base.input_elements = input_elements(entry.fixture)
        return base
    try:
        sizing = sizer.at(cap)
    except rule.NotScalable as exc:
        base.reason = str(exc)
        return base
    base.knobs, base.divisor, base.largest = sizing.knobs, sizing.divisor, sizing.largest
    key = size_key(entry.parameters, sizing.a)
    for previous, previous_key in earlier.items():
        if key == previous_key:
            base.status, base.same_as = UNCHANGED, previous
            return base
    try:
        point, _ = write_point(entry, cap, sizing, data)
    except rule.NotScalable as exc:
        base.reason = str(exc)
        return base
    earlier[cap] = key
    return point


def write_case_sets(entries: list[Entry], points: list[Point], data: Path, caps: tuple[int, ...]) -> None:
    by_case = {e.perf["case"]: e for e in entries}
    (data / "case_sets").mkdir(parents=True, exist_ok=True)
    for cap in caps:
        cases = []
        for point in (p for p in points if p.cap == cap):
            original = by_case[point.case.removesuffix(f"_cap{cap}")].perf
            case = {k: v for k, v in original.items() if k not in ("acceptance", "zkvm_port", "reason")}
            case.update(case=point.case, status="perf" if point.status == SCALED else point.status,
                        reason=None if point.status == SCALED else
                        point.reason or f"the rule gives the sizes of CAP {point.same_as}",
                        base_case=original["case"],
                        scaling={"cap": cap, "knobs": point.knobs, "divisor": point.divisor, "largest": point.largest,
                                 "input_elements": point.input_elements})
            if point.status == SCALED:
                case.update(path=str((data / point.folder).resolve()), data_folder=point.folder)
            cases.append(case)
        counts: dict[str, int] = {}
        for case in cases:
            counts[case["status"]] = counts.get(case["status"], 0) + 1
        document = {
            "case_set": f"npbench_pythran_cap{cap}",
            "version": 1,
            "description": f"The NPBench/Pythran performance entries at the scaling rule's CAP {cap} "
                           "(benchmarking/scaling_npb). Written by `python -m benchmarking.scaling_npb "
                           "generate`; folder paths are absolute (`relocate` after moving DATA).",
            "rules": {"sizes": rule.__doc__.strip(),
                      "statuses": {"perf": "scaled: measured at this cap",
                                   UNCHANGED: "the rule gives the sizes of an earlier point (reason)",
                                   FIXED: "the rule cannot scale the entry (reason)"}},
            "counts": counts,
            "cases": cases,
        }
        case_set_path(data, cap).write_text(json.dumps(document, indent=1) + "\n")


def relocate(data: Path) -> int:
    """Point every case set's folders at DATA's current location (after copying it)."""
    changed = 0
    for path in sorted((data / "case_sets").glob("npbench_pythran_cap*.json")):
        document = json.loads(path.read_text())
        for case in document["cases"]:
            if "data_folder" in case:
                case["path"] = str((data / case["data_folder"]).resolve())
                changed += 1
        path.write_text(json.dumps(document, indent=1) + "\n")
    return changed
