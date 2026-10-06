#!/usr/bin/env python3
"""Write `datasets/case_sets/acceptance.json` and derive the NPBench and Pythran
kernel signatures from upstream type information.

    python acceptance_case_set.py --npbench <spcl/npbench checkout> \\
                                  --pythran <serge-sans-paille/pythran checkout> \\
                                  [--write-kernels] [--write-fixtures]

Run it with a Python that has numpy, scipy (NPBench's spmv builds its input with
scipy.sparse) and pythran (its own parser reads the `#pythran export` lines).
Use the NumPy version of the repository's .venv (2.2.2 on 2026-10-06): the
fixture check runs the kernels as NumPy programs, and NumPy 2.4 removed
functions some kernels call (`np.in1d`). The NumPy version is recorded.

One acceptance entry per program and entry function:

* DS-1000 (220): status from `ds1000_numpy/manifest.json`; the kernel is
  `ds1000_numpy/generated/caseNNNN/kernel.py` (every DS-1000 input a runtime
  parameter: the runtime-shape variants belong here, audit DB-11).
* NPBench (54) and Pythran (115): status from `scope_rules.py` applied to the
  retained upstream code (the kernel without its Zinnia decorators and
  annotations) and to the upstream input types. Entry functions are NPBench's
  `func_name` and Pythran's exported functions that no other exported function
  calls; every other function is a `@zk_chip` helper without annotations.

Signatures (audit DB-10): each entry parameter's Zinnia annotation comes from
upstream, never from a guess:

* dtype and rank: NPBench's `initialize()` output (or its preset parameter) for
  the argument; for Pythran, the exported signature that accepts the first
  upstream example call (else the first all-numeric one). Pythran lists become
  arrays; a list of k-tuples becomes an array whose last axis is k.
* shape: the small inputs of acceptance_sizes.py (every array dimension and
  every size or trip-count scalar at most 16). NPBench: preset T, derived from
  preset S. Pythran: the upstream example (the first `#runas` call of the
  function that an all-numeric export accepts, else the first `#bench` call)
  scaled down by hand; with no example, every axis has length DEFAULT_AXIS.
  The shapes at the upstream example are kept as `upstream_signature`.
* a parameter upstream never passes keeps its default and is annotated from it.

The body of every function stays verbatim (checked against upstream: equal
ASTs once docstrings are dropped). Exported functions the 2026-05 extraction
left out are copied from upstream. Existing `@requires` preconditions are kept
on entry functions and listed per entry; they are machine drafts (an integral
float literal compared with a parameter now typed Integer becomes an integer;
a bound `0 <= x <= c` on a parameter that is now an array is dropped and
listed under `preconditions_dropped`).

With --write-fixtures, inputs A and B for every in-scope entry function
(acceptance_fixtures.py), checked by running the kernel as plain NumPy.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance_fixtures  # noqa: E402
import acceptance_sizes as sizes  # noqa: E402
import perf_fixtures  # noqa: E402
import ds1000_records as records  # noqa: E402
import generate_ds1000_kernels as generator  # noqa: E402
from scope_rules import RULES, RuleHits, text_rules  # noqa: E402

DATASETS = records.DATASETS
CASE_SET = DATASETS / "case_sets" / "acceptance.json"
SOURCES = DATASETS / "upstream_sources.json"
DEFAULT_AXIS = 8
ZINNIA_DECORATORS = {"zk_circuit", "zk_chip", "zk_external"}


# ---------------------------------------------------------------------------
# Zinnia annotations from values and Pythran types
# ---------------------------------------------------------------------------


class NotNumeric(Exception):
    pass


def annotation_of_value(name: str, value: Any) -> str:
    """Annotation of an example value; lists become arrays, tuples become Tuple[...]."""
    if isinstance(value, tuple):
        return "Tuple[" + ", ".join(annotation_of_value(name, v) for v in value) + "]"
    try:
        return generator.value_type(name, value).annotation()
    except generator.UntypeableInput as exc:
        raise NotNumeric(f"{name}: {exc.detail}") from exc


def _pythran_scalar(t: Any) -> str:
    name = getattr(t, "__name__", str(t)).lower()
    if name in ("bool", "bool_"):
        return "Boolean"
    if name.startswith(("int", "uint", "long")):
        return "Integer"
    if name.startswith(("float", "double")):
        return "Float"
    if name.startswith("complex"):
        return "Complex"
    raise NotNumeric(f"Pythran type {name}")


def describe_pythran(t: Any) -> tuple:
    """A Pythran type as ("scalar", dtype) | ("array", dtype, rank) | ("tuple", [items]).

    Lists become arrays: `float list` is rank 1, `float list list` rank 2, and a
    list of numeric tuples of one dtype gains one axis for the tuple. Dicts,
    sets, strings, modules and functions raise NotNumeric.
    """
    kind = type(t).__name__
    if kind == "NDArray" or (isinstance(t, type) and t.__name__ == "NDArray"):
        return ("array", _pythran_scalar(t.__args__[0]), len(t.__args__) - 1)
    if kind == "List":
        inner = describe_pythran(t.__args__[0])
        if inner[0] == "scalar":
            return ("array", inner[1], 1)
        if inner[0] == "array":
            return ("array", inner[1], inner[2] + 1)
        dtypes = {item[1] for item in inner[1]}
        if all(item[0] == "scalar" for item in inner[1]) and len(dtypes) == 1:
            return ("array", dtypes.pop(), 2, len(inner[1]))
        raise NotNumeric("list of mixed tuples")
    if kind == "Tuple":
        return ("tuple", [describe_pythran(a) for a in t.__args__])
    if kind in ("Dict", "Set", "Optional", "Union", "Fun", "Generator", "Iterable"):
        raise NotNumeric(f"Pythran {kind.lower()} type")
    return ("scalar", _pythran_scalar(t))


ACCEPTS = {"Boolean": "b", "Integer": "iu", "Float": "iuf", "Complex": "iufc"}


def is_ragged(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    try:
        return np.asarray(value).dtype == object
    except ValueError:
        return True


def matches(desc: tuple, value: Any) -> bool:
    if desc[0] == "tuple":
        return isinstance(value, tuple) and len(value) == len(desc[1]) and all(
            matches(d, v) for d, v in zip(desc[1], value))
    try:
        array = np.asarray(value)
    except ValueError:
        return False
    if array.dtype.kind not in ACCEPTS[desc[1]]:
        return False
    return array.ndim == (0 if desc[0] == "scalar" else desc[2])


def annotation_of(desc: tuple, value: Any) -> str:
    """Zinnia annotation; array shapes come from `value`, or DEFAULT_AXIS per axis without one."""
    if desc[0] == "tuple":
        values = value if value is not None else [None] * len(desc[1])
        return "Tuple[" + ", ".join(annotation_of(d, v) for d, v in zip(desc[1], values)) + "]"
    if desc[0] == "scalar":
        return desc[1]
    if value is not None:
        shape = np.asarray(value).shape
    else:  # a list of k-tuples keeps k as its last axis
        shape = (DEFAULT_AXIS,) * (desc[2] - 1) + ((desc[3],) if len(desc) > 3 else (DEFAULT_AXIS,))
    return f'{desc[1]}[Array, "{" ".join(str(d) for d in shape)}"]'


# ---------------------------------------------------------------------------
# Upstream sources
# ---------------------------------------------------------------------------


def function_defs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def body_key(fn: ast.FunctionDef) -> str:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def calls(fn: ast.FunctionDef) -> set[str]:
    return {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


@dataclass
class Signature:
    params: list[str]
    annotations: dict[str, str]
    shape_sources: dict[str, str]
    source: str                          # what the types come from
    example: dict[str, Any] = field(default_factory=dict)
    adaptations: list[str] = field(default_factory=list)
    example_call: str | None = None      # the example line the values come from
    b_example: dict[str, Any] | None = None   # input B when it is given, not drawn


def npbench_info(root: Path, upstream_path: str) -> dict[str, Any]:
    directory, _, filename = upstream_path.rpartition("/")
    module = filename.removesuffix("_numpy.py")
    for path in sorted((root / "bench_info").glob("*.json")):
        bench = json.loads(path.read_text())["benchmark"]
        if bench["relative_path"] == directory and bench["module_name"] == module:
            return bench
    raise LookupError(f"no bench_info for {upstream_path}")


def npbench_values(root: Path, info: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any]:
    """The preset's parameters plus what NPBench's initialize() returns for them.
    NumPy's global generator is seeded first: mlp's initialize() draws its input
    from it (np.random.rand) while its weights use default_rng(42)."""
    values: dict[str, Any] = dict(preset)
    init = info.get("init")
    if init:
        np.random.seed(42)
        directory = root / "npbench" / "benchmarks" / info["relative_path"]
        spec = importlib.util.spec_from_file_location(f"npb_{info['module_name']}",
                                                      directory / f"{info['module_name']}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(directory))
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            out = getattr(mod, init["func_name"])(*[preset[a] for a in init["input_args"]])
        finally:
            sys.path.pop(0)
        values.update(zip(init["output_args"], out if isinstance(out, tuple) else (out,)))
    return values


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def small_preset(root: Path, info: dict[str, Any]) -> tuple[dict[str, Any], int, str | None]:
    """Preset T (acceptance_sizes): every integer parameter p of preset S becomes
    max(min(p, 2), ceil(p / d)) for the smallest d that keeps every integer parameter,
    integer input and array dimension at most CAP; overrides are applied on top."""
    preset = info["parameters"]["S"]
    override, why = sizes.NPBENCH_T_OVERRIDES.get(info["short_name"], ({}, None))
    exempt = sizes.NPBENCH_DIMENSION_EXEMPT.get(info["short_name"], set())
    scaled = [k for k, v in preset.items() if _is_int(v) and k not in override]
    largest = max([preset[k] for k in scaled] or [1])
    for d in range(1, largest + 1):
        candidate = dict(preset)
        candidate.update({k: max(min(preset[k], 2), -(-preset[k] // d)) for k in scaled})
        candidate.update(override)
        if any(candidate[k] > sizes.CAP for k in scaled):
            continue
        try:
            values = npbench_values(root, info, candidate)
        except Exception:  # noqa: BLE001 - a combination initialize() rejects: try the next d
            continue
        inputs = [values[a] for a in info["input_args"]]
        if any(_is_int(v) and v > sizes.CAP for v in inputs):
            continue
        dims = {dim for v in inputs if isinstance(v, np.ndarray) for dim in v.shape}
        if all(dim <= sizes.CAP or dim in exempt for dim in dims):
            return candidate, d, why
    raise ValueError(f"no preset T for {info['short_name']}")


def npbench_signature(info: dict[str, Any], fn: ast.FunctionDef, values: dict[str, Any], label: str) -> Signature:
    params = [a.arg for a in fn.args.args]
    defaults = dict(zip(params[len(params) - len(fn.args.defaults):],
                        (ast.literal_eval(d) for d in fn.args.defaults)))
    passed = list(info["input_args"])
    if len(params) < len(passed) or any(p not in defaults for p in params[len(passed):]):
        raise ValueError(f"kernel parameters {params} do not take NPBench's input_args {passed}")
    # NPBench passes input_args positionally; a parameter it does not pass keeps its default.
    adaptations = []
    example: dict[str, Any] = {}
    for index, name in enumerate(params):
        if index < len(passed):
            example[name] = values[passed[index]]
        else:
            example[name] = defaults[name]
            adaptations.append(f"{name} is never passed by NPBench; annotated from its default {defaults[name]!r}")
    annotations, sources = {}, {}
    narrow: dict[str, list[str]] = {}
    for name in params:
        value = example[name]
        annotations[name] = annotation_of_value(name, value)
        sources[name] = label if isinstance(value, np.ndarray) else "scalar"
        kind = np.asarray(value).dtype
        if isinstance(value, np.ndarray) and kind not in (np.float64, np.int64, np.complex128, np.bool_):
            narrow.setdefault(str(kind), []).append(name)
    for kind, names in narrow.items():
        target = "Float (Q32 fixed point)" if kind.startswith("float") else "64-bit Integer"
        adaptations.append(f"{', '.join(names)}: {kind} in NPBench, {target} in the kernel")
    return Signature(params, annotations, sources, f"npbench bench_info/{info['short_name']}.json, initialize() at "
                     f"{label}", example, adaptations)


class _Recorded(Exception):
    pass


def upstream_example_lines(source: str, func: str) -> list[tuple[str, str]]:
    """The `#runas` lines that call `func`, then the `#bench` lines, in file order."""
    found = []
    for kind in ("runas", "bench"):
        for line in source.splitlines():
            match = re.match(rf"^#\s*{kind}:?\s+(.*)$", line)
            if match and re.search(rf"\b{re.escape(func)}\s*\(", match.group(1)):
                found.append((match.group(1), f"#{kind}"))
    return found


def call_arguments(code: str, func: str, module_globals: dict[str, Any]) -> list[list[Any]]:
    """Run an example line with `func` replaced by a recorder; the arguments of its calls."""
    captured: list[list[Any]] = []

    def recorder(*args: Any, **kwargs: Any) -> Any:
        captured.append(list(args))
        raise _Recorded

    namespace = dict(module_globals)
    namespace[func] = recorder
    try:
        exec(code, namespace)
    except _Recorded:
        pass
    except Exception:  # noqa: BLE001 - an example line that fails before the call has no arguments
        return []
    return captured


def pythran_spec(root: Path, upstream_path: str):
    import pythran.spec as S

    path = root / "pythran" / "tests" / "cases" / upstream_path
    source = path.read_text()
    spec = S.spec_parser(source)
    if not spec.functions and path.with_suffix(".pythran").exists():
        spec = S.load_specfile(str(path.with_suffix(".pythran")))
    return spec, source


def pythran_signature(spec, fn: ast.FunctionDef, examples: list[tuple[list[Any], str, str]]) -> Signature:
    """The first example call that some exported signature accepts fixes both the
    signature and the shapes; without one, the first all-numeric signature with
    DEFAULT_AXIS per axis."""
    import pythran.spec as S

    overloads = []
    for sig in spec.functions[fn.name]:
        try:
            overloads.append((sig, [describe_pythran(t) for t in sig]))
        except NotNumeric:
            continue
    exported = sorted({S.spec_to_string(fn.name, sig).replace(" order(F)", "") for sig in spec.functions[fn.name]})
    if not overloads:
        raise NotNumeric(f"every exported signature has a non-numeric type: {'; '.join(exported)}")
    chosen, example, shape_source, example_call = None, None, f"no example; {DEFAULT_AXIS} per axis", None
    for args, label, code in examples:
        for sig, descs in overloads:
            if len(args) == len(descs) and all(matches(d, v) for d, v in zip(descs, args)):
                chosen, example, shape_source, example_call = (sig, descs), args, label, code
                break
        if chosen:
            break
    if chosen is None:
        if any(is_ragged(v) for args, _, _ in examples for v in args):
            raise NotNumeric("the upstream example passes a ragged sequence (lists of different lengths)")
        chosen = overloads[0]
    sig, descs = chosen
    all_params = [a.arg for a in fn.args.args]
    params = all_params[:len(descs)]
    annotations = {name: annotation_of(d, example[i] if example else None)
                   for i, (name, d) in enumerate(zip(params, descs))}
    sources = {name: ("scalar" if d[0] == "scalar" else shape_source) for name, d in zip(params, descs)}
    values = dict(zip(params, example)) if example else {}
    adaptations = [f"{name} is a Pythran list; the kernel takes an array"
                   for name, t in zip(params, sig) if type(t).__name__ == "List"]
    # Parameters the export does not list keep their default; Zinnia still needs an annotation.
    defaults = dict(zip(all_params[len(all_params) - len(fn.args.defaults):], fn.args.defaults))
    for name in all_params[len(descs):]:
        try:
            value = ast.literal_eval(defaults[name])
            annotations[name] = annotation_of_value(name, value)
            sources[name] = "default value"
            values[name] = value
            adaptations.append(f"{name} is not in the export; annotated from its default {value!r}")
        except (KeyError, ValueError, NotNumeric):
            adaptations.append(f"{name} is not in the export and its default is not a literal; left unannotated")
    used = S.spec_to_string(fn.name, sig).replace(" order(F)", "")
    more = "" if len(exported) == 1 else f" (chosen from {len(exported)} exported signatures)"
    return Signature(all_params, annotations, sources, f"#pythran export {used}{more}", values, adaptations,
                     example_call)


def oversized(signature: Signature, exempt: set[int] = frozenset()) -> list[str]:
    """Parameters with an array dimension above the cap."""
    bad = []
    for name, annotation in signature.annotations.items():
        dims = [int(x) for x in re.findall(r"\d+", annotation.split("Array", 1)[1])] if "Array" in annotation else []
        if any(dim > sizes.CAP and dim not in exempt for dim in dims):
            bad.append(f"{name}: {annotation}")
    return bad


def pythran_module_globals(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "pythran_case"}
    try:
        exec(compile(source, "<pythran case>", "exec"), namespace)
    except Exception:  # noqa: BLE001 - some cases need modules we do not have; examples may still run
        pass
    return namespace


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _decorator_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    return target.id if isinstance(target, ast.Name) else ast.unparse(target)


def strip_function(node: ast.FunctionDef) -> list[ast.expr]:
    """Remove Zinnia decorators and annotations from `node` and every function
    nested in it (in place); return its `@requires` decorators."""
    requires = [d for d in node.decorator_list if _decorator_name(d) == "requires"]
    for sub in ast.walk(node):
        if isinstance(sub, ast.FunctionDef):
            sub.decorator_list = [d for d in sub.decorator_list
                                  if _decorator_name(d) not in ZINNIA_DECORATORS | {"requires"}]
            sub.returns = None
        if isinstance(sub, (ast.FunctionDef, ast.Lambda)):
            for arg in sub.args.posonlyargs + sub.args.args + sub.args.kwonlyargs:
                arg.annotation = None
    return requires


def is_zinnia_import(node: ast.stmt) -> bool:
    return (isinstance(node, ast.ImportFrom) and node.module == "zinnia") or \
        (isinstance(node, ast.Import) and any(a.name == "zinnia" for a in node.names))


def retained_code(tree: ast.Module) -> ast.Module:
    """The upstream code a kernel keeps: no Zinnia import, decorators or annotations."""
    tree = copy.deepcopy(tree)
    tree.body = [n for n in tree.body if not is_zinnia_import(n)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            strip_function(node)
    return tree


def assemble(kernel: ast.Module, upstream: ast.Module, exported: list[str]) -> tuple[ast.Module, list[str]]:
    """The kernel module plus any exported function (and the upstream functions
    it calls) that the 2026-05 extraction left out, copied from upstream."""
    tree = copy.deepcopy(kernel)
    upstream_fns = function_defs(upstream)
    added: list[str] = []
    pending = [n for n in exported if n not in function_defs(tree)]
    while pending:
        name = pending.pop(0)
        if name in function_defs(tree) or name not in upstream_fns:
            continue
        tree.body.append(copy.deepcopy(upstream_fns[name]))
        added.append(name)
        pending += [c for c in calls(upstream_fns[name]) if c in upstream_fns and c not in function_defs(tree)]
    return tree, added


def retype_requires(decorator: ast.expr, signature: Signature | None) -> ast.expr:
    """Make a drafted precondition agree with the derived types: a comparison of an
    Integer parameter with an integral float literal (`TSTEPS != 0.0`, drafted when
    TSTEPS was wrongly typed Float) compares with the integer literal instead."""
    if signature is None:
        return decorator
    integers = {name for name, annotation in signature.annotations.items() if annotation == "Integer"}

    class Retype(ast.NodeTransformer):
        def visit_Compare(self, node: ast.Compare) -> ast.Compare:
            operands = [node.left, *node.comparators]
            if any(isinstance(o, ast.Name) and o.id in integers for o in operands):
                for i, o in enumerate(operands):
                    if isinstance(o, ast.Constant) and isinstance(o.value, float) and o.value.is_integer():
                        operands[i] = ast.Constant(int(o.value))
                node.left, node.comparators = operands[0], operands[1:]
            return node

    return Retype().visit(copy.deepcopy(decorator))


def ill_typed_bound(decorator: ast.expr, signature: Signature | None) -> str | None:
    """Why a drafted precondition no longer type-checks, or None. A bound
    `0 <= x <= c` was drafted for an integer x; when the derived type of x is an
    array the comparison has no single truth value."""
    if signature is None or not isinstance(decorator, ast.Call) or not decorator.args:
        return None
    lam = decorator.args[0]
    if not isinstance(lam, ast.Lambda) or not isinstance(lam.body, ast.Compare):
        return None
    for operand in [lam.body.left, *lam.body.comparators]:
        if isinstance(operand, ast.Name) and "Array" in signature.annotations.get(operand.id, ""):
            return (f"{operand.id} is {signature.annotations[operand.id]} in the derived signature; a bound on it "
                    f"has no single truth value (drafted when {operand.id} was taken for an integer)")
    return None


def rewrite_kernel(tree: ast.Module, entries: dict[str, Signature | None], header: list[str]) -> str:
    """Entry functions get @zk_circuit, their kept @requires and the derived
    annotations; every other function is an unannotated @zk_chip."""
    tree = copy.deepcopy(tree)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        requires = strip_function(node)
        others = list(node.decorator_list)
        if node.name in entries:
            signature = entries[node.name]
            requires = [retype_requires(d, signature) for d in requires if not ill_typed_bound(d, signature)]
            node.decorator_list = [ast.Name("zk_circuit"), *requires, *others]
            for arg in node.args.args:
                if signature and arg.arg in signature.annotations:
                    arg.annotation = ast.parse(signature.annotations[arg.arg], mode="eval").body
        else:
            node.decorator_list = [ast.Name("zk_chip"), *others]
    tree.body = [ast.ImportFrom("zinnia", [ast.alias("*")], 0), *[n for n in tree.body if not is_zinnia_import(n)]]
    return "\n".join(header) + "\n" + ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def current_annotations(fn: ast.FunctionDef) -> dict[str, str | None]:
    return {a.arg: ast.unparse(a.annotation) if a.annotation is not None else None for a in fn.args.args}


def current_requires(fn: ast.FunctionDef) -> list[str]:
    return [ast.unparse(d) for d in fn.decorator_list if _decorator_name(d) == "requires"]


def check_requires(requires: list[str], example: dict[str, Any]) -> list[dict[str, Any]]:
    """Each precondition and whether the upstream example satisfies it."""
    out = []
    for text in requires:
        call = ast.parse(text, mode="eval").body
        lam = call.args[0] if isinstance(call, ast.Call) and call.args else None
        holds: Any = None
        if isinstance(lam, ast.Lambda) and example:
            try:
                fn = eval(compile(ast.Expression(lam), "<requires>", "eval"), {"np": np})
                holds = bool(fn(*[example.get(a.arg) for a in lam.args.args]))
            except Exception:  # noqa: BLE001
                holds = None
        kind = "trip-count bound" if "<=" in text and "0 <=" in text else ("non-zero divisor" if "!=" in text else "other")
        out.append({"precondition": text, "kind": kind, "holds_on_example": holds})
    return out


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def ds1000_entries() -> list[dict[str, Any]]:
    manifest = records.load_manifest()
    index = json.loads((DATASETS / "ds1000_numpy" / "generated" / "index.json").read_text())
    entries = []
    for pid, row in sorted(manifest.items()):
        key = f"case{pid:04d}"
        generated = index.get(key, {})
        perf = DATASETS / "ds1000" / f"ds1000_{key}"
        entries.append({
            "dataset": "ds1000_numpy", "program": key, "function": "kernel",
            "status": row["status"], "rules": row["scope_rules"], "notes": row["scope_notes"],
            "kernel": f"ds1000_numpy/generated/{key}/kernel.py" if "params" in generated else None,
            "signature": {p["name"]: p["annotation"] for p in generated.get("params", [])},
            "signature_source": f"DS-1000 test case {generated['shape_test_case']} shapes; dtypes over all test cases"
                                if "params" in generated else None,
            "performance_set_case": f"ds1000/ds1000_{key}" if perf.exists() else None,
        })
    return entries


def npbench_sizes(root: Path, info: dict[str, Any], fn: ast.FunctionDef
                  ) -> tuple[Signature, Signature, dict[str, Any], set[int]]:
    """(signature at preset T, signature at preset S, sizing record, exempt dimensions)."""
    bench = npbench_info(root, info["upstream_path"])
    preset_s = bench["parameters"]["S"]
    upstream = npbench_signature(bench, fn, npbench_values(root, bench, preset_s), "preset S")
    preset_t, divisor, exception = small_preset(root, bench)
    small = npbench_signature(bench, fn, npbench_values(root, bench, preset_t), "preset T")
    sizing = {"rule": "NPBench preset T (acceptance_sizes.py)", "preset_S": preset_s, "preset_T": preset_t,
              "divisor": divisor, "exception": exception}
    if bench["short_name"] in acceptance_fixtures.NPBENCH_Q32_INPUTS:
        factors, why = acceptance_fixtures.NPBENCH_Q32_INPUTS[bench["short_name"]]
        for name, factor in factors.items():
            small.example[name] = small.example[name] * factor
        sizing["q32_change"] = why
    return small, upstream, sizing, sizes.NPBENCH_DIMENSION_EXEMPT.get(bench["short_name"], set())


def reproducible(code: str) -> str:
    """An example line whose arguments do not depend on uninitialized memory:
    `np.empty(...)` becomes `np.zeros(...)` (spectraldns builds cross2's output
    buffer with np.empty; the kernel overwrites it)."""
    return re.sub(r"\bempty\(", "zeros(", code)


def signature_from_line(spec, fn: ast.FunctionDef, module_globals: dict[str, Any], code: str,
                        label: str) -> Signature:
    examples = [(args, label, code) for args in call_arguments(code, fn.name, module_globals)]
    if not examples:
        raise SystemExit(f"{fn.name}: the example does not reach the call: {code}")
    return pythran_signature(spec, fn, examples)


def pythran_sizes(spec, source: str, module_globals: dict[str, Any], program: str, fn: ast.FunctionDef
                  ) -> tuple[Signature, Signature, dict[str, Any]]:
    """(signature at the small example, signature at the upstream example, sizing record)."""
    lines = [(reproducible(code), label) for code, label in upstream_example_lines(source, fn.name)]
    upstream_examples = [(args, label, code) for code, label in lines
                         for args in call_arguments(code, fn.name, module_globals)]
    upstream = pythran_signature(spec, fn, upstream_examples)
    key = (program, fn.name)
    table = sizes.PYTHRAN_SMALL_EXAMPLES if key in sizes.PYTHRAN_SMALL_EXAMPLES else sizes.PYTHRAN_NO_UPSTREAM_EXAMPLE
    if key not in table:
        small = upstream
        sizing = {"rule": "upstream example used as it is (already within the cap)",
                  "example": upstream.example_call}
    else:
        code, change = table[key]
        small = signature_from_line(spec, fn, module_globals, reproducible(code), "small example")
        sizing = {"rule": "upstream example with sizes scaled to the cap (acceptance_sizes.py)"
                  if table is sizes.PYTHRAN_SMALL_EXAMPLES else "no upstream example calls it; example chosen here",
                  "example": small.example_call, "change": change, "upstream_example": upstream.example_call}
    if key in acceptance_fixtures.PYTHRAN_Q32_INPUTS:
        code_a, code_b, why = acceptance_fixtures.PYTHRAN_Q32_INPUTS[key]
        fixed = signature_from_line(spec, fn, module_globals, code_a, "example changed for Q32")
        if fixed.annotations != small.annotations:
            raise SystemExit(f"{program}.{fn.name}: the Q32 input changes the signature: {fixed.annotations}")
        if code_b:
            fixed.b_example = signature_from_line(spec, fn, module_globals, code_b, "B").example
        sizing.update(example_before_q32_change=sizing["example"], example=code_a, example_B=code_b,
                      q32_change=why)
        small = fixed
    return small, upstream, sizing


def upstream_program_entries(program: str, info: dict[str, Any], roots: dict[str, Path],
                             write: bool) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Signature]]:
    dataset = info["dataset"]
    kernel_path = DATASETS / dataset / program / "kernel.py"
    original = ast.parse(kernel_path.read_text())
    # The drafted kernel before 2026-10-06, for the "previous" columns; the rewrite is idempotent.
    drafted_fns = function_defs(ast.parse(previous_kernel(dataset, program)))

    if dataset == "npbench":
        upstream_source = (roots["npbench"] / "npbench" / "benchmarks" / info["upstream_path"]).read_text()
        exported = [npbench_info(roots["npbench"], info["upstream_path"])["func_name"]]
        spec = module_globals = None
    else:
        spec, upstream_source = pythran_spec(roots["pythran"], info["upstream_path"])
        exported = list(spec.functions)
        module_globals = pythran_module_globals(upstream_source)
    upstream = ast.parse(upstream_source)
    upstream_fns = function_defs(upstream)

    tree, added = assemble(original, upstream, exported)
    kernel_fns = function_defs(tree)
    retained = retained_code(tree)
    retained_fns = function_defs(retained)
    verbatim = {name: name in upstream_fns and body_key(fn) == body_key(upstream_fns[name])
                for name, fn in retained_fns.items()}
    hits = text_rules(retained)
    if not kernel_fns:
        hits.add("no_callable_kernel", "no top-level function")

    present = [n for n in exported if n in kernel_fns]
    called_by_export = {c for n in present for c in calls(kernel_fns[n]) if c in present and c != n}
    entry_names = [n for n in present if n not in called_by_export]

    entries, signatures = [], {}
    for name in present:
        entry: dict[str, Any] = {"dataset": dataset, "program": program, "function": name,
                                 "kernel": f"{dataset}/{program}/kernel.py", "upstream": info["upstream_path"],
                                 "body_verbatim": verbatim[name], "added_from_upstream": name in added}
        previous = current_annotations(drafted_fns[name]) if name in drafted_fns else None
        entry["previous_signature"] = previous
        if name in called_by_export:
            callers = sorted(n for n in present if name in calls(kernel_fns[n]))
            entry.update(entry_function=False, status="out_of_scope" if hits.rules else "in_scope",
                         rules=hits.sorted(),
                         notes=[f"exported, but called by the exported function(s) {callers}; it is a @zk_chip "
                                "helper there and is compiled as part of them"])
            entries.append(entry)
            continue
        entry["entry_function"] = True
        function_hits = RuleHits()
        function_hits.merge(hits)
        try:
            if dataset == "npbench":
                sig, upstream_sig, entry["sizing"], exempt = npbench_sizes(roots["npbench"], info, upstream_fns[name])
            else:
                sig, upstream_sig, entry["sizing"] = pythran_sizes(spec, upstream_source, module_globals, program,
                                                                   upstream_fns[name])
                exempt = set()
            too_big = oversized(sig, exempt)
            if too_big and not function_hits.rules:
                raise SystemExit(f"{program}.{name}: above the size cap {sizes.CAP}: {too_big}")
            signatures[name] = sig
            entry.update(signature=sig.annotations, upstream_signature=upstream_sig.annotations,
                         signature_source=sig.source, shape_source=sig.shape_sources, adaptations=sig.adaptations)
            entry["signature_changed"] = {p: {"previous": (previous or {}).get(p), "derived": a}
                                          for p, a in upstream_sig.annotations.items()
                                          if (previous or {}).get(p) != a}
            drafted = [d for d in (drafted_fns[name].decorator_list if name in drafted_fns else [])
                       if _decorator_name(d) == "requires"]
            kept = [d for d in drafted if not ill_typed_bound(d, sig)]
            entry["preconditions"] = check_requires([ast.unparse(retype_requires(d, sig)) for d in kept], sig.example)
            for item, d in zip(entry["preconditions"], kept):
                if item["precondition"] != ast.unparse(d):
                    item["drafted_as"] = ast.unparse(d)
            dropped_here = [{"precondition": ast.unparse(d), "dropped_because": ill_typed_bound(d, sig)}
                            for d in drafted if ill_typed_bound(d, sig)]
            if dropped_here:
                entry["preconditions_dropped"] = dropped_here
        except NotNumeric as exc:
            function_hits.add("non_numeric_data", f"input types: {exc}")
        entry["status"] = "out_of_scope" if function_hits.rules else "in_scope"
        entry["rules"] = function_hits.sorted()
        entries.append(entry)

    dropped = {n: current_requires(fn) for n, fn in drafted_fns.items()
               if n not in entry_names and current_requires(fn)}
    program_info = {"exported": exported, "entry_functions": entry_names, "added_from_upstream": added,
                    "preconditions_dropped_from_helpers": dropped,
                    "bodies_verbatim": all(verbatim.values()),
                    "functions_not_in_upstream_or_not_verbatim": sorted(n for n, ok in verbatim.items() if not ok)}
    if write:
        repo = "spcl/npbench npbench/benchmarks" if dataset == "npbench" else "serge-sans-paille/pythran pythran/tests/cases"
        header = [
            f"# {dataset} program {info['upstream_path']} ({repo}).",
            "# Function bodies are the upstream text. Entry-function annotations are derived from upstream",
            "# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.",
            "",
        ]
        source = rewrite_kernel(tree, {n: signatures.get(n) for n in entry_names}, header)
        rewritten = function_defs(retained_code(ast.parse(source)))
        for name, fn in rewritten.items():
            if body_key(fn) != body_key(retained_fns[name]):
                raise SystemExit(f"{program}.{name}: rewriting changed the body")
        kernel_path.write_text(source, encoding="utf-8")
    return entries, program_info, signatures


PREVIOUS_KERNELS = "11cd122"   # bench-reorg: the drafted annotations before 2026-10-06


def previous_kernel(dataset: str, program: str) -> str:
    return subprocess.run(["git", "show", f"{PREVIOUS_KERNELS}:benchmarking/datasets/{dataset}/{program}/kernel.py"],
                          cwd=DATASETS, capture_output=True, text=True, check=True).stdout


def ds1000_inputs(problem_id: int) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """(A, B or None, note) for a DS-1000 acceptance kernel."""
    record = records.load_manifest()[problem_id]
    perf_dir = DATASETS / "ds1000" / f"ds1000_case{problem_id:04d}"
    if problem_id in perf_fixtures.DRAWS and perf_dir.exists():
        fixture = perf_fixtures.build_fixture(perf_dir, record)
        constants = fixture.data["constants"]
        a = {**fixture.data["inputs"]["A"], **constants}
        b = {**fixture.data["inputs"]["B"], **constants}
        note = (f"A and B are the performance set's draws for this problem ({fixture.data['notes']}) with "
                f"{', '.join(constants) or 'no'} shape parameters at their DS-1000 value.")
        return a, b, note
    case = records.evaluate_test_case(record, records.test_case_ids(record)[-1])
    return case.env, None, f"A is DS-1000's own input of test case {case.case_id}."


def write_fixtures(entries: list[dict[str, Any]], examples: dict[tuple[str, str], Signature]) -> None:
    """fixture.json for every in-scope entry function; the path (or the reason there
    is none) goes into the entry."""
    per_program: dict[str, int] = {}
    for e in entries:
        if e.get("entry_function") is not False and e["dataset"] != "ds1000_numpy":
            per_program[e["program"]] = per_program.get(e["program"], 0) + 1
    for e in entries:
        if e["status"] != "in_scope" or e.get("entry_function") is False or not e.get("kernel"):
            continue
        kernel = DATASETS / e["kernel"]
        function = e["function"]
        if e["dataset"] == "ds1000_numpy":
            a, b, note = ds1000_inputs(int(e["program"][4:]))
            name = "fixture.json"
        else:
            sig = examples[(e["program"], function)]
            a, b = sig.example, sig.b_example
            if e["dataset"] == "npbench":
                note = f"A is NPBench's initialize() data at preset T {e['sizing']['preset_T']}."
            else:
                note = f"A is the arguments of the example call `{e['sizing']['example']}`."
            if e["sizing"].get("q32_change"):
                note += f" Changed for Q32: {e['sizing']['q32_change']}."
            if b is not None:
                note += f" B is the arguments of `{e['sizing']['example_B']}`."
            name = acceptance_fixtures.fixture_name(per_program[e["program"]], function)
        try:
            data = acceptance_fixtures.build(e["program"], function, kernel, e["signature"], a,
                                             acceptance_fixtures.seed_for(e["program"], function), note, b)
        except (acceptance_fixtures.FixtureError, KeyError) as exc:
            e["fixture"] = None
            e["fixture_error"] = f"{type(exc).__name__}: {exc}"
            print("fixture", e["program"], function, e["fixture_error"], flush=True)
            continue
        tolerance = e["float_tolerance"]
        if "measure" in tolerance:
            inputs = [[acceptance_fixtures.coerce(data["inputs"][label][p["name"]], p) for p in data["parameters"]]
                      for label in "AB"]
            measured = acceptance_fixtures.measured_difference(kernel, function, tolerance["measure"], inputs)
            if measured > tolerance["atol"]:
                raise SystemExit(f"{e['program']}.{function}: measured {measured:.3g} exceeds atol {tolerance['atol']}")
            tolerance["measured_max_abs_difference"] = float(f"{measured:.3g}")
            data["notes"] += (f" Float tolerance: atol = {tolerance['atol']:g} instead of the default 1e-6, "
                              f"rtol = {tolerance['rtol']:g}. Why: {tolerance['reason']} (measured on A and B: "
                              f"{measured:.2g}).")
        path = kernel.parent / name
        acceptance_fixtures.dump(path, data)
        e["fixture"] = str(path.relative_to(DATASETS))


def git_head(path: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npbench", type=Path, required=True)
    parser.add_argument("--pythran", type=Path, required=True)
    parser.add_argument("--write-kernels", action="store_true")
    parser.add_argument("--write-fixtures", action="store_true",
                        help="write fixture.json for every in-scope entry (after --write-kernels)")
    args = parser.parse_args()
    roots = {"npbench": args.npbench, "pythran": args.pythran}

    sources = json.loads(SOURCES.read_text())
    entries = ds1000_entries()
    programs: dict[str, Any] = {}
    examples: dict[tuple[str, str], Signature] = {}
    for program, info in sources["programs"].items():
        program_entries, programs[program], signatures = upstream_program_entries(program, info, roots,
                                                                                  args.write_kernels)
        entries += program_entries
        examples.update({(program, name): sig for name, sig in signatures.items()})
        print(program, [e["status"] for e in program_entries], flush=True)
    for e in entries:
        override = acceptance_fixtures.FLOAT_TOLERANCE_OVERRIDES.get((e["program"], e["function"]))
        e["float_tolerance"] = dict(override) if override else dict(acceptance_fixtures.FLOAT_TOLERANCE_DEFAULT)
    if args.write_fixtures:
        write_fixtures(entries, examples)

    counts: dict[str, dict[str, int]] = {}
    for e in entries:
        if e.get("entry_function") is False:
            continue
        bucket = counts.setdefault(e["dataset"], {"in_scope": 0, "out_of_scope": 0})
        bucket[e["status"]] += 1
    document = {
        "case_set": "acceptance",
        "version": 1,
        "description": "Every program and entry function of the acceptance (coverage) study, with its scope status "
                       "under datasets/FILTERING.md and the reason. Written by tools/acceptance_case_set.py.",
        "rules": RULES,
        "upstream": {"npbench_commit": git_head(args.npbench), "pythran_commit": git_head(args.pythran),
                     "default_axis_without_example": DEFAULT_AXIS, "numpy_version": np.__version__},
        "sizing": {"cap": sizes.CAP, "rule": sizes.__doc__.split("\n\n", 1)[1].strip()},
        "float_tolerance": {
            "default": acceptance_fixtures.FLOAT_TOLERANCE_DEFAULT,
            "rule": "Each entry's `float_tolerance`: a float (or complex) output passes when |actual - expected| "
                    "<= atol + rtol * |expected|, expected being NumPy's output on the same fixture input. "
                    "Integer and boolean outputs are compared exactly. The default is atol = rtol = 1e-6. An "
                    "entry whose NumPy program Q32 cannot reproduce has a larger atol with the reason and the "
                    "measured effect (`measure`: q32_inputs = NumPy on the inputs rounded to Q32 against NumPy "
                    "on the inputs; float64_buffers = float32 buffers made float64 against as written)."},
        "counts_entry_functions": counts,
        "programs": programs,
        "entries": entries,
    }
    CASE_SET.write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
