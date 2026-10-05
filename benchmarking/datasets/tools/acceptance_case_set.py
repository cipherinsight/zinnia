#!/usr/bin/env python3
"""Write `datasets/case_sets/acceptance.json` and derive the NPBench and Pythran
kernel signatures from upstream type information.

    python acceptance_case_set.py --npbench <spcl/npbench checkout> \\
                                  --pythran <serge-sans-paille/pythran checkout> [--write-kernels]

Run it with a Python that has numpy, scipy (NPBench's spmv builds its input with
scipy.sparse) and pythran (its own parser reads the `#pythran export` lines).

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
* shape: the upstream example input. NPBench: preset S (the smallest it
  defines). Pythran: the first `#runas` call of the function, else the first
  `#bench` call; with neither, every axis has length DEFAULT_AXIS.
* a parameter upstream never passes keeps its default and is annotated from it.

The body of every function stays verbatim (checked against upstream: equal
ASTs once docstrings are dropped). Exported functions the 2026-05 extraction
left out are copied from upstream. Existing `@requires` preconditions are kept
on entry functions and listed per entry; they are machine drafts (an integral
float literal compared with a parameter now typed Integer becomes an integer).
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
import ds1000_records as records  # noqa: E402
import generate_ds1000_kernels as generator  # noqa: E402
from scope_rules import RULES, RuleHits, text_rules  # noqa: E402

DATASETS = records.DATASETS
CASE_SET = DATASETS / "case_sets" / "acceptance.json"
SOURCES = DATASETS / "upstream_sources.json"
DEFAULT_AXIS = 8
NPBENCH_PRESET = "S"   # NPBench's smallest preset; see --npbench-preset
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


def npbench_signature(root: Path, upstream_path: str, fn: ast.FunctionDef, preset_name: str) -> Signature:
    directory, _, filename = upstream_path.rpartition("/")
    module = filename.removesuffix("_numpy.py")
    info = None
    for path in sorted((root / "bench_info").glob("*.json")):
        bench = json.loads(path.read_text())["benchmark"]
        if bench["relative_path"] == directory and bench["module_name"] == module:
            info = bench
            break
    if info is None:
        raise LookupError(f"no bench_info for {upstream_path}")
    preset = info["parameters"][preset_name]
    values: dict[str, Any] = dict(preset)
    init = info.get("init")
    if init:
        spec = importlib.util.spec_from_file_location(f"npb_{module}", root / "npbench" / "benchmarks" / directory / f"{module}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(root / "npbench" / "benchmarks" / directory))
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            out = getattr(mod, init["func_name"])(*[preset[a] for a in init["input_args"]])
        finally:
            sys.path.pop(0)
        out = out if isinstance(out, tuple) else (out,)
        values.update(zip(init["output_args"], out))
    params = [a.arg for a in fn.args.args]
    defaults = dict(zip(params[len(params) - len(fn.args.defaults):],
                        (ast.literal_eval(d) for d in fn.args.defaults)))
    passed = list(info["input_args"])
    if len(params) < len(passed) or any(p not in defaults for p in params[len(passed):]):
        raise ValueError(f"kernel parameters {params} do not take NPBench's input_args {passed}")
    # NPBench passes input_args positionally; a parameter it does not pass keeps its default.
    adaptations = []
    by_param: dict[str, Any] = {}
    for index, name in enumerate(params):
        if index < len(passed):
            by_param[name] = values[passed[index]]
        else:
            by_param[name] = defaults[name]
            adaptations.append(f"{name} is never passed by NPBench; annotated from its default {defaults[name]!r}")
    values = by_param
    annotations, sources = {}, {}
    narrow: dict[str, list[str]] = {}
    for name in params:
        value = values[name]
        annotations[name] = annotation_of_value(name, value)
        sources[name] = f"NPBench preset {preset_name}" if isinstance(value, np.ndarray) else "scalar"
        kind = np.asarray(value).dtype
        if isinstance(value, np.ndarray) and kind not in (np.float64, np.int64, np.complex128, np.bool_):
            narrow.setdefault(str(kind), []).append(name)
    for kind, names in narrow.items():
        target = "Float (Q32 fixed point)" if kind.startswith("float") else "64-bit Integer"
        adaptations.append(f"{', '.join(names)}: {kind} in NPBench, {target} in the kernel")
    example = {n: values[n] for n in params}
    return Signature(params, annotations, sources, f"npbench bench_info/{info['short_name']}.json, initialize() at "
                     f"preset {preset_name} {preset}", example, adaptations)


class _Recorded(Exception):
    pass


def pythran_calls(source: str, func: str, module_globals: dict[str, Any]) -> list[tuple[list[Any], str]]:
    """Arguments of every `#runas` call of `func`, then every `#bench` call, in file order."""
    found = []
    for kind in ("runas", "bench"):
        for line in source.splitlines():
            match = re.match(rf"^#\s*{kind}\s+(.*)$", line)
            if not match or not re.search(rf"\b{re.escape(func)}\s*\(", match.group(1)):
                continue
            captured: list[list[Any]] = []

            def recorder(*args: Any, **kwargs: Any) -> Any:
                captured.append(list(args))
                raise _Recorded

            namespace = dict(module_globals)
            namespace[func] = recorder
            try:
                exec(match.group(1), namespace)
            except _Recorded:
                pass
            except Exception:  # noqa: BLE001 - an example line that fails before the call is skipped
                continue
            found += [(args, f"#{kind}") for args in captured]
    return found


def pythran_spec(root: Path, upstream_path: str):
    import pythran.spec as S

    path = root / "pythran" / "tests" / "cases" / upstream_path
    source = path.read_text()
    spec = S.spec_parser(source)
    if not spec.functions and path.with_suffix(".pythran").exists():
        spec = S.load_specfile(str(path.with_suffix(".pythran")))
    return spec, source


def pythran_signature(spec, source: str, module_globals: dict[str, Any], fn: ast.FunctionDef) -> Signature:
    """The first upstream example call that some exported signature accepts fixes both
    the signature and the shapes; without one, the first all-numeric signature with
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
    chosen, example, shape_source = None, None, f"no upstream example; {DEFAULT_AXIS} per axis"
    calls_found = pythran_calls(source, fn.name, module_globals)
    for args, kind in calls_found:
        for sig, descs in overloads:
            if len(args) == len(descs) and all(matches(d, v) for d, v in zip(descs, args)):
                chosen, example, shape_source = (sig, descs), args, kind
                break
        if chosen:
            break
    if chosen is None:
        if any(is_ragged(v) for args, _ in calls_found for v in args):
            raise NotNumeric("the upstream example passes a ragged sequence (lists of different lengths)")
        chosen = overloads[0]
    sig, descs = chosen
    all_params = [a.arg for a in fn.args.args]
    params = all_params[:len(descs)]
    annotations = {name: annotation_of(d, example[i] if example else None)
                   for i, (name, d) in enumerate(zip(params, descs))}
    sources = {name: ("scalar" if d[0] == "scalar" else shape_source) for name, d in zip(params, descs)}
    adaptations = [f"{name} is a Pythran list; the kernel takes an array"
                   for name, t in zip(params, sig) if type(t).__name__ == "List"]
    # Parameters the export does not list keep their default; Zinnia still needs an annotation.
    defaults = dict(zip(all_params[len(all_params) - len(fn.args.defaults):], fn.args.defaults))
    for name in all_params[len(descs):]:
        try:
            value = ast.literal_eval(defaults[name])
            annotations[name] = annotation_of_value(name, value)
            sources[name] = "default value"
            adaptations.append(f"{name} is not in the export; annotated from its default {value!r}")
        except (KeyError, ValueError, NotNumeric):
            adaptations.append(f"{name} is not in the export and its default is not a literal; left unannotated")
    used = S.spec_to_string(fn.name, sig).replace(" order(F)", "")
    more = "" if len(exported) == 1 else f" (chosen from {len(exported)} exported signatures)"
    return Signature(all_params, annotations, sources, f"#pythran export {used}{more}",
                     dict(zip(params, example)) if example else {}, adaptations)


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
            requires = [retype_requires(d, signature) for d in requires]
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
        out.append({"precondition": text, "kind": kind, "holds_on_upstream_example": holds})
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


def upstream_program_entries(program: str, info: dict[str, Any], roots: dict[str, Path],
                             write: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = info["dataset"]
    kernel_path = DATASETS / dataset / program / "kernel.py"
    original = ast.parse(kernel_path.read_text())
    # The drafted kernel before 2026-10-06, for the "previous" columns; the rewrite is idempotent.
    drafted_fns = function_defs(ast.parse(previous_kernel(dataset, program)))

    if dataset == "npbench":
        upstream_source = (roots["npbench"] / "npbench" / "benchmarks" / info["upstream_path"]).read_text()
        directory, _, filename = info["upstream_path"].rpartition("/")
        benches = [json.loads(p.read_text())["benchmark"] for p in sorted((roots["npbench"] / "bench_info").glob("*.json"))]
        bench = next(b for b in benches if b["relative_path"] == directory
                     and b["module_name"] == filename.removesuffix("_numpy.py"))
        exported = [bench["func_name"]]
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
                sig = npbench_signature(roots["npbench"], info["upstream_path"], upstream_fns[name], NPBENCH_PRESET)
            else:
                sig = pythran_signature(spec, upstream_source, module_globals, upstream_fns[name])
            signatures[name] = sig
            entry.update(signature=sig.annotations, signature_source=sig.source, shape_source=sig.shape_sources,
                         adaptations=sig.adaptations)
            entry["signature_changed"] = {p: {"previous": (previous or {}).get(p), "derived": a}
                                          for p, a in sig.annotations.items() if (previous or {}).get(p) != a}
            drafted = drafted_fns[name].decorator_list if name in drafted_fns else []
            kept = [ast.unparse(retype_requires(d, sig)) for d in drafted if _decorator_name(d) == "requires"]
            entry["preconditions"] = check_requires(kept, sig.example)
            for item, before in zip(entry["preconditions"], current_requires(drafted_fns[name]) if name in drafted_fns else []):
                if item["precondition"] != before:
                    item["drafted_as"] = before
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
    return entries, program_info


PREVIOUS_KERNELS = "11cd122"   # bench-reorg: the drafted annotations before 2026-10-06


def previous_kernel(dataset: str, program: str) -> str:
    return subprocess.run(["git", "show", f"{PREVIOUS_KERNELS}:benchmarking/datasets/{dataset}/{program}/kernel.py"],
                          cwd=DATASETS, capture_output=True, text=True, check=True).stdout


def git_head(path: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npbench", type=Path, required=True)
    parser.add_argument("--pythran", type=Path, required=True)
    parser.add_argument("--write-kernels", action="store_true")
    parser.add_argument("--npbench-preset", default="S", help="NPBench parameter preset for example shapes")
    args = parser.parse_args()
    global NPBENCH_PRESET
    NPBENCH_PRESET = args.npbench_preset
    roots = {"npbench": args.npbench, "pythran": args.pythran}

    sources = json.loads(SOURCES.read_text())
    entries = ds1000_entries()
    programs: dict[str, Any] = {}
    for program, info in sources["programs"].items():
        program_entries, programs[program] = upstream_program_entries(program, info, roots, args.write_kernels)
        entries += program_entries
        print(program, [e["status"] for e in program_entries], flush=True)

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
                     "default_axis_without_example": DEFAULT_AXIS, "npbench_preset": NPBENCH_PRESET},
        "counts_entry_functions": counts,
        "programs": programs,
        "entries": entries,
    }
    CASE_SET.write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
