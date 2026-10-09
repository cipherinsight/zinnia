"""The DS-1000 performance set: per-case decisions and the kernels they imply.

The performance inventory includes every existing folder under
`datasets/ds1000/`. RQ1_ACCEPTED is the author's frozen acceptance list:
all of these programs are `perf`, including duplicates and constant-output
programs. Earlier exclusion decisions remain in each entry's `former_status`
and `former_duplicate_of`. The eight historical perf programs outside RQ1's
acceptance list stay perf for downstream analysis to filter.

Other entries retain their historical out_of_scope / duplicate /
zinnia_rejects decision. No compiler probe is needed to refresh this selection.
The dictionaries below still document shape constants, degenerate behavior,
duplicate relationships and data movement; they no longer exclude an accepted
program from the performance set.
This module holds the hand-made inputs to those rules (the shape parameters that
become compile-time constants, the degenerate cases and their reasons, the
duplicate that is not textual) and derives each kernel from the generator. It
never writes kernels; `perf_case_set.py --check` checks their generated source.
"""

from __future__ import annotations

import ast
import itertools
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds1000_records as records  # noqa: E402
import generate_ds1000_kernels as generator  # noqa: E402

PERF_DIR = records.DATASETS / "ds1000"

# Author decision 2026-10-09: time every program accepted in RQ1. Frozen from
# eval-redo/all-work/rq1-ds-accepted.txt (132 programs); no compiler inference.
# Keep this decision here so performance regeneration is reproducible without
# an external acceptance-results directory or a Zinnia invocation.
RQ1_ACCEPTED: frozenset[int] = frozenset({
    291, 298, 300, 301, 302, 303, 304, 309, 310, 311, 312, 313,
    314, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 329,
    330, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 345,
    346, 347, 349, 353, 355, 356, 357, 358, 359, 360, 361, 363,
    364, 365, 368, 369, 370, 373, 374, 375, 381, 382, 383, 385,
    386, 387, 389, 391, 392, 393, 398, 399, 400, 401, 402, 403,
    409, 412, 413, 414, 415, 416, 417, 418, 419, 420, 422, 428,
    429, 430, 431, 433, 434, 435, 436, 437, 438, 439, 440, 441,
    442, 443, 444, 448, 449, 450, 451, 452, 453, 454, 456, 457,
    458, 462, 463, 467, 468, 471, 472, 477, 478, 480, 481, 485,
    490, 491, 500, 501, 502, 503, 504, 505, 507, 508, 509, 510,
})

# Author decision (BRIEF.md, 2026-10-06): in the cross-system performance set,
# DS-1000 shape parameters are compile-time constants bound to their DS-1000
# test-case value; the runtime variants stay in the acceptance set
# (ds1000_numpy/generated/). A shape parameter is one whose value fixes the
# shape of an intermediate or output array or a trip count. Each entry says
# what the value controls.
SHAPE_CONSTANTS: dict[int, dict[str, str]] = {
    301: {"ncol": "number of output columns (reshape)"},
    302: {"nrow": "number of output rows (reshape)"},
    303: {"ncol": "number of output columns and kept prefix length"},
    304: {"ncol": "number of output columns and kept suffix length"},
    327: {"length": "output length (pad width)"},
    328: {"length": "output length and which branch runs"},
    329: {"power": "exponent (number of multiplications); listed by the author"},
    330: {"power": "exponent (number of multiplications); listed by the author"},
    362: {"del_col": "which columns are deleted, hence the output width"},
    383: {"N": "output length (number of indices kept)"},
    388: {"patch_size": "patch shape and number of patches"},
    389: {"h": "output rows", "w": "output columns"},
    390: {"patch_size": "patch shape and number of patches"},
    391: {"low": "slice start, hence output width", "high": "slice stop, hence output width"},
    392: {"low": "slice start, hence output height", "high": "slice stop, hence output height"},
    393: {"low": "slice start, hence output width", "high": "slice stop, hence output width"},
    402: {"dims": "the shape the index is raveled into (DS-1000 draws `a` with this shape)"},
    403: {"dims": "the shape the index is raveled into (DS-1000 draws `a` with this shape)"},
    411: {"low_index": "crop position and crop shape", "high_index": "crop shape"},
    414: {"bin_size": "bin width, hence the number of bins"},
    415: {"bin_size": "bin width, hence the number of bins"},
    416: {"bin_size": "bin width, hence the number of bins"},
    417: {"bin_size": "bin width, hence the number of bins"},
    418: {"bin_size": "bin width, hence the number of bins"},
    419: {"bin_size": "bin width, hence the number of bins"},
    425: {"m": "number of output bits per value (output width)"},
    426: {"m": "number of output bits per value (output width)"},
    427: {"m": "number of output bits (output width)"},
    457: {"dim": "input array width (DS-1000 constructs a with this many columns)"},
    458: {"dim": "input array width (DS-1000 constructs a with this many columns)"},
    467: {"size": "window shape (slice extents of every window)"},
    468: {"size": "window shape (slice extents of every window)"},
    495: {"shape": "output shape (pad widths)"},
    496: {"shape": "output shape (pad widths)"},
    497: {"shape": "output shape (pad widths)"},
    498: {"shape": "output shape (pad widths)"},
    499: {"shape": "output shape (pad widths)"},
}

DEGENERATE: dict[int, str] = {
    291: "returns a.shape; the annotation fixes it to (2, 5, 5, 2), so `a` is never read",
    349: "returns a.shape[1]; the annotation fixes it to 6, so `a` is never read",
    373: "returns np.arange(1, n+1)/n for n = len(grades) = 50, a constant; the sort is dead code",
    400: "no inputs; returns the constant np.array([])",
    401: "no inputs; returns the constant np.array([[], [], []])",
    451: "no inputs; returns the constant np.zeros((20, 10, 10, 2))",
}

# Duplicates that renaming local variables does not reveal.
SEMANTIC_DUPLICATES: dict[int, tuple[int, str]] = {
    330: (329, "computes `a ** power` like 0329; 0329 assigns the result back to `a`, 0330 to `result`"),
}

# Kernels whose outputs depend on private inputs only by copying them (and
# constants): no arithmetic or comparison on private values. They stay in the
# performance set (their output does depend on the inputs) but are marked so the
# analysis can report them separately. Checked against the fixtures by
# `perf_fixtures.py` (every output value must be an input value or a constant).
DATA_MOVEMENT_ONLY: dict[int, str] = {
    301: "reshape", 302: "reshape", 303: "slice + reshape", 304: "slice, reverse, reshape",
    317: "np.array of an array (identity copy)", 327: "pad with zeros", 328: "copy, zero a tail",
    336: "anti-diagonal", 337: "anti-diagonal", 338: "diagonal and anti-diagonal", 339: "diagonals of a slice",
    340: "flatten", 341: "flatten", 342: "flatten", 343: "flatten transposed",
    359: "delete a column", 360: "delete a row", 361: "delete two columns", 362: "delete a column",
    385: "reshape and transpose", 386: "overlapping 2x2 sliding windows", 387: "reshape and transpose", 388: "patches (reshape)",
    389: "reshape", 390: "patches (reshape)", 391: "column slice", 392: "row slice", 393: "column slice",
    411: "copy a crop into zeros", 435: "overwrite a fixed row and column with 0",
    448: "interleave two arrays", 449: "interleave two arrays", 467: "sliding windows (slices)",
    468: "sliding windows (slices)", 471: "slice", 472: "slice", 477: "diagonal with zero padding", 495: "pad with zeros",
    496: "pad with zeros", 497: "pad with a private constant", 498: "pad with zeros",
    499: "pad with zeros (centred)", 500: "reshape",
}


def case_dirs() -> list[Path]:
    return sorted(PERF_DIR.glob("ds1000_case[0-9][0-9][0-9][0-9]"))


def problem_id(case_dir: Path) -> int:
    return int(case_dir.name[-4:])


def expected_kernel(record: dict[str, Any]) -> generator.GeneratedKernel:
    """The performance-set kernel for an in-scope record."""
    constants = list(SHAPE_CONSTANTS.get(record["problem_id"], {}))
    kernel = generator.generate(record, constants)
    # Existing author-approved contract (FILTERING.md, permutation inputs;
    # commit 2561803d): uninitialized inverse-permutation cells are undefined
    # unless the private index array is a permutation. Keep the generator's
    # acceptance form unchanged; reproduce the performance contract here.
    if record["problem_id"] in (318, 319):
        kernel.source = kernel.source.replace(
            "@zk_circuit\n", "@zk_circuit\n@requires(lambda a, permutation: is_permutation(permutation))\n")
    return kernel


def kernel_ast(source: str) -> str:
    """A comparison key that ignores comments and formatting."""
    return ast.dump(ast.parse(source))


class _Alpha(ast.NodeTransformer):
    """Renames every non-builtin name to v0, v1, ... in order of appearance."""

    KEEP = set(records.BUILTIN_NAMES)

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def _rename(self, name: str) -> str:
        return name if name in self.KEEP else self.names.setdefault(name, f"v{len(self.names)}")

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._rename(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._rename(node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = self._rename(node.name)
        self.generic_visit(node)
        return node


def computation_key(source: str) -> str:
    """Same key <=> same kernel up to renaming (annotations are kept)."""
    return ast.dump(_Alpha().visit(ast.parse(source)))


@dataclass
class Duplicate:
    of: int
    reason: str


def find_duplicates(kernels: dict[int, str]) -> dict[int, Duplicate]:
    """Map each duplicate case to the lowest-id case with the same computation."""
    first: dict[str, int] = {}
    found: dict[int, Duplicate] = {}
    for pid in sorted(kernels):
        key = computation_key(kernels[pid])
        if key in first:
            found[pid] = Duplicate(first[key], f"same kernel as {first[key]:04d} after renaming local variables")
        else:
            first[key] = pid
    for pid, (of, reason) in SEMANTIC_DUPLICATES.items():
        if pid in kernels:
            found.setdefault(pid, Duplicate(of, reason))
    return found


# ---------------------------------------------------------------------------
# Functional duplicates
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A case with a kernel and fixture inputs, for the functional duplicate test."""

    kernel: Path
    parameters: list[dict[str, Any]]     # fixture.json "parameters"
    inputs: list[list[Any]]              # argument lists (fixture inputs A and B)
    data_movement_only: bool


def _outputs(candidate: Candidate, flat_args: list[Any]) -> list[complex] | None:
    """The kernel's outputs as one flat list (rounded to 9 decimals), or None on error."""
    import numpy_reference

    args = [np.asarray(v).reshape(p["shape"]) if p["kind"] == "array" else v
            for v, p in zip(flat_args, candidate.parameters)]
    try:
        result = numpy_reference.run_kernel(candidate.kernel, args)
    except Exception:  # noqa: BLE001 - an input this kernel does not accept: not a duplicate
        return None

    def flat(value: Any) -> list[complex]:
        if isinstance(value, (list, tuple)):
            return [x for item in value for x in flat(item)]
        return [complex(round(complex(v).real, 9), round(complex(v).imag, 9)) for v in np.asarray(value).ravel()]

    return flat(result)


def _flat_inputs(candidate: Candidate) -> list[list[Any]]:
    return [[np.asarray(a).ravel() if p["kind"] == "array" else a for a, p in zip(args, candidate.parameters)]
            for args in candidate.inputs]


def _sizes(candidate: Candidate) -> list[tuple[str, int]]:
    return [(p["dtype"], int(np.prod(p.get("shape", [1])))) for p in candidate.parameters]


def same_up_to_reordering(x: Candidate, y: Candidate) -> bool:
    """Fed the same input values (reshaped to each kernel's shapes), y's outputs are a
    fixed reordering of x's: the multisets of per-position value tuples over all test
    inputs are equal. Outputs of data-dependent length must be identical."""
    if _sizes(x) != _sizes(y):
        return False
    inputs = _flat_inputs(x) + _flat_inputs(y)
    xs = [_outputs(x, i) for i in inputs]
    ys = [_outputs(y, i) for i in inputs]
    if any(o is None for o in xs + ys):
        return False
    if len({len(o) for o in xs + ys}) == 1:
        return Counter(zip(*xs)) == Counter(zip(*ys))
    return all(a == b for a, b in zip(xs, ys))


def copy_signature(candidate: Candidate) -> tuple | None:
    """For a data-movement kernel: how often each read input cell is copied to an
    output, and the constant outputs. Every input cell gets a distinct value, so each
    output is identified as a copy of one cell or as a constant."""
    values, start = [], 1000
    for p in candidate.parameters:
        n = int(np.prod(p.get("shape", [1])))
        cells = np.arange(start, start + n) + (0.5 if p["dtype"] == "float" else 0)
        values.append(cells if p["kind"] == "array" else cells[0].item())
        start += n
    outputs = _outputs(candidate, values)
    if outputs is None:
        return None
    cells = {complex(c) for v in values for c in np.atleast_1d(v).tolist()}
    copies = Counter(o for o in outputs if o in cells)
    constants = Counter(o for o in outputs if o not in cells)
    return tuple(sorted(copies.values())), tuple(sorted((c.real, c.imag, n) for c, n in constants.items()))


def functional_duplicates(candidates: dict[int, Candidate]) -> dict[int, Duplicate]:
    """Cases whose outputs equal a lower-id case's up to a fixed reordering or
    reshaping. Two numeric tests:

    * same input sizes and dtypes: `same_up_to_reordering` on the fixture inputs of
      both cases;
    * data-movement kernels (every output is a copy of an input cell or a constant):
      the same `copy_signature`, i.e. the same number of cells copied the same number
      of times and the same constants. Input cells no output reads are ignored: no
      system needs to load them (0303 copies 20 of its 23 cells, like 0301).

    Each class is represented by its lowest id."""
    parent = {pid: pid for pid in candidates}

    def find(pid: int) -> int:
        while parent[pid] != pid:
            pid = parent[pid]
        return pid

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    reasons: dict[int, str] = {}
    for x, y in itertools.combinations(sorted(candidates), 2):
        if same_up_to_reordering(candidates[x], candidates[y]):
            reasons.setdefault(y, "same outputs up to a fixed reordering or reshaping, on the same inputs")
            union(x, y)
    by_signature: dict[tuple, list[int]] = {}
    for pid in sorted(candidates):
        signature = copy_signature(candidates[pid]) if candidates[pid].data_movement_only else None
        if signature is not None:
            by_signature.setdefault(signature, []).append(pid)
    for signature, group in by_signature.items():
        for y in group[1:]:
            reasons.setdefault(y, f"copies {len(signature[0])} input cells to outputs with no arithmetic "
                                  "(only which cells, their order and unread input cells differ)")
            union(group[0], y)
    return {pid: Duplicate(find(pid), f"{reasons[pid]}, like {find(pid):04d}")
            for pid in sorted(candidates) if find(pid) != pid}
