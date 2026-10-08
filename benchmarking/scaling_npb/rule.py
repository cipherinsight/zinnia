"""The size rule of the scaling study, applied at a cap.

The rule is the acceptance study's (`datasets/tools/acceptance_sizes.py`) with
CAP as a parameter: every array dimension and every size or trip-count scalar
is at most CAP, and the relations among dimensions are kept. The study applies
it at CAP = 32, 64 and 128; CAP = 16 is the acceptance study's own cap, the
sizes of the performance set.

NPBench (as acceptance_sizes.py states it): every integer parameter p of
NPBench's preset S becomes max(min(p, 2), ceil(p / d)) for the smallest whole
d such that NPBench's `initialize()` succeeds and every integer parameter,
integer input and array dimension is at most CAP. The arrays come from
`initialize()` at those parameters.

Pythran: the upstream example line (the first `#runas` line of the function,
else the first `#bench` line) with the same formula applied to its size knobs.
A size knob is a number written in the line that sets an array length (or a
size scalar); the knobs of each line are the numbers the acceptance study's
reduction to CAP = 16 changed (`acceptance_sizes.PYTHRAN_SMALL_EXAMPLES`,
its "what changed" column), so which numbers count as sizes was fixed before
this study. Each line is written below as a template, the upstream line with
every knob replaced by `{expression}` over the knob names; the generator
checks that the template at the upstream knob values gives the upstream
line's arguments. d is the smallest whole number for which the line runs and
every array dimension and integer scalar it passes is at most CAP.

Lines are prepared the same way at every cap (as the CAP = 16 fixtures were):
uninitialised buffers (`np.empty`, `np.empty_like`) become zeros, an import of
the benchmark's own module is dropped (the function is in the kernel), and
NumPy's global generator is seeded with 0 before the line runs.

The rule's clause for loops over permutations (n = 4 items) holds at every
cap. An entry the rule cannot scale is "fixed size" with the reason; nothing
is repaired by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CAPS = (32, 64, 128)
#: The acceptance study's cap: the sizes of the performance set.
REFERENCE_CAP = 16


def scaled_knob(p: int, d: int) -> int:
    """NPBench's preset-T formula: p divided by d, rounded up, never below min(p, 2)."""
    return max(min(p, 2), -(-p // d))


class NotScalable(Exception):
    """The rule cannot scale the entry at this cap (the message is the reason)."""


# ---------------------------------------------------------------------------
# Pythran example lines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExampleLine:
    """An upstream example line with its size knobs marked."""

    #: The upstream `#runas` (or `#bench`) line, verbatim.
    upstream: str
    #: The prepared upstream line with each knob written as `{expression}`.
    template: str
    #: Knob name -> its value in the upstream line.
    knobs: dict[str, int]
    #: Set when the input values come from the CAP-16 fixture's Q32 generator
    #: (`acceptance_fixtures.PYTHRAN_Q32_INPUTS`) instead of the upstream line;
    #: the template then is that generator's line, and only its shapes are
    #: compared with the upstream line's.
    q32_values: bool = False


#: (program, function) -> its example line, for every Pythran performance entry
#: whose upstream example passes an array dimension above 16. The comment gives
#: the CAP-16 reduction the knobs are read from.
PYTHRAN_LINES: dict[tuple[str, str], ExampleLine] = {
    # "#bench line (no #runas); u = d = 300 -> 16"
    ("pythran_allpairs", "sqr_dists"): ExampleLine(
        "u = 300; d = 300; import numpy as np; b = np.ones((u,d)); a = np.ones((u,d)); sqr_dists(a, b)",
        "u = {u}; d = {d}; import numpy as np; b = np.ones((u,d)); a = np.ones((u,d)); sqr_dists(a, b)",
        {"u": 300, "d": 300}),
    # "N 1000 -> 16"
    ("pythran_create_grid", "create_grid"): ExampleLine(
        "import numpy as np ; N = 1000 ; x = np.arange(0,1,1./N) ; create_grid(x)",
        "import numpy as np ; N = {N} ; x = np.arange(0,1,1./N) ; create_grid(x)",
        {"N": 1000}),
    # "N 800 -> 16"
    ("pythran_cronbach", "cronbach"): ExampleLine(
        "import numpy as np ; N = 800 ; items = np.arange(N*N, dtype=float).reshape(N,N) ; cronbach(items)",
        "import numpy as np ; N = {N} ; items = np.arange(N*N, dtype=float).reshape(N,N) ; cronbach(items)",
        {"N": 800}),
    # "grid 50x50 -> 16x16; the 9 lattice directions of D2Q9 kept"
    ("pythran_d2q9_nxnyns", "one_time_step"): ExampleLine(
        "import numpy as np ; r = np.ones((50,50,9)); one_time_step(r,r)",
        "import numpy as np ; r = np.ones(({n},{n},9)); one_time_step(r,r)",
        {"n": 50}),
    # "image 300x300 -> 16x16"
    ("pythran_deuxd_convolution", "conv"): ExampleLine(
        "import numpy as np ; x = np.tri(300,300)*0.5 ; w = np.tri(5,5)*0.25 ; conv(x,w)",
        "import numpy as np ; x = np.tri({n},{n})*0.5 ; w = np.tri(5,5)*0.25 ; conv(x,w)",
        {"n": 300}),
    # "n 20 -> 16"
    ("pythran_goodexpomeasure", "goodExpoMeasure"): ExampleLine(
        "import numpy as np; n = 20; a = np.arange(n*n*n).reshape((n,n,n)).astype(np.uint8); b = 2. ; "
        "goodExpoMeasure(a, b)",
        "import numpy as np; n = {n}; a = np.arange(n*n*n).reshape((n,n,n)).astype(np.uint8); b = 2. ; "
        "goodExpoMeasure(a, b)",
        {"n": 20}),
    # "N 100 -> 16"
    ("pythran_l2norm", "l2_norm"): ExampleLine(
        "import numpy as np ; N = 100 ; x = np.arange(N*N, dtype=np.float64).reshape((N,N)) ; l2_norm(x)",
        "import numpy as np ; N = {N} ; x = np.arange(N*N, dtype=np.float64).reshape((N,N)) ; l2_norm(x)",
        {"N": 100}),
    # "length 100 -> 16; the line's `from lap import laplacian` is dropped"
    ("pythran_lap", "laplacian"): ExampleLine(
        "import numpy as np; from lap import laplacian; var = np.arange(100, dtype=float) ; dh2=.01 ; "
        "laplacian(var, dh2)",
        "import numpy as np; var = np.arange({n}, dtype=float) ; dh2=.01 ; laplacian(var, dh2)",
        {"n": 100}),
    # "N 500 -> 16; seeded so the example is reproducible"
    ("pythran_lapl3d", "laplacien"): ExampleLine(
        "import numpy as np ; N = 500 ; X = np.random.randn(N,N,3); laplacien(X)",
        "import numpy as np ; N = {N} ; X = np.random.randn(N,N,3); laplacien(X)",
        {"N": 500}),
    # "length 100 -> 16; the output buffer b is zeros_like instead of empty_like"
    ("pythran_ln", "ln"): ExampleLine(
        "import numpy; a = numpy.arange(0, 1., 1./100); b = numpy.empty_like(a) ; ln(a,b)",
        "import numpy; a = numpy.arange(0, 1., 1./{n}); b = numpy.zeros_like(a) ; ln(a,b)",
        {"n": 100}),
    # "N 10000 -> 16"; values from the Q32 line of acceptance_fixtures.PYTHRAN_Q32_INPUTS
    ("pythran_log_likelihood", "log_likelihood"): ExampleLine(
        "import numpy as np ; N = 10000 ; a = np.arange(float(N)); b = 0.1; c =1.1; log_likelihood(a, b, c)",
        "import numpy as np ; N = {N} ; rs = np.random.RandomState(16) ; b = 0.1 ; c = 1.1 ; "
        "a = b + c * rs.uniform(-2., 2., N) ; log_likelihood(a, b, c)",
        {"N": 10000}, q32_values=True),
    # "length 160 -> 16 (symbols still every other entry, 80 -> 8)"
    ("pythran_make_decision", "md"): ExampleLine(
        "import numpy as np;  s=np.arange(160.)+np.arange(160.)*1.j ; sc = s[::2]; md(s, sc)",
        "import numpy as np;  s=np.arange({n}.)+np.arange({n}.)*1.j ; sc = s[::2]; md(s, sc)",
        {"n": 160}),
    # "60x60 -> 16x16"
    ("pythran_matmul", "matrix_multiply"): ExampleLine(
        "a=[ [ float(i) for i in range(60)] for j in range(60)] ; matrix_multiply(a,a)",
        "a=[ [ float(i) for i in range({n})] for j in range({n})] ; matrix_multiply(a,a)",
        {"n": 60}),
    # "X 200x100 -> 16x8 (2:1 kept), same value range": 20000 = 200 * 100 is computed from them
    ("pythran_pairwise", "pairwise"): ExampleLine(
        "import numpy as np ; X = np.linspace(0,10,20000).reshape(200,100) ; pairwise(X)",
        "import numpy as np ; X = np.linspace(0,10,{m*n}).reshape({m},{n}) ; pairwise(X)",
        {"m": 200, "n": 100}),
    # "points 200 -> 16 (N 20 -> 1.6 at the same 0.1 spacing); box L = 4 kept": the knob is the
    # number of points, c = N / 0.1
    ("pythran_periodic_dist", "dist"): ExampleLine(
        "import numpy as np ; N = 20 ; x = np.arange(0., N, 0.1) ; L = 4 ; periodic = True ; "
        "dist(x, x, x, L,periodic, periodic, periodic)",
        "import numpy as np ; N = {c / 10} ; x = np.arange(0., N, 0.1) ; L = 4 ; periodic = True ; "
        "dist(x, x, x, L,periodic, periodic, periodic)",
        {"c": 200}),
    # "1000 points on [0, 10) -> 16": the knob is the number of points, c = 10 / step
    ("pythran_rosen", "rosen"): ExampleLine(
        "import numpy as np; r = np.arange(0.,10., .01); rosen(r)",
        "import numpy as np; r = np.arange(0.,10., {10 / c}); rosen(r)",
        {"c": 1000}),
    # "100 points on [0, 1) -> 16": the knob is the number of points, c = 1 / step
    ("pythran_smoothing", "smoothing"): ExampleLine(
        "import numpy as np ; a = np.arange(0,1,10e-3) ; smoothing(a, .4)",
        "import numpy as np ; a = np.arange(0,1,{1 / c}) ; smoothing(a, .4)",
        {"c": 100}),
    # "100x100 -> 16x16"
    ("pythran_specialconvolve", "specialconvolve"): ExampleLine(
        "import numpy as np ; r = np.arange(100*100, dtype=np.uint32).reshape(100,100); specialconvolve(r)",
        "import numpy as np ; r = np.arange({n}*{n}, dtype=np.uint32).reshape({n},{n}); specialconvolve(r)",
        {"n": 100}),
    # "N 30 -> 16"
    ("pythran_sumarray3d", "summation"): ExampleLine(
        "import numpy as np ; N=30 ; pos = np.arange(1, N*3. + 1, dtype=np.float32).reshape((N,3)) ; "
        "w = np.ones(N, dtype=np.float32) ; p =  np.arange(N*3., dtype=np.float32).reshape((N,3)) ; "
        "summation(pos, w, p)",
        "import numpy as np ; N={N} ; pos = np.arange(1, N*3. + 1, dtype=np.float32).reshape((N,3)) ; "
        "w = np.ones(N, dtype=np.float32) ; p =  np.arange(N*3., dtype=np.float32).reshape((N,3)) ; "
        "summation(pos, w, p)",
        {"N": 30}),
    # "length 1000000 -> 16"
    ("pythran_vibr_energy", "calculate_vibr_energy"): ExampleLine(
        "import numpy as np ; a = np.sin(np.ones(1000000)) ; b = np.cos(np.ones(1000000)) ; "
        "n = np.arange(1000000); calculate_vibr_energy(a, b, n)",
        "import numpy as np ; a = np.sin(np.ones({n})) ; b = np.cos(np.ones({n})) ; "
        "n = np.arange({n}); calculate_vibr_energy(a, b, n)",
        {"n": 1000000}),
}

#: Pythran performance entries the rule keeps at their CAP-16 size at every cap, and why.
PYTHRAN_FIXED: dict[tuple[str, str], str] = {
    ("pythran_allpairs", "sqr_dists_loops"): (
        "no upstream example line calls the function (the CAP-16 example was chosen by hand: "
        "sqr_dists's #bench line), so the rule has no size knob to scale"),
    ("pythran_fbcorr", "fbcorr"): (
        "no upstream example line calls the function (the CAP-16 example was chosen by hand from "
        "the list version's formulas), so the rule has no size knob to scale"),
    ("pythran_hasting", "fweb"): (
        "no upstream example line calls the function (the CAP-16 example was chosen by hand: the "
        "three-species food web), so the rule has no size knob to scale"),
}

#: Pythran performance entries for which the rule itself gives the CAP-16 sizes at every cap.
PYTHRAN_KEPT_BY_RULE: dict[tuple[str, str], str] = {
    ("pythran_perm", "permutations"): (
        "the rule sets n = 4 for a loop that runs once per permutation of n items, whatever the cap "
        "(acceptance_sizes.py: 4! = 24 iterations), so every cap gives CAP 16's 4 items"),
}


def prepare(line: str, module: str) -> str:
    """An upstream line as every cap runs it: uninitialised buffers become zeros and the
    import of the benchmark's own module is dropped."""
    line = re.sub(r"\bempty(_like)?\(", r"zeros\1(", line)
    return re.sub(rf"\bfrom {re.escape(module)} import [\w, ]+;\s*", "", line)


def render(template: str, knobs: dict[str, int]) -> str:
    """The template with every `{expression}` evaluated over the knob values."""
    def value(match: re.Match) -> str:
        result = eval(match.group(1), {"__builtins__": {}}, dict(knobs))  # noqa: S307 - our table
        return repr(result) if isinstance(result, float) else str(result)
    return re.sub(r"\{([^{}]+)\}", value, template)


# ---------------------------------------------------------------------------
# The divisor search, shared by both corpora
# ---------------------------------------------------------------------------


@dataclass
class Choice:
    """The knob values the rule picks at a cap, and what they give."""

    knobs: dict[str, Any]
    divisor: int
    values: Any = None          # what the probe returned for these knobs
    tried: list[str] = field(default_factory=list)   # why smaller divisors failed


def is_int(value: Any) -> bool:
    import numpy as np
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def within_cap(inputs: list[Any], cap: int, exempt: set[int] = frozenset()) -> str | None:
    """Why the inputs break the cap (None when every integer scalar and dimension fits)."""
    import numpy as np
    for value in inputs:
        if is_int(value) and value > cap:
            return f"integer input {value} > {cap}"
        if isinstance(value, np.ndarray):
            big = [d for d in value.shape if d > cap and d not in exempt]
            if big:
                return f"dimension {big[0]} > {cap}"
    return None


def choose(knobs: dict[str, Any], cap: int, probe) -> Choice:
    """The smallest whole d whose scaled knobs pass `probe` (returns (values, problem));
    integer knobs scale, others stay. Raises NotScalable when no d works."""
    scalable = [k for k, v in knobs.items() if is_int(v)]
    largest = max([knobs[k] for k in scalable] or [1])
    tried: list[str] = []
    for d in range(1, largest + 1):
        candidate = dict(knobs)
        candidate.update({k: scaled_knob(knobs[k], d) for k in scalable})
        if any(candidate[k] > cap for k in scalable):
            continue
        values, problem = probe(candidate)
        if problem is None:
            return Choice(candidate, d, values, tried)
        if len(tried) < 3:
            tried.append(f"d={d}: {problem}")
    raise NotScalable(f"no divisor gives sizes within {cap}: " + "; ".join(tried))
