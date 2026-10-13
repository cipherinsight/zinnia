"""Shared pieces of the tightness study: the array inventory, the input domain, and the
runtime size of every inventoried array on a given input.

The inventory is `../final/arrays.json` (36 programs, 141 arrays, the set behind the
paper's 54% fill). A row's id is `<site>#<occurrence><path>`: the source expression
(line:col:endline:endcol:kind), which evaluation of it (loops and repeated calls give
several), and the position inside a tuple/list result. Sizes are measured exactly the way
`../final/probe.py` and `../final/collect_arrays.py` measured the A/B lengths: the kernel
is run as NumPy with every ndarray-valued expression observed (`probe.Observe`), after the
same desugaring of Boolean-mask augmented assignments. The three branch-merged crop
results (0507, 0509, 0510) are the kernel's return value.
"""
from __future__ import annotations

import ast
import collections
import copy
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent / 'final'
# The checkout whose `benchmarking` harness and `zinnia` package are used. Defaults to the
# repository this folder lives in; set ZINNIA_CHECKOUT to a checkout with a built extension.
ROOT = Path(os.environ.setdefault('ZINNIA_CHECKOUT', str(HERE.parents[3])))
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(FINAL))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import probe  # noqa: E402  (../final/probe.py: the observation transformer)
from benchmarking.harness.case_sets import load_acceptance_programs  # noqa: E402
from benchmarking.harness.fixtures import load_fixture  # noqa: E402
from benchmarking.harness.reference import kernel_namespace  # noqa: E402

# Value domain the compiler assumes for an input (src/ir_defs.rs, src/helpers/dtype.rs):
# Integer is int64; Float is Q32 fixed point, a value v stored as the int64 code
# round(v * 2^32), so |v| < 2^31; Complex is a pair of Q32 floats; Boolean is 0/1.
INT_MIN, INT_MAX = -2**63, 2**63 - 1
FLOAT_LIMIT = 2.0**31 - 2.0**-32   # largest |v| whose Q32 code fits int64


def ident(p):
    return f'{p.dataset}__{p.program}__{p.function}'


ROWS = json.loads((FINAL / 'arrays.json').read_text())
PROGRAM_IDS = sorted({r['program'] for r in ROWS})
_ENTRIES = {ident(p): p for p in load_acceptance_programs(
    ROOT / 'benchmarking/datasets/case_sets/acceptance.json')}
PROGRAMS = {i: _ENTRIES[i] for i in PROGRAM_IDS}


def rows_of(pid):
    return [r for r in ROWS if r['program'] == pid]


def short(pid):
    return pid.split('__')[1]


class Program:
    """One benchmark program: its parameters, fixtures and inventoried arrays."""

    def __init__(self, pid):
        self.id = pid
        self.entry = PROGRAMS[pid]
        self.fixture = load_fixture(self.entry.fixture_path)
        self.params = self.fixture.parameters
        self.rows = rows_of(pid)
        tree = ast.parse(self.entry.kernel_path.read_text())
        self.tree = ast.fix_missing_locations(probe.DesugarSubscriptAugAssign().visit(tree))
        self.names = probe.candidates(self.tree)
        observed = probe.Observe().visit(copy.deepcopy(self.tree))
        self.code = compile(ast.fix_missing_locations(observed), str(self.entry.kernel_path), 'exec')
        self.plain = compile(self.entry.kernel_path.read_text().replace('from zinnia import *', ''),
                             str(self.entry.kernel_path), 'exec')

    # ---- inputs ------------------------------------------------------------------------
    def fixture_input(self, label):
        """A fixture input as a {name: JSON value} dict."""
        return self.fixture.values(label)

    def arguments(self, values):
        """Kernel arguments of declared dtype and shape from {name: JSON value}."""
        from benchmarking.harness.fixtures import _argument
        return [_argument(p, values[p.name]) for p in self.params]

    def in_domain(self, values):
        """Whether every input value lies in the compiler's assumed value domain."""
        for p in self.params:
            if p.dtype == 'int':
                v = np.asarray(values[p.name], dtype=object).ravel()
                if not all(isinstance(x, (int, np.integer)) and INT_MIN <= int(x) <= INT_MAX for x in v):
                    return False
        for p, a in zip(self.params, self.arguments(values)):
            arr = np.asarray(a)
            if p.dtype == 'float':
                if not np.all(np.isfinite(arr)) or np.any(np.abs(arr) > FLOAT_LIMIT):
                    return False
            elif p.dtype == 'complex':
                for part in (arr.real, arr.imag):
                    if not np.all(np.isfinite(part)) or np.any(np.abs(part) > FLOAT_LIMIT):
                        return False
        return True

    # ---- sizes -------------------------------------------------------------------------
    def sizes(self, values):
        """Runtime size (element count) of every inventoried array on one input, as
        {array id: size}, or None when the input is not valid: outside the value domain,
        or the NumPy program raises (including floating-point errors, since non-finite
        values are outside the benchmark's rules)."""
        if not self.in_domain(values):
            return None
        seen = collections.defaultdict(list)

        def observe(k, v):
            if isinstance(v, np.ndarray) or (isinstance(v, (tuple, list)) and any(isinstance(x, np.ndarray) for x in v)):
                seen[k].append(_sizes(v))
            return v

        ns = kernel_namespace()
        ns['_observe'] = observe
        try:
            with np.errstate(all='raise'), warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                exec(self.code, ns)
                ns[self.entry.function](*self.arguments(values))
                ret = None
                if any(r['source'] == 'original_output' for r in self.rows):
                    ns2 = kernel_namespace()
                    exec(self.plain, ns2)
                    ret = ns2[self.entry.function](*self.arguments(values))
        except Exception:
            return None
        out = {}
        for r in self.rows:
            if r['source'] == 'original_output':
                out[r['array']] = int(np.asarray(ret).size)
                continue
            site, rest = r['array'].split('#', 1)
            nth = int(rest.split('[', 1)[0])
            try:
                v = seen[site][nth]
                for i in r['path'][2:]:
                    v = v[i]
                out[r['array']] = v['length']
            except (IndexError, KeyError, TypeError):
                out[r['array']] = 0   # the expression was not evaluated on this input
        return out


def _sizes(x):
    if isinstance(x, np.ndarray):
        return dict(shape=list(x.shape), length=int(x.size))
    if isinstance(x, (tuple, list)):
        return [_sizes(y) for y in x]
    return None


def to_json(values):
    """Make an input dict JSON-serialisable (complex numbers as [re, im] pairs)."""
    def conv(v):
        if isinstance(v, np.ndarray):
            return conv(v.tolist())
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, (complex, np.complexfloating)):
            return [float(v.real), float(v.imag)]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    return {k: conv(v) for k, v in values.items()}


def capacity(row):
    return int(row['capacity'])


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))
