"""Regression tests for the second soundness review of array-ssa
(`eval-redo/layer/array-ssa-review/REVIEW-2.md`): every reproducer of
`r2/q_*.py`, on the mock, halo2-base and Noir.

Each reproducer has one of four expected outcomes:

- *matches NumPy*: the mock's outputs equal NumPy's, and halo2-base and Noir
  prove the mock's values;
- *rejected as stale*: the program reads an array that NumPy shares with an
  array written afterwards, where Zinnia holds a copy that misses the write
  (`src/builder/stale.rs`); a compile error, never a result without the write;
- *rejected* for another reason NumPy (or Zinnia's merge) cannot be followed:
  a write to an array NumPy returns read-only (N10), or one name bound to
  arrays of different shapes on different paths (N7);
- *unsupported*: a construct Zinnia does not implement (`np.rollaxis`,
  chained subscript assignment through a copy, ...). The test asks only that
  such a program is rejected or computes NumPy's result.

The findings:

- N1: a method call (`t.sum()`, `t.copy()`, `t.T`, ...) read a stale array
  without the check. Every read of an array's cells now goes through the
  array store, which rejects stale cells whatever the syntax.
- N2: `out=` writes of `np.clip` and reductions were not recorded. Every
  write is recorded by the store, so the arrays linked to its target become
  stale.
- N3: the check looked one container level deep. It is part of the store
  read now, so the depth does not matter.
- N4: a merge read a stale operand. A merge carries its operands' staleness
  to its result, conjoined with the branch condition.
- N5: `np.asarray(x, dtype=<x's dtype>)`, `np.array(x, copy=False/None)` and
  `x.astype(<dtype>, copy=False)` return `x` itself in NumPy 2.2.
- N6: an assignment in a loop in an `else`/`elif` branch was lost
  (`IRContext::set`).
- N7: a name bound to arrays of different shapes on paths a loop break
  selects is rejected, as under an `if`.
- N8: a reshape NumPy does without a copy (a strided slice, a column, a
  transpose to the same shape) is linked to its source.
- N9: `np.real` / `np.imag` of a complex array merged under a condition.
- N10: writes to the read-only results of `np.broadcast_to` and of the
  imaginary part of a real array are compile errors.
- N11: `np.expand_dims` of a runtime-shaped array merged under a condition
  published the old rank.
- F1, F4: the reviewer's re-checks of the first review's findings.
- Fuzz: the classes the reviewer's fuzzers found when re-run on this
  branch's first commit (see that section).
"""
import itertools

import numpy as np
import pytest

from testing.lang.all_backends import (assert_all_backends_match_numpy, halo2_field_publics,
                                       mock_field_publics, nargo_available, noir_field_publics)
from testing.lang.mock_outputs import compile_kernel, numpy_in_place_outputs, run_mock_layout
from zinnia.debug.exception.base import ZinniaException

X6 = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
X6B = np.array([11, 21, 31, 41, 51, 61], dtype=np.int64)
XS6 = np.array([3, -1, 4, -6, 5, 9], dtype=np.int64)
M34 = np.arange(12, dtype=np.int64).reshape(3, 4)
M45 = np.arange(1, 21, dtype=np.int64).reshape(4, 5)
F3 = np.array([0.5, 1.5, 2.5])
F6 = np.array([1.5, -2.0, 3.25, -4.0, 5.0, 0.5])
F6B = np.array([2.5, -1.0, 4.25, -3.0, 6.0, 1.5])
Z4F = np.zeros(4)
M34F = np.arange(12, dtype=np.float64).reshape(3, 4)
C3 = np.array([1 + 2j, 3 - 1j, -2 + 0.5j])

STALE = "is not supported here: .*linked|is not supported here: a name that refers"
READ_ONLY = "assignment destination is read-only"
SHAPES = "different shapes|different ranks"

# ── N1: reads through a method call ─────────────────────────────────────────

N1_STALE = {
    'method: stale flip read via t.sum()': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.sum()\n', (X6,)),
    'method: stale flip read via t.copy()': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.copy()\n', (X6,)),
    'method: stale flip read via t.flatten()': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.flatten()\n', (X6,)),
    'method: stale flip read via t.max()': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.max()\n', (X6,)),
    'method: stale flip read via t.reshape(2, 3)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.reshape(2, 3)\n', (X6,)),
    'method: stale flip read via t.T': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.T\n', (X6,)),
    'method: stale flip read via t.astype(np.int64)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.astype(np.int64)\n', (X6,)),
    'method: stale flip read via t.tolist()': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.tolist()\n', (X6,)),
    'method: stale flip read via t.mean()': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.mean()\n', (X6,)),
    'method: stale flip read via t.cumsum()': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.cumsum()\n', (X6,)),
    'method: stale flip read via t.shape[0] + t[0]': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.shape[0] + t[0]\n', (X6,)),
    'method: stale flip read via t + 0': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t + 0\n', (X6,)),
    'method: stale alias via method [True]': ('def kernel(x: Integer[Array, "6"], w: Integer[Array, "6"], c: Boolean):\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    return x.sum(), w.sum()\n', (X6, X6B, True,)),
    'method: stale alias via method copy [True]': ('def kernel(x: Integer[Array, "6"], w: Integer[Array, "6"], c: Boolean):\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    return x.copy()\n', (X6, X6B, True,)),
    'method: stale reversed slice via method': ('def kernel(x: Integer[Array, "6"]):\n    r = x[::-1]\n    x[0] = 99\n    return r.copy()\n', (X6,)),
    'rr: rr view flatten after runtime write [-1, 5]': ('def kernel(a: Integer[Array, "4 5"], i: Integer, j: Integer):\n    v2 = a[::-1][::-1]\n    a[3, (j + 3) % 5] = 0\n    v4 = v2.flatten()\n    return a, v4\n', (M45, -1, 5,)),
    'rr: rr view read after runtime write [-1, 5]': ('def kernel(a: Integer[Array, "4 5"], i: Integer, j: Integer):\n    v2 = a[::-1][::-1]\n    a[3, (j + 3) % 5] = 0\n    return v2\n', (M45, -1, 5,)),
    'rr: rr view const write [-1, 5]': ('def kernel(a: Integer[Array, "4 5"], i: Integer, j: Integer):\n    v2 = a[::-1][::-1]\n    a[3, 3] = 0\n    return v2.flatten()\n', (M45, -1, 5,)),
}

N1_MATCH = {
    'rr: plain view flatten after runtime write [-1, 5]': ('def kernel(a: Integer[Array, "4 5"], i: Integer, j: Integer):\n    v2 = a[1:]\n    a[3, (j + 3) % 5] = 0\n    return v2.flatten()\n', (M45, -1, 5,)),
}

N1_UNSUPPORTED = {
    'method: write through stale method target': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    t.reshape(2, 3)[0, 0] = 99\n    return x\n', (X6,)),
}

# ── N2, N3: writes through out=, stale arrays in nested containers ──────────

N2_STALE = {
    'f4b: np.clip out=x (reversed slice)': ('def kernel(x: Float[Array, "6"]):\n    t = x[::-1]\n    np.clip(x, -1.0, 1.0, out=x)\n    return t\n', (F6,)),
    'f4b: x.clip out=x': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    x.clip(-1.0, 1.0, out=x)\n    return t\n', (F6,)),
    'f4b: np.clip positional out': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    np.clip(x, -1.0, 1.0, x)\n    return t\n', (F6,)),
    'f4b: np.cumsum out=x': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    np.cumsum(x, out=x)\n    return t\n', (F6,)),
    'f4b: np.sum axis out=y': ('def kernel(m: Float[Array, "3 4"], y: Float[Array, "4"]):\n    t = np.flip(y)\n    np.sum(m, axis=0, out=y)\n    return t\n', (M34F, Z4F,)),
    'f4b: np.max axis out=y': ('def kernel(m: Float[Array, "3 4"], y: Float[Array, "4"]):\n    t = np.flip(y)\n    np.max(m, axis=0, out=y)\n    return t\n', (M34F, Z4F,)),
    'f4b: np.round out=x': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    np.round(x, 0, out=x)\n    return t\n', (F6,)),
    'f4b: np.negative out=x': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    np.negative(x, out=x)\n    return t\n', (F6,)),
    'f4b: np.maximum out=x': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    np.maximum(x, 0.0, out=x)\n    return t\n', (F6,)),
    'f4b: np.where then copyto': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    np.copyto(x, 0.0, where=x > 1.0)\n    return t\n', (F6,)),
    'f4b: np.dot out': ('def kernel(m: Float[Array, "3 4"], y: Float[Array, "4"]):\n    t = np.flip(y)\n    np.dot(m, np.ones(4), out=y[:3])\n    return t\n', (M34F, Z4F,)),
    'f4b: np.matmul out': ('def kernel(m: Float[Array, "3 4"], y: Float[Array, "4"]):\n    t = np.flip(y)\n    np.matmul(m, np.ones(4), out=y[:3])\n    return t\n', (M34F, Z4F,)),
    'f4b: x[:] = in fn via out': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    x[:] = 0.0\n    return t\n', (F6,)),
    'f4b: x *= via name': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    x *= 2.0\n    return t\n', (F6,)),
    'f4b: x[...] =': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    x[...] = 1.0\n    return t\n', (F6,)),
    'f4b: list containing flip, nested': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    L = [[t, t], 1]\n    x[0] = 99.0\n    return L[0][1]\n', (F6,)),
    'f4b: tuple of list': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    T = ([t],)\n    x[0] = 99.0\n    return T[0][0]\n', (F6,)),
    'f4b: list then index into name': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    L = [[t]]\n    u = L[0]\n    x[0] = 99.0\n    return u[0]\n', (F6,)),
    'f4b: dict-free: list comprehension of views': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    L = [t[k:] for k in range(2)]\n    x[0] = 99.0\n    return L[1]\n', (F6,)),
    'f4c: np.clip out, path-dependent alias [True]': ('def kernel(x: Float[Array, "6"], w: Float[Array, "6"], c: Boolean):\n    z = w\n    if c:\n        z = x\n    np.clip(z, -1.0, 1.0, out=z)\n    return x, w\n', (F6, F6B, True,)),
}

N2_MATCH = {
    'f4c: np.sum out=y returns y': ('def kernel(m: Float[Array, "3 4"], y: Float[Array, "4"]):\n    np.sum(m, axis=0, out=y)\n    return y\n', (M34F, Z4F,)),
    'f4c: np.sum out=y through view': ('def kernel(m: Float[Array, "3 4"], y: Float[Array, "4"]):\n    v = y[1:]\n    np.sum(m[:, 1:], axis=0, out=v)\n    return y\n', (M34F, Z4F,)),
    'f4c: np.clip out=x returns x': ('def kernel(x: Float[Array, "6"]):\n    np.clip(x, -1.0, 1.0, out=x)\n    return x\n', (F6,)),
    'f4c: np.clip out= view': ('def kernel(x: Float[Array, "6"]):\n    v = x[2:]\n    np.clip(v, -1.0, 1.0, out=v)\n    return x\n', (F6,)),
    'f4c: np.sum out=y, aliased name': ('def kernel(m: Float[Array, "3 4"], y: Float[Array, "4"]):\n    z = y\n    np.sum(m, axis=0, out=y)\n    return z\n', (M34F, Z4F,)),
    'f4c: np.clip out, alias name': ('def kernel(x: Float[Array, "6"]):\n    z = x\n    np.clip(x, -1.0, 1.0, out=x)\n    return z\n', (F6,)),
}

N2_UNSUPPORTED = {
    'f4b: np.add.accumulate out=x': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    np.add.accumulate(x, out=x)\n    return t\n', (F6,)),
}

# ── F4: other aliasing paths, N10 read-only results ─────────────────────────

F4_STALE = {
    'f4: flip then slice, write slice': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    u = t[1:]\n    u[0] = 99\n    return x\n', (X6,)),
    'f4: slice of flip expr, write': ('def kernel(x: Integer[Array, "6"]):\n    u = np.flip(x)[2:]\n    u[0] = 99\n    return x\n', (X6,)),
    'f4: flip then base write, read slice': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    u = t[1:]\n    x[0] = 99\n    return u\n', (X6,)),
    'f4: flip of flip': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(np.flip(x))\n    t[0] = 99\n    return x\n', (X6,)),
    'f4: flip of view': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x[1:4])\n    t[0] = 99\n    return x\n', (X6,)),
    'f4: view of view of base, then flip read': ('def kernel(x: Integer[Array, "6"]):\n    v = x[1:]\n    t = np.flip(v)\n    x[5] = 99\n    return t\n', (X6,)),
    'f4: flip axis arg': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.flip(m, 1)\n    t[0, 0] = 99\n    return m\n', (M34,)),
    'f4: flip axis kw': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.flip(m, axis=0)\n    t[0, 0] = 99\n    return m\n', (M34,)),
    'f4: rot90 k=2': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.rot90(m, 2)\n    t[0, 0] = 99\n    return m\n', (M34,)),
    'f4: m[::-1, :] write': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m[::-1, :]\n    t[0, 0] = 99\n    return m\n', (M34,)),
    'f4: m[:, ::-1] write': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m[:, ::-1]\n    t[0, 0] = 99\n    return m\n', (M34,)),
    'f4: alias of flip': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    s = t\n    s[0] = 99\n    return x\n', (X6,)),
    'f4: flip in list, base write': ('def kernel(x: Integer[Array, "6"]):\n    L = [np.flip(x)]\n    x[0] = 99\n    return L[0]\n', (X6,)),
    'f4: flip in nested list': ('def kernel(x: Integer[Array, "6"]):\n    L = [[np.flip(x)]]\n    x[0] = 99\n    return L[0][0]\n', (X6,)),
    'f4: flip in tuple of tuple': ('def kernel(x: Integer[Array, "6"]):\n    T = ((np.flip(x), 1), 2)\n    x[0] = 99\n    return T[0][0]\n', (X6,)),
    'f4: base aug op': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x += 1\n    return t\n', (X6,)),
    'f4: base aug op, reversed slice': ('def kernel(x: Integer[Array, "6"]):\n    t = x[::-1]\n    x += 1\n    return t\n', (X6,)),
    'f4: base view write': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    v = x[2:]\n    v[0] = 99\n    return t\n', (X6,)),
    'f4: base out=': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    np.add(x, 1, out=x)\n    return t\n', (X6,)),
    'f4: base copyto': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    np.copyto(x, np.arange(6))\n    return t\n', (X6,)),
    'f4: base putmask': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    np.putmask(x, x > 30, 0)\n    return t\n', (X6,)),
    'f4: base clip out': ('def kernel(x: Float[Array, "6"]):\n    t = np.flip(x)\n    np.clip(x, -1.0, 1.0, out=x)\n    return t\n', (F6,)),
    'f4: base mask write': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[x > 30] = 0\n    return t\n', (X6,)),
    'f4: base write in fn': ('def f(a):\n    a[0] = 99\n\ndef kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    f(x)\n    return t\n', (X6,)),
    'f4: flip passed to fn that writes': ('def f(a):\n    a[0] = 99\n\ndef kernel(x: Integer[Array, "6"]):\n    f(np.flip(x))\n    return x\n', (X6,)),
    'f4: fn returns flip, caller writes': ('def h(a):\n    return np.flip(a)\n\ndef kernel(x: Integer[Array, "6"]):\n    t = h(x)\n    t[0] = 99\n    return x\n', (X6,)),
    'f4: out= into flip': ('def kernel(x: Integer[Array, "6"]):\n    np.add(x, 1, out=np.flip(x))\n    return x\n', (X6,)),
    'f4: out= into flip name': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    np.add(x, 1, out=t)\n    return x\n', (X6,)),
    'f4: in-place kernel through flip': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    t[0] = 99\n', (X6,)),
    'f4: stale read via sum': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return np.sum(t)\n', (X6,)),
    'f4: stale read via len/iter': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    s = 0\n    for v in t:\n        s = s + v\n    return s\n', (X6,)),
    'f4: stale via branch [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    return t\n', (X6, True,)),
    'f4: stale via branch false [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    return t\n', (X6, False,)),
    'f4: atleast_3d': ('def kernel(x: Integer[Array, "6"]):\n    t = np.atleast_3d(x)\n    t[0, 1, 0] = 99\n    return x\n', (X6,)),
    'f4: reshape of strided view (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = x[::2].reshape(3)\n    t[1] = 99\n    return x, t\n', (X6,)),
    'f4: atleast_1d of flip': ('def kernel(x: Integer[Array, "6"]):\n    t = np.atleast_1d(np.flip(x))\n    t[0] = 99\n    return x\n', (X6,)),
    'f4: asarray of flip, base write': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(np.flip(x))\n    x[0] = 99\n    return t\n', (X6,)),
}

F4_READONLY = {
    'f4: broadcast_to write (numpy read-only)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.broadcast_to(x, (2, 6))\n    t[0, 0] = 99\n    return t\n', (X6,)),
    'f4: np.imag of real (read-only)': ('def kernel(x: Float[Array, "6"]):\n    t = np.imag(x)\n    t[0] = 9.0\n    return t\n', (F6,)),
}

F4_MATCH = {
    'f4: np.real view (real array)': ('def kernel(x: Float[Array, "6"]):\n    t = np.real(x)\n    t[0] = 9.0\n    return x\n', (F6,)),
    'f4: x.real view': ('def kernel(x: Float[Array, "6"]):\n    t = x.real\n    t[0] = 9.0\n    return x\n', (F6,)),
    'f4: np.squeeze': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.squeeze(m[1:2])\n    t[1] = 99\n    return m\n', (M34,)),
    'f4: m.squeeze()': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m[1:2].squeeze()\n    t[1] = 99\n    return m\n', (M34,)),
    'f4: np.expand_dims': ('def kernel(x: Integer[Array, "6"]):\n    t = np.expand_dims(x, 1)\n    t[2, 0] = 99\n    return x\n', (X6,)),
    'f4: np.transpose axes': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.transpose(m, (1, 0))\n    t[2, 1] = 99\n    return m\n', (M34,)),
    'f4: np.moveaxis': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.moveaxis(m, 1, 0)\n    t[2, 1] = 99\n    return m\n', (M34,)),
    'f4: np.reshape non-contig (copy in numpy)': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.reshape(m.T, (12,))\n    t[1] = 99\n    return m, t\n', (M34,)),
    'f4: ravel of transposed (copy)': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m.T.ravel()\n    t[1] = 99\n    return m, t\n', (M34,)),
    'f4: np.reshape contiguous (view)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.reshape(x, (2, 3))\n    t[1, 1] = 99\n    return x\n', (X6,)),
    'f4: np.asarray dtype float (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x, dtype=float)\n    t[0] = 99.0\n    return x, t\n', (X6,)),
    'f4: np.asarray of view': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x[1:])\n    t[0] = 99\n    return x\n', (X6,)),
    'f4: np.array of view': ('def kernel(x: Integer[Array, "6"]):\n    t = np.array(x[1:])\n    t[0] = 99\n    return x, t\n', (X6,)),
    'f4: np.split piece write': ('def kernel(x: Integer[Array, "6"]):\n    a, b = np.split(x, 2)\n    b[0] = 99\n    return x\n', (X6,)),
    'f4: np.diag 2d (view, readonly)': ('def kernel(m: Integer[Array, "3 4"]):\n    d = np.diag(m)\n    return d\n', (M34,)),
    'f4: x.T on 1d': ('def kernel(x: Integer[Array, "6"]):\n    t = x.T\n    t[0] = 99\n    return x\n', (X6,)),
}

F4_UNSUPPORTED = {
    'f4: rollaxis': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.rollaxis(m, 1)\n    t[0, 1] = 99\n    return m\n', (M34,)),
    'f4: flip into list, write via list': ('def kernel(x: Integer[Array, "6"]):\n    L = [np.flip(x)]\n    L[0][0] = 99\n    return x\n', (X6,)),
    'f4: np.hsplit 1d': ('def kernel(x: Integer[Array, "6"]):\n    p = np.hsplit(x, 3)\n    p[2][0] = 99\n    return x\n', (X6,)),
    'f4: np.vsplit': ('def kernel(m: Integer[Array, "3 4"]):\n    p = np.vsplit(m, 3)\n    p[1][0, 0] = 99\n    return m\n', (M34,)),
    'f4: ndarray.flat': ('def kernel(x: Integer[Array, "6"]):\n    x.flat[2] = 99\n    return x\n', (X6,)),
}

# ── N4: merges, N7: different shapes on different paths ─────────────────────

N4_STALE = {
    'merge: stale through merge [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    a = x[::-1]\n    x[5] = 99\n    if c:\n        a = x[::2]\n    return a\n', (XS6, False,)),
    'merge: stale through merge (flip) [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    a = np.flip(x)\n    x[5] = 99\n    if c:\n        a = x * 2\n    return a\n', (XS6, False,)),
}

# The last-but-one case is N7 (a loop break selects the binding).
N4_SHAPES = {
    'merge: merge different shapes, branch not taken [False]': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], c: Boolean):\n    if c:\n        m = x[1:3]\n    return m\n', (XS6, M34, False,)),
    'merge: merge different shapes, branch taken [True]': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], c: Boolean):\n    if c:\n        m = x[1:3]\n    return m\n', (XS6, M34, True,)),
    'merge: merge different shapes via loop break [False]': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], c: Boolean):\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::2]\n    return m\n', (XS6, M34, False,)),
    'merge: merge 2D vs row [False]': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], c: Boolean):\n    t = m\n    if c:\n        t = m[0]\n    return t\n', (XS6, M34, False,)),
}

# ── N5: functions that return their argument itself ─────────────────────────

N5_MATCH = {
    'asarray: asarray dtype=np.int64': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x, dtype=np.int64)\n    t[0] = 99\n    return x\n', (X6,)),
    'asarray: asarray dtype=int': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x, dtype=int)\n    t[0] = 99\n    return x\n', (X6,)),
    'asarray: asarray positional dtype': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x, np.int64)\n    t[0] = 99\n    return x\n', (X6,)),
    'asarray: array copy=False': ('def kernel(x: Integer[Array, "6"]):\n    t = np.array(x, copy=False)\n    t[0] = 99\n    return x\n', (X6,)),
    'asarray: array copy=None': ('def kernel(x: Integer[Array, "6"]):\n    t = np.array(x, copy=None)\n    t[0] = 99\n    return x\n', (X6,)),
    'asarray: astype copy=False': ('def kernel(x: Integer[Array, "6"]):\n    t = x.astype(np.int64, copy=False)\n    t[0] = 99\n    return x\n', (X6,)),
    'asarray: astype same dtype default copies': ('def kernel(x: Integer[Array, "6"]):\n    t = x.astype(np.int64)\n    t[0] = 99\n    return x, t\n', (X6,)),
    'asarray: asarray then base write read': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x, dtype=np.int64)\n    x[0] = 99\n    return t\n', (X6,)),
    'asarray: asarray view of 2D': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.asarray(m[1])\n    t[0] = 99\n    return m\n', (M34,)),
    'asarray: atleast_2d of 2D': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.atleast_2d(m)\n    t[0, 0] = 99\n    return m\n', (M34,)),
    'asarray: atleast_2d of 1D base write': ('def kernel(x: Integer[Array, "6"]):\n    t = np.atleast_2d(x)\n    x[0] = 99\n    return t\n', (X6,)),
    'asarray: atleast_1d of scalar [5]': ('def kernel(a: Integer):\n    t = np.atleast_1d(a)\n    t[0] = 99\n    return t, a\n', (5,)),
    'asarray: np.asarray of list [5]': ('def kernel(a: Integer):\n    L = [a, 2]\n    t = np.asarray(L)\n    t[0] = 99\n    return t, L[0]\n', (5,)),
}

# Zinnia has one float dtype for every width, so it cannot tell whether
# NumPy returns `x` itself; it links the result, and a write makes the
# other array stale.
N5_STALE = {
    'asarray: asarray float dtype=float': ('def kernel(x: Float[Array, "6"]):\n    t = np.asarray(x, dtype=float)\n    t[0] = 9.0\n    return x\n', (F6,)),
    'asarray: asarray float dtype=np.float64': ('def kernel(x: Float[Array, "6"]):\n    t = np.asarray(x, dtype=np.float64)\n    t[0] = 9.0\n    return x\n', (F6,)),
}

# ── N6: assignments in a loop in an else or elif branch ─────────────────────

N6_MATCH = {
    'loopelse: loop in else (no break) [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    if c:\n        s = s + 100\n    else:\n        for k in range(2):\n            s = s + 1\n    return s\n', (XS6, False,)),
    'loopelse: loop with break in else [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    if c:\n        s = s + 100\n    else:\n        for k in range(2):\n            if x[k] > 3:\n                break\n            s = s + 1\n    return s\n', (XS6, False,)),
    'loopelse: loop with break in if [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    if c:\n        for k in range(2):\n            if x[k] > 3:\n                break\n            s = s + 1\n    return s\n', (XS6, True,)),
    'loopelse: loop with break in if, not taken [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    if c:\n        for k in range(2):\n            if x[k] > 3:\n                break\n            s = s + 1\n    return s\n', (XS6, False,)),
    'loopelse: loop with break, top level [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    for k in range(2):\n        if x[k] > 3:\n            break\n        s = s + 1\n    return s\n', (XS6, False,)),
    'loopelse: loop with break in else, if branch writes s [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    if c:\n        s = 7\n    else:\n        for k in range(2):\n            if x[k] > 3:\n                break\n            s = s + 1\n    return s\n', (XS6, False,)),
    'loopelse: loop with break in else, if taken [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    if c:\n        s = s + 100\n    else:\n        for k in range(2):\n            if x[k] > 3:\n                break\n            s = s + 1\n    return s\n', (XS6, True,)),
    'loopelse: array write in loop-with-break in else [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    if c:\n        x[0] = 100\n    else:\n        for k in range(2):\n            if x[k] > 3:\n                break\n            x[5] = x[5] + 1\n    return x\n', (XS6, False,)),
    'loopelse: rebind in loop-with-break in else [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    t = x * 1\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            if x[k] > 3:\n                break\n            t = t + 1\n    return t\n', (XS6, False,)),
    'loopelse2: minimal: for in else [False]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        s = s + 100\n    else:\n        for k in range(2):\n            s = s + 1\n    return s\n', (False,)),
    'loopelse2: for in else, runtime c true [True]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        s = s + 100\n    else:\n        for k in range(2):\n            s = s + 1\n    return s\n', (True,)),
    'loopelse2: while in else [False]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        k = 0\n        while k < 2:\n            s = s + 1\n            k = k + 1\n    return s\n', (False,)),
    'loopelse2: for in elif [3]': ('def kernel(a: Integer):\n    s = 0\n    if a > 5:\n        s = 100\n    elif a > 0:\n        for k in range(2):\n            s = s + 1\n    return s\n', (3,)),
    'loopelse2: for in else, no if-branch assignment [False]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        pass\n    else:\n        for k in range(2):\n            s = s + 1\n    return s\n', (False,)),
    'loopelse2: for in else, static condition [3]': ('def kernel(a: Integer):\n    s = 0\n    if 1 > 2:\n        s = 100\n    else:\n        for k in range(2):\n            s = s + a\n    return s\n', (3,)),
    'loopelse2: for in if-branch (control) [True]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        for k in range(2):\n            s = s + 1\n    else:\n        s = 100\n    return s\n', (True,)),
    'loopelse2: nested for in for in else [False]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(2):\n            for q in range(2):\n                s = s + 1\n    return s\n', (False,)),
    'loopelse3: loops in both branches [True]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        for k in range(2):\n            s = s + 1\n    else:\n        for k in range(2):\n            s = s + 10\n    return s\n', (True,)),
    'loopelse3: loops in both branches [False]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        for k in range(2):\n            s = s + 1\n    else:\n        for k in range(2):\n            s = s + 10\n    return s\n', (False,)),
    'loopelse3: if-branch reads only [False]': ('def kernel(c: Boolean):\n    s = 0\n    t = 0\n    if c:\n        t = s + 1\n    else:\n        for k in range(2):\n            s = s + 1\n    return s, t\n', (False,)),
    'loopelse3: loop var assigned in if-branch of loop body [False]': ('def kernel(c: Boolean):\n    s = 0\n    if c:\n        s = 5\n    else:\n        k = 0\n        for q in range(3):\n            k = k + 1\n        s = k\n    return s\n', (False,)),
}

# ── N8: reshapes NumPy does without a copy ──────────────────────────────────

N8_STALE = {
    'reshape: strided reshape same shape': ('def kernel(x: Integer[Array, "6"]):\n    t = x[::2].reshape(3)\n    t[1] = 99\n    return x\n', (X6,)),
    'reshape: strided reshape (3,1)': ('def kernel(x: Integer[Array, "6"]):\n    t = x[::2].reshape(3, 1)\n    t[1, 0] = 99\n    return x\n', (X6,)),
    'reshape: column reshape same': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m[:, 1].reshape(3)\n    t[1] = 99\n    return m\n', (M34,)),
    'reshape: column reshape (3,1)': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m[:, 1].reshape(3, 1)\n    t[1, 0] = 99\n    return m\n', (M34,)),
    'reshape: transposed reshape same shape': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m.T.reshape(4, 3)\n    t[1, 2] = 99\n    return m\n', (M34,)),
    'reshape: np.reshape strided same': ('def kernel(x: Integer[Array, "6"]):\n    t = np.reshape(x[::2], (3,))\n    t[1] = 99\n    return x\n', (X6,)),
    'reshape: strided reshape, base write, read': ('def kernel(x: Integer[Array, "6"]):\n    t = x[::2].reshape(3)\n    x[2] = 99\n    return t\n', (X6,)),
}

N8_MATCH = {
    'reshape: transposed reshape (copy in numpy)': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m.T.reshape(3, 4)\n    t[1, 2] = 99\n    return m, t\n', (M34,)),
}

# ── N9: the real and imaginary parts of a merged complex array ──────────────

N9_MATCH = {
    'real: np.real of complex expr': ('def kernel(x: Float[Array, "3"]):\n    t = x * 1j\n    return np.real(t)\n', (F3,)),
    'real: np.imag of complex expr': ('def kernel(x: Float[Array, "3"]):\n    t = x + 2j\n    return np.imag(t)\n', (F3,)),
    'real: np.real of complex input': ('def kernel(z: Complex[Array, "3"]):\n    return np.real(z)\n', (C3,)),
    'real: np.imag of complex input': ('def kernel(z: Complex[Array, "3"]):\n    return np.imag(z)\n', (C3,)),
    'real: np.real sum': ('def kernel(z: Complex[Array, "3"]):\n    return np.sum(np.real(z))\n', (C3,)),
    'real: np.abs complex': ('def kernel(z: Complex[Array, "3"]):\n    return np.real(z) + np.imag(z)\n', (C3,)),
    'real2: merged complex real [False]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return np.real(t)\n', (F3, False,)),
    'real2: merged complex sum [False]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return np.sum(t)\n', (F3, False,)),
    'real2: merged complex abs [False]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return np.abs(t)\n', (F3, False,)),
    'real2: merged complex itself [False]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return t\n', (F3, False,)),
    'real2: merged complex t[1] [False]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return t[1]\n', (F3, False,)),
    'real2: merged complex real [True]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return np.real(t)\n', (F3, True,)),
    'real2: merged complex sum [True]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return np.sum(t)\n', (F3, True,)),
    'real2: merged complex abs [True]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return np.abs(t)\n', (F3, True,)),
    'real2: merged complex itself [True]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return t\n', (F3, True,)),
    'real2: merged complex t[1] [True]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return t[1]\n', (F3, True,)),
}

N9_UNSUPPORTED = {
    'real2: merged complex .real [False]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return t.real\n', (F3, False,)),
    'real2: merged complex .real [True]': ('def kernel(x: Float[Array, "3"], c: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n    return t.real\n', (F3, True,)),
}

# ── N11: the rank of a merged runtime-shaped array ──────────────────────────

N11_MATCH = {
    'layout: expand_dims of dyn': ('def kernel(x: Integer[Array, "6"]):\n    y = x[x > 0]\n    return np.expand_dims(y, 0)\n', (XS6,)),
    'layout: expand_dims of atleast_1d dyn': ('def kernel(x: Integer[Array, "6"]):\n    y = x[x > 0]\n    m = np.atleast_1d(y)\n    return np.expand_dims(m, 0)\n', (XS6,)),
    'layout: expand_dims of dyn slice [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    y = x[::2]\n    if c:\n        y = y[1:]\n    return np.expand_dims(y, 0)\n', (XS6, False,)),
    'layout: expand_dims of dyn slice [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    y = x[::2]\n    if c:\n        y = y[1:]\n    return np.expand_dims(y, 0)\n', (XS6, True,)),
    'layout: atleast_1d dyn': ('def kernel(x: Integer[Array, "6"]):\n    y = x[x > 0]\n    return np.atleast_1d(y)\n', (XS6,)),
}

# ── F1: rebinding under runtime conditions ──────────────────────────────────

F1_MATCH = {
    'f1: ternary rebind then write [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    t = x[3:] * 1 if c else t\n    if c:\n        t[1] = 7\n    return t\n', (X6, 1, False, False,)),
    'f1: ternary rebind then write [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    t = x[3:] * 1 if c else t\n    if c:\n        t[1] = 7\n    return t\n', (X6, 1, False, True,)),
    'f1: ternary rebind then write [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    t = x[3:] * 1 if c else t\n    if c:\n        t[1] = 7\n    return t\n', (X6, 1, True, False,)),
    'f1: ternary rebind then write [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    t = x[3:] * 1 if c else t\n    if c:\n        t[1] = 7\n    return t\n', (X6, 1, True, True,)),
    'f1: elif rebind [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t[0] = 1\n    elif d:\n        t = x[3:] * 1\n        t[2] = 9\n    return t\n', (X6, 1, False, False,)),
    'f1: elif rebind [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t[0] = 1\n    elif d:\n        t = x[3:] * 1\n        t[2] = 9\n    return t\n', (X6, 1, False, True,)),
    'f1: elif rebind [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t[0] = 1\n    elif d:\n        t = x[3:] * 1\n        t[2] = 9\n    return t\n', (X6, 1, True, False,)),
    'f1: elif rebind [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t[0] = 1\n    elif d:\n        t = x[3:] * 1\n        t[2] = 9\n    return t\n', (X6, 1, True, True,)),
    'f1: rebind, write in later if [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n    if c:\n        t[1] = 7\n    return t\n', (X6, 1, False, False,)),
    'f1: rebind, write in later if [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n    if c:\n        t[1] = 7\n    return t\n', (X6, 1, False, True,)),
    'f1: rebind, write in later if [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n    if c:\n        t[1] = 7\n    return t\n', (X6, 1, True, False,)),
    'f1: rebind, write in later if [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n    if c:\n        t[1] = 7\n    return t\n', (X6, 1, True, True,)),
    'f1: rebind, write in else-if same cond [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n    if not c:\n        pass\n    else:\n        t[i] = 7\n    return t\n', (X6, 1, False, False,)),
    'f1: rebind, write in else-if same cond [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n    if not c:\n        pass\n    else:\n        t[i] = 7\n    return t\n', (X6, 1, False, True,)),
    'f1: rebind, write in else-if same cond [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n    if not c:\n        pass\n    else:\n        t[i] = 7\n    return t\n', (X6, 1, True, False,)),
    'f1: rebind, write in else-if same cond [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n    if not c:\n        pass\n    else:\n        t[i] = 7\n    return t\n', (X6, 1, True, True,)),
    'f1: helper fn write [1, False, False]': ('def f(a):\n    a[1] = 7\n\ndef kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        f(t)\n    return t\n', (X6, 1, False, False,)),
    'f1: helper fn write [1, False, True]': ('def f(a):\n    a[1] = 7\n\ndef kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        f(t)\n    return t\n', (X6, 1, False, True,)),
    'f1: helper fn write [1, True, False]': ('def f(a):\n    a[1] = 7\n\ndef kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        f(t)\n    return t\n', (X6, 1, True, False,)),
    'f1: helper fn write [1, True, True]': ('def f(a):\n    a[1] = 7\n\ndef kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        f(t)\n    return t\n', (X6, 1, True, True,)),
    'f1: helper fn returns new, write [1, False, False]': ('def g(a):\n    b = a + 1\n    b[0] = 0\n    return b\n\ndef kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = g(x[3:])\n    return t\n', (X6, 1, False, False,)),
    'f1: helper fn returns new, write [1, False, True]': ('def g(a):\n    b = a + 1\n    b[0] = 0\n    return b\n\ndef kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = g(x[3:])\n    return t\n', (X6, 1, False, True,)),
    'f1: helper fn returns new, write [1, True, False]': ('def g(a):\n    b = a + 1\n    b[0] = 0\n    return b\n\ndef kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = g(x[3:])\n    return t\n', (X6, 1, True, False,)),
    'f1: helper fn returns new, write [1, True, True]': ('def g(a):\n    b = a + 1\n    b[0] = 0\n    return b\n\ndef kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = g(x[3:])\n    return t\n', (X6, 1, True, True,)),
    'f1: copyto in branch [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.copyto(t, np.array([1, 2, 3]))\n    return t\n', (X6, 1, False, False,)),
    'f1: copyto in branch [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.copyto(t, np.array([1, 2, 3]))\n    return t\n', (X6, 1, False, True,)),
    'f1: copyto in branch [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.copyto(t, np.array([1, 2, 3]))\n    return t\n', (X6, 1, True, False,)),
    'f1: copyto in branch [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.copyto(t, np.array([1, 2, 3]))\n    return t\n', (X6, 1, True, True,)),
    'f1: putmask in branch [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.putmask(t, t > 45, 0)\n    return t\n', (X6, 1, False, False,)),
    'f1: putmask in branch [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.putmask(t, t > 45, 0)\n    return t\n', (X6, 1, False, True,)),
    'f1: putmask in branch [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.putmask(t, t > 45, 0)\n    return t\n', (X6, 1, True, False,)),
    'f1: putmask in branch [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.putmask(t, t > 45, 0)\n    return t\n', (X6, 1, True, True,)),
    'f1: out= ufunc in branch [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.multiply(t, 2, out=t)\n    return t\n', (X6, 1, False, False,)),
    'f1: out= ufunc in branch [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.multiply(t, 2, out=t)\n    return t\n', (X6, 1, False, True,)),
    'f1: out= ufunc in branch [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.multiply(t, 2, out=t)\n    return t\n', (X6, 1, True, False,)),
    'f1: out= ufunc in branch [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        np.multiply(t, 2, out=t)\n    return t\n', (X6, 1, True, True,)),
    'f1: tuple unpack rebind [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    u = x[3:] * 1\n    if c:\n        t, u = u * 2, t * 2\n        t[0] = 5\n        u[i] = 6\n    return t, u\n', (X6, 1, False, False,)),
    'f1: tuple unpack rebind [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    u = x[3:] * 1\n    if c:\n        t, u = u * 2, t * 2\n        t[0] = 5\n        u[i] = 6\n    return t, u\n', (X6, 1, False, True,)),
    'f1: tuple unpack rebind [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    u = x[3:] * 1\n    if c:\n        t, u = u * 2, t * 2\n        t[0] = 5\n        u[i] = 6\n    return t, u\n', (X6, 1, True, False,)),
    'f1: tuple unpack rebind [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    u = x[3:] * 1\n    if c:\n        t, u = u * 2, t * 2\n        t[0] = 5\n        u[i] = 6\n    return t, u\n', (X6, 1, True, True,)),
    'f1: while rebind break [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    k = 0\n    while k < 3:\n        if x[k] > 15:\n            break\n        t = x[3:] * 1\n        t[k] = -1\n        k += 1\n    return t\n', (X6, 1, False, False,)),
    'f1: while rebind break [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    k = 0\n    while k < 3:\n        if x[k] > 15:\n            break\n        t = x[3:] * 1\n        t[k] = -1\n        k += 1\n    return t\n', (X6, 1, False, True,)),
    'f1: while rebind break [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    k = 0\n    while k < 3:\n        if x[k] > 15:\n            break\n        t = x[3:] * 1\n        t[k] = -1\n        k += 1\n    return t\n', (X6, 1, True, False,)),
    'f1: while rebind break [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    k = 0\n    while k < 3:\n        if x[k] > 15:\n            break\n        t = x[3:] * 1\n        t[k] = -1\n        k += 1\n    return t\n', (X6, 1, True, True,)),
    'f1: for over array rebind [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    for v in x[:2]:\n        if v > i:\n            t = x[3:] + v\n            t[2] = 0\n    return t\n', (X6, 1, False, False,)),
    'f1: for over array rebind [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    for v in x[:2]:\n        if v > i:\n            t = x[3:] + v\n            t[2] = 0\n    return t\n', (X6, 1, False, True,)),
    'f1: for over array rebind [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    for v in x[:2]:\n        if v > i:\n            t = x[3:] + v\n            t[2] = 0\n    return t\n', (X6, 1, True, False,)),
    'f1: for over array rebind [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    for v in x[:2]:\n        if v > i:\n            t = x[3:] + v\n            t[2] = 0\n    return t\n', (X6, 1, True, True,)),
    'f1: nested rebind both levels [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        if d:\n            t = x[:3] * 2\n            t[0] = 1\n        t[1] = 2\n    return t\n', (X6, 1, False, False,)),
    'f1: nested rebind both levels [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        if d:\n            t = x[:3] * 2\n            t[0] = 1\n        t[1] = 2\n    return t\n', (X6, 1, False, True,)),
    'f1: nested rebind both levels [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        if d:\n            t = x[:3] * 2\n            t[0] = 1\n        t[1] = 2\n    return t\n', (X6, 1, True, False,)),
    'f1: nested rebind both levels [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        if d:\n            t = x[:3] * 2\n            t[0] = 1\n        t[1] = 2\n    return t\n', (X6, 1, True, True,)),
    'f1: rebind then mask write [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        t[t > 45] = 0\n    return t\n', (X6, 1, False, False,)),
    'f1: rebind then mask write [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        t[t > 45] = 0\n    return t\n', (X6, 1, False, True,)),
    'f1: rebind then mask write [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        t[t > 45] = 0\n    return t\n', (X6, 1, True, False,)),
    'f1: rebind then mask write [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        t[t > 45] = 0\n    return t\n', (X6, 1, True, True,)),
    'f1: rebind then slice aug [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        t[1:] += 5\n    return t\n', (X6, 1, False, False,)),
    'f1: rebind then slice aug [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        t[1:] += 5\n    return t\n', (X6, 1, False, True,)),
    'f1: rebind then slice aug [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        t[1:] += 5\n    return t\n', (X6, 1, True, False,)),
    'f1: rebind then slice aug [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        t[1:] += 5\n    return t\n', (X6, 1, True, True,)),
    'f1: 2D rebind fancy [1, False]': ('def kernel(m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n    t = m * 1\n    if c:\n        t = m + 1\n        t[np.array([0, 2]), i] = -1\n    return t\n', (M34, 1, False,)),
    'f1: 2D rebind fancy [1, True]': ('def kernel(m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n    t = m * 1\n    if c:\n        t = m + 1\n        t[np.array([0, 2]), i] = -1\n    return t\n', (M34, 1, True,)),
    'f1: float rebind out= [1, False, False]': ('def kernel(x: Float[Array, "3"], i: Integer, c: Boolean, d: Boolean):\n    t = x * 1.0\n    if c:\n        t = x + 1.0\n        np.add(t, 0.5, out=t)\n    return t\n', (F3, 1, False, False,)),
    'f1: float rebind out= [1, True, False]': ('def kernel(x: Float[Array, "3"], i: Integer, c: Boolean, d: Boolean):\n    t = x * 1.0\n    if c:\n        t = x + 1.0\n        np.add(t, 0.5, out=t)\n    return t\n', (F3, 1, True, False,)),
    'f1: dyn rebind runtime [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    y = x[x > 0]\n    if c:\n        y = x[x > 30]\n        y[i] = 100\n    return y\n', (X6, 1, False, False,)),
    'f1: dyn rebind runtime [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    y = x[x > 0]\n    if c:\n        y = x[x > 30]\n        y[i] = 100\n    return y\n', (X6, 1, False, True,)),
    'f1: dyn rebind runtime [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    y = x[x > 0]\n    if c:\n        y = x[x > 30]\n        y[i] = 100\n    return y\n', (X6, 1, True, False,)),
    'f1: dyn rebind runtime [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    y = x[x > 0]\n    if c:\n        y = x[x > 30]\n        y[i] = 100\n    return y\n', (X6, 1, True, True,)),
    'f1: complex rebind [1, False, False]': ('def kernel(x: Float[Array, "3"], i: Integer, c: Boolean, d: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n        t[i] = 2 + 3j\n    return np.real(t), np.imag(t)\n', (F3, 1, False, False,)),
    'f1: complex rebind [1, True, False]': ('def kernel(x: Float[Array, "3"], i: Integer, c: Boolean, d: Boolean):\n    t = x * 1j\n    if c:\n        t = x + 0j\n        t[i] = 2 + 3j\n    return np.real(t), np.imag(t)\n', (F3, 1, True, False,)),
    'f1: comprehension guard write [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        s = [t[k] for k in range(3) if k > i]\n        t[0] = len(s)\n    return t\n', (X6, 1, False, False,)),
    'f1: comprehension guard write [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        s = [t[k] for k in range(3) if k > i]\n        t[0] = len(s)\n    return t\n', (X6, 1, False, True,)),
    'f1: comprehension guard write [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        s = [t[k] for k in range(3) if k > i]\n        t[0] = len(s)\n    return t\n', (X6, 1, True, False,)),
    'f1: comprehension guard write [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n        s = [t[k] for k in range(3) if k > i]\n        t[0] = len(s)\n    return t\n', (X6, 1, True, True,)),
}

F1_STALE = {
    'f1: rebind view of base [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3]\n    if c:\n        t = x[3:]\n        t[1] = 7\n    return x\n', (X6, 1, False, False,)),
    'f1: rebind view of base [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3]\n    if c:\n        t = x[3:]\n        t[1] = 7\n    return x\n', (X6, 1, False, True,)),
    'f1: rebind view of base [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3]\n    if c:\n        t = x[3:]\n        t[1] = 7\n    return x\n', (X6, 1, True, False,)),
    'f1: rebind view of base [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    t = x[:3]\n    if c:\n        t = x[3:]\n        t[1] = 7\n    return x\n', (X6, 1, True, True,)),
}

F1_UNSUPPORTED = {
    'f1: list element rebind [1, False, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    L = [x[:3] * 1, x[3:] * 1]\n    if c:\n        L[0] = x[3:] + 1\n        L[0][1] = 7\n    return L[0], L[1]\n', (X6, 1, False, False,)),
    'f1: list element rebind [1, False, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    L = [x[:3] * 1, x[3:] * 1]\n    if c:\n        L[0] = x[3:] + 1\n        L[0][1] = 7\n    return L[0], L[1]\n', (X6, 1, False, True,)),
    'f1: list element rebind [1, True, False]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    L = [x[:3] * 1, x[3:] * 1]\n    if c:\n        L[0] = x[3:] + 1\n        L[0][1] = 7\n    return L[0], L[1]\n', (X6, 1, True, False,)),
    'f1: list element rebind [1, True, True]': ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean, d: Boolean):\n    L = [x[:3] * 1, x[3:] * 1]\n    if c:\n        L[0] = x[3:] + 1\n        L[0][1] = 7\n    return L[0], L[1]\n', (X6, 1, True, True,)),
}

# ── N6: more control flow around a loop in a branch ──────────────────────

N6_H = 'def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n'
N6_CONTROL_FLOW = {
    "branch inside the loop in else": N6_H + '    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(3):\n            if x[k] > 0:\n                s = s + 1\n            else:\n                s = s - 10\n    return s\n',
    "loop in else inside a loop": N6_H + '    s = 0\n    for j in range(2):\n        if x[j] > 3:\n            s = s + 100\n        else:\n            for k in range(2):\n                s = s + 1\n    return s\n',
    "while with break in else": N6_H + '    s = 0\n    if c:\n        s = 100\n    else:\n        k = 0\n        while k < 4:\n            if x[k] < 0:\n                break\n            s = s + x[k]\n            k = k + 1\n    return s\n',
    "continue in a loop in elif": N6_H + '    s = 0\n    if c:\n        s = 100\n    elif d:\n        for k in range(6):\n            if x[k] < 0:\n                continue\n            s = s + x[k]\n    else:\n        s = -1\n    return s\n',
    "return in a loop in else": N6_H + '    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(6):\n            if x[k] < 0:\n                return s\n            s = s + x[k]\n    return s + 1000\n',
    "loops in if, elif and else": N6_H + '    s = 0\n    if c:\n        for k in range(2):\n            s = s + 1\n    elif d:\n        for k in range(2):\n            s = s + 10\n    else:\n        for k in range(2):\n            s = s + 100\n    return s\n',
    "loop in an else nested in an else": N6_H + '    s = 0\n    if c:\n        s = 1\n    else:\n        if d:\n            s = 2\n        else:\n            for k in range(3):\n                s = s + 5\n    return s\n',
    "loop in a branch in a loop in else": N6_H + '    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(2):\n            if d:\n                s = s + 1000\n            else:\n                for q in range(2):\n                    s = s + 1\n    return s\n',
    "array rebinding in nested loops in elif": N6_H + '    t = x * 1\n    if c:\n        t = x * 0\n    elif d:\n        for k in range(2):\n            for q in range(2):\n                t = t + 1\n    return t\n',
    "array write in a while in else": N6_H + '    if c:\n        x[0] = 100\n    else:\n        k = 0\n        while k < 3:\n            x[k + 1] = x[k] + x[k + 1]\n            k = k + 1\n    return x\n',
}

# ── In-place kernels (no return: the outputs are the written arguments) ──

IN_PLACE_MATCH = {
    "direct": ('def kernel(x: Integer[Array, "6"]):\n    x[0] = 99\n', ["x"], (X6,)),
    "alias": ('def kernel(x: Integer[Array, "6"]):\n    t = x\n    t[0] = 99\n', ["x"], (X6,)),
    "slice view": ('def kernel(x: Integer[Array, "6"]):\n    t = x[1:]\n    t[0] = 99\n', ["x"], (X6,)),
    "transpose view 1d": ('def kernel(x: Integer[Array, "6"]):\n    t = x.T\n    t[0] = 99\n', ["x"], (X6,)),
    "asarray": ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x)\n    t[0] = 99\n', ["x"], (X6,)),
    "atleast_1d": ('def kernel(x: Integer[Array, "6"]):\n    t = np.atleast_1d(x)\n    t[2] = 99\n', ["x"], (X6,)),
    "atleast_2d": ('def kernel(x: Integer[Array, "6"]):\n    t = np.atleast_2d(x)\n    t[0, 2] = 99\n', ["x"], (X6,)),
    "np.swapaxes": ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.swapaxes(m, 0, 1)\n    t[2, 1] = 99\n', ["m"], (M34,)),
    "asarray, runtime index": ('def kernel(x: Integer[Array, "6"], i: Integer):\n    t = np.asarray(x)\n    t[i] = 99\n', ["x"], (X6, 3)),
    "flip read, write base": ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = t[0]\n', ["x"], (X6,)),
}
IN_PLACE_STALE = {
    "flip": 'def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    t[0] = 99\n',
    "reversed slice": 'def kernel(x: Integer[Array, "6"]):\n    t = x[::-1]\n    t[0] = 99\n',
}


def _cases(*tables):
    return [pytest.param(source, args, id=name) for table in tables for name, (source, args) in table.items()]


MATCH = _cases(N1_MATCH, N2_MATCH, F4_MATCH, N5_MATCH, N6_MATCH, N8_MATCH, N9_MATCH, N11_MATCH, F1_MATCH)
STALE_CASES = _cases(N1_STALE, N2_STALE, F4_STALE, N4_STALE, N5_STALE, N8_STALE, F1_STALE)
UNSUPPORTED = _cases(N1_UNSUPPORTED, N2_UNSUPPORTED, F4_UNSUPPORTED, N9_UNSUPPORTED, F1_UNSUPPORTED)


@pytest.mark.parametrize("source, args", MATCH)
def test_the_reproducer_matches_numpy(source, args):
    assert_all_backends_match_numpy(source, *args)


@pytest.mark.parametrize("source, args", STALE_CASES)
def test_reading_a_stale_array_is_rejected(source, args):
    with pytest.raises(ZinniaException, match=STALE):
        compile_kernel(source)


@pytest.mark.parametrize("source, args", _cases(F4_READONLY))
def test_a_write_to_a_read_only_result_is_rejected(source, args):
    with pytest.raises(ZinniaException, match=READ_ONLY):
        compile_kernel(source)


@pytest.mark.parametrize("source, args", _cases(N4_SHAPES))
def test_a_name_bound_to_different_shapes_on_different_paths_is_rejected(source, args):
    with pytest.raises(ZinniaException, match=SHAPES):
        compile_kernel(source)


@pytest.mark.parametrize("source, args", UNSUPPORTED)
def test_an_unsupported_reproducer_is_rejected_or_correct(source, args):
    try:
        compile_kernel(source)
    except ZinniaException:
        return
    assert_all_backends_match_numpy(source, *args)


@pytest.mark.parametrize("name", list(N6_CONTROL_FLOW))
@pytest.mark.parametrize("c, d", list(itertools.product([False, True], repeat=2)))
def test_an_assignment_in_a_loop_in_a_branch_is_kept(name, c, d):
    assert_all_backends_match_numpy(N6_CONTROL_FLOW[name], XS6, c, d)


@pytest.mark.parametrize("name", list(IN_PLACE_MATCH))
def test_an_in_place_kernel_matches_numpy(name):
    source, names, args = IN_PLACE_MATCH[name]
    program = compile_kernel(source)
    satisfied, values, _ = run_mock_layout(program, *args)
    assert satisfied, "honest witness rejected by the mock"
    expected, _ = numpy_in_place_outputs(source, names, *args)
    assert values == expected
    mock = mock_field_publics(program, *args)
    assert halo2_field_publics(program, *args) == mock, "halo2-base differs from the mock"
    if nargo_available():
        assert noir_field_publics(program, *args) == mock, "Noir differs from the mock"


@pytest.mark.parametrize("name", list(IN_PLACE_STALE))
def test_an_in_place_write_through_a_copied_view_is_rejected(name):
    with pytest.raises(ZinniaException, match=STALE):
        compile_kernel(IN_PLACE_STALE[name])


# ── Found by the reviewer's fuzzers, re-run on this branch's first commit ─
#
# - an `op=` update of a runtime-shaped array, or one that rebinds a name to
#   a new storage, did not make the arrays linked to it stale, and the new
#   storage did not take over the links (a write through a linked copy then
#   missed the array);
# - two multi-dimensional arrays of different shapes merged into a list of
#   rows, on which `m[i, j]` read row `i` and `m[i, j] = v` wrote nothing;
# - a name bound in one arm of an `if` (which the join drops) and bound again
#   in a loop that may stop before the assignment kept the loop's value on
#   every path.

FUZZ_H = 'def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n'
FUZZ_STALE = {
    "op= on a merged runtime-shaped array": FUZZ_H + '    t = x[::2]\n    if c:\n        t = x[1:]\n    t += 1\n    return x\n',
    "op= that rebinds, then a write through a copy": FUZZ_H + '    m = m * 1\n    a1 = np.fliplr(m)\n    m += 1\n    a1[1, 1] = -7\n    return m\n',
}
FUZZ_MATCH = {
    "op= on a merged runtime-shaped array, read it": FUZZ_H + '    t = x[::2]\n    if c:\n        t = x[1:]\n    t += 1\n    return t\n',
    "a name bound before the branch and the loop": FUZZ_H + '    a1 = x * 0\n    if c:\n        a1 = x * 1\n        x[2] = 7\n    for k in range(2):\n        if x[k] > 2:\n            break\n        a1 = x + 0\n    return x, a1\n',
    "a loop inside a branch rebinding a branch-local name": FUZZ_H + '    for line in range(3):\n        if x[line] > 0:\n            for j in range(2):\n                t = x[j]\n                x[j] = x[j + 1]\n                x[j + 1] = t\n    return x\n',
}
FUZZ_SHAPES = {
    "read": FUZZ_H + '    s = 0\n    if c:\n        m = m[:, ::2]\n    s = s + m[i % 3, 1]\n    return s\n',
    "write": FUZZ_H + '    m = m * 1\n    if c:\n        m = m[:, ::2]\n    m[2, 1] = 100\n    return m\n',
    "loop break": FUZZ_H + '    t = m * 1\n    for k in range(2):\n        if x[k] > 2:\n            break\n        t = m[:2] * 1\n    return t\n',
}
FUZZ_BRANCH_ONLY = FUZZ_H + '    if c:\n        a1 = x * 1\n        x[2] = 7\n    for k in range(2):\n        if x[k] > 2:\n            break\n        a1 = x + 0\n    return x, a1\n'
FUZZ_X = np.array([3, -1, 4, -6, 5, 9], dtype=np.int64)
FUZZ_M = np.array([[1, -2, 3, 0], [-4, 5, -6, 2], [7, -8, 9, -1]], dtype=np.int64)

# The fuzz programs themselves, with their scalars (i, j, c, d).
FUZZ_PROGRAMS = {
    'fz_rebind_101_auto #1254': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    m = np.expand_dims(x, 0)\n    np.putmask(x, x > 2, -2)\n    x[(i + 1) % 6] = x[-3] + -1\n    a2 = x[::2]\n    x = np.transpose(m)\n    if i > 1:\n        s = s + -9\n        if c:\n            a2[(i % 3 - 3)] *= 2\n            x = x[1:]\n    s = s + m[(i % 1 - 1), 3] + -1\n    x += 1\n    return s, x, m, a2\n', (3, 2, True, False)),
    'fz_rebind_101_auto #1996': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    m = np.flip(m)\n    if not c:\n        np.putmask(x, x > 2, -1)\n        m = m[:, ::2]\n    m[-1, (j + 1) % 4] = x[(i % 6 - 6)] + -2\n    return s, x, m\n', (0, 4, True, False)),
    'fz_rebind_101_auto #2239': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    s = s + x[(i + 0) % 6] + 2\n    x = m.copy()\n    np.putmask(m, m > 2, 5)\n    a2 = np.array(m)\n    m = np.swapaxes(a2, 0, 1)\n    if c:\n        a2[-3, -3] += x[(j % 3 - 3), -2] + 0\n        if i > 1:\n            x = a2[:, ::2]\n            m[m > 0] = -3\n    else:\n        m[(i + 1) % 4, (j + 3) % 3] *= 6\n    return s, x, m\n', (2, 1, True, True)),
    'fz_rebind_101_auto #2746': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    x = m[:, ::2]\n    np.putmask(m, m > 2, -5)\n    if i > 1:\n        m = np.atleast_1d(x)\n        for k in range(2):\n            if x[k] > 5:\n                break\n            s = s + x[(j % 3 - 3), 0] + 1\n    np.putmask(x, x > 2, 0)\n    a3 = m[1:]\n    m += 1\n    return s, x, m\n', (-3, 0, True, True)),
    'fz_rebind_102_ram #121': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    m = x[::2]\n    x[-2] *= m[(j % 3 - 3)] + 2\n    if i > 1:\n        if i > 1:\n            x = np.atleast_1d(x)\n            s = s + x[3] + 2\n        if c:\n            m = m[1:]\n            m += 1\n    return s, x, m\n', (3, 0, True, False)),
    'fz_rebind_102_ram #261': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    m[0, (j + 3) % 4] *= x[(j % 6 - 6)] + -2\n    s = s + m[0, -3] + -2\n    a1 = np.fliplr(m)\n    m += 1\n    a1[1, (i % 4 - 4)] = -7\n    np.multiply(x, 2, out=x)\n    return s, x, m\n', (1, 0, False, True)),
    'fz_rebind_102_ram #573': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    a1 = x[::2]\n    a2 = m[:, ::2]\n    s = s + 4\n    m[(i + 2) % 3, -1] = a2[-3, (i % 2 - 2)] + 1\n    if d:\n        if not c:\n            a3 = a2.copy()\n            a2 += 1\n        else:\n            m += 1\n        for k in range(2):\n            if x[k] > 0:\n                break\n            a2 = np.asarray(m)\n    a2[(i % 3 - 3), (j % 2 - 2)] = a1[(j + 2) % 3] + 2\n    s = s + x[4] + 1\n    return s, x, m, a1, a2\n', (0, 2, True, False)),
    'fz_rebind_102_ram #1778': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    m[(j + 1) % 3, (i + 3) % 4] *= -2\n    m[2, (i + 1) % 4] = x[(j + 3) % 6] + -1\n    m[(i + 3) % 3, -4] = m[(i + 0) % 3, -4] + -2\n    for k in range(2):\n        if x[k] > 6:\n            break\n        s = s + m[(i + 2) % 3, (i + 3) % 4] + -1\n    a1 = m + -2\n    if not c:\n        m = a1[:, ::2]\n        if not c:\n            s = s + a1[2, (j + 1) % 4] + -2\n            a1[(i % 3 - 3), 3] = a1[(i + 2) % 3, (j % 4 - 4)] + 0\n    m[(j % 3 - 3), (i + 3) % 2] += -9\n    return s, x, m\n', (-3, 2, False, True)),
    'fz_rebind_102_ram #2086': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    x[-6] *= 4\n    m[(j % 3 - 3), -1] *= -7\n    m += 1\n    s = s + m[(i + 1) % 3, -3] + 2\n    x[(j + 1) % 6] += -9\n    if c:\n        np.putmask(x, x > 2, 0)\n        m = m[:, ::2]\n    s = s + m[(i % 3 - 3), (i % 2 - 2)] + 1\n    return s, x, m\n', (-3, 3, True, False)),
    'fz_rebind_103_noirram #22': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    a1 = m.T\n    if not c:\n        a1 = a1 + 2\n        a1 = np.fliplr(m)\n    else:\n        if c:\n            a1[2, (j + 2) % 3] = m[-2, (i % 4 - 4)] + -2\n            s = s + a1[(j % 4 - 4), -1] + -1\n    x[x > -3] = -5\n    np.multiply(x, 2, out=x)\n    for k in range(2):\n        if x[k] > 0:\n            break\n        x = x[1:]\n    a5 = np.flip(m)\n    for k in range(2):\n        if x[k] > 6:\n            break\n        for k in range(2):\n            if x[k] > 5:\n                break\n            a1 += 1\n    return s, x, m, a5\n', (-1, 3, True, True)),
    'fz_rebind_103_noirram #87': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    if i > 1:\n        np.copyto(m, -5, where=m < 0)\n        m[0, (i % 4 - 4)] += m[(j + 3) % 3, (i + 0) % 4] + -2\n    x = m[:, ::2]\n    if i > 1:\n        s = s + -7\n        x[(i % 3 - 3), (i % 2 - 2)] *= -4\n    else:\n        x = m + 3\n    s = s + x[(j % 3 - 3), (j % 4 - 4)] + 2\n    m = np.transpose(m)\n    m[-2, 2] += m[(i % 4 - 4), (j % 3 - 3)] + -2\n    m[m > 5] = 1\n    np.multiply(m, 2, out=m)\n    return s, x, m\n', (-3, 1, False, True)),
    'fz_rebind_103_noirram #685': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    x[(j % 6 - 6)] += -3\n    if c:\n        x = np.atleast_1d(x)\n        x = m[0]\n    else:\n        m = m[:, ::2]\n    x[(j + 3) % 4] *= x[(j % 4 - 4)] + 1\n    return s, x, m\n', (-3, 2, True, False)),
    'fz_rebind_103_noirram #1899': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    m[(i % 3 - 3), (j % 4 - 4)] *= x[-3] + -2\n    s = s + m[(j % 3 - 3), -2] + 2\n    x = m[1:]\n    x[(i + 2) % 2, -3] *= x[(j + 2) % 2, (i + 2) % 4] + 1\n    if i > 1:\n        x = x[:, ::2]\n        if d:\n            x = np.array(x)\n            s = s + 2\n    m[(i % 3 - 3), 0] = m[-3, (j % 4 - 4)] + -1\n    np.copyto(m, -3, where=m < 0)\n    return s, x, m\n', (-3, 1, False, False)),
    'fz_rebind_131_backends #405': ('def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], i: Integer, j: Integer, c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        a1 = x * 1\n        x[(j + 2) % 6] += 2\n    for k in range(2):\n        if x[k] > 2:\n            break\n        a1 = x + 0\n    np.multiply(m, 2, out=m)\n    return s, x, m, a1\n', (0, 4, True, False)),
}


@pytest.mark.parametrize("name", list(FUZZ_STALE))
@pytest.mark.parametrize("c", [False, True])
def test_fuzz_an_update_keeps_the_links(name, c):
    with pytest.raises(ZinniaException, match=STALE):
        compile_kernel(FUZZ_STALE[name])


@pytest.mark.parametrize("name", list(FUZZ_MATCH))
@pytest.mark.parametrize("c", [False, True])
def test_fuzz_controls_match_numpy(name, c):
    assert_all_backends_match_numpy(FUZZ_MATCH[name], FUZZ_X, FUZZ_M, 1, c)


@pytest.mark.parametrize("name", list(FUZZ_SHAPES))
def test_fuzz_a_multidimensional_merge_of_different_shapes_is_rejected(name):
    with pytest.raises(ZinniaException, match=SHAPES):
        compile_kernel(FUZZ_SHAPES[name])


def test_fuzz_a_branch_only_name_bound_again_in_a_guarded_loop_is_rejected():
    with pytest.raises(ZinniaException, match="bound in one branch"):
        compile_kernel(FUZZ_BRANCH_ONLY)


@pytest.mark.parametrize("name", list(FUZZ_PROGRAMS))
def test_fuzz_program_is_rejected_or_correct(name):
    source, scalars = FUZZ_PROGRAMS[name]
    try:
        compile_kernel(source)
    except ZinniaException:
        return
    assert_all_backends_match_numpy(source, FUZZ_X, FUZZ_M, *scalars)
