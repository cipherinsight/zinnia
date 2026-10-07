"""Regression tests for the third soundness review of array-ssa
(`eval-redo/layer/array-ssa-review/REVIEW-3.md`): every reproducer of
`r3/q3_*.py`, on the mock, halo2-base and Noir, and more cases for each
finding.

Each reproducer has one of these expected outcomes:

- *matches NumPy* on all three backends;
- *no proof*: Python raises (UnboundLocalError), and neither the mock,
  halo2-base nor Noir accepts the input;
- *rejected* at compile time: a read of a stale array, a write to a
  read-only one, or a name bound to arrays of different shapes that is used;
- *unsupported*: a construct Zinnia does not implement; such a program must
  be rejected or compute NumPy's result.

The findings:

- B1: unpacking a multi-dimensional array (`a, b, c = m`) gives views of its
  rows, as iterating it does.
- B2: `np.broadcast_to` is linked to its argument as a view (and read-only).
- B3: `np.diag` of a 2-D array is a read-only view of its diagonal.
- B4: a name first bound in a loop body the loop may leave before the
  binding is bound only where an assignment to it ran; a read asserts that.
- FR1: a name bound in one branch and later in a loop is not rejected.
- FR2: a name bound to arrays of different shapes is rejected only if it is
  used before being rebound.
- FR3: `.shape`, `.ndim` and `.size` of a stale array are available.
"""
import itertools

import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy, nargo_available, NOIR_PARAMS
from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from zinnia.debug.exception.base import ZinniaException

STALE = "is not supported here: .*(linked|a name that refers)"
READ_ONLY = "assignment destination is read-only"
SHAPES = "different shapes|different ranks"

A0 = np.array([3, -1, 4, -6, 5, 9], dtype=np.int64)
A1 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.int64)
A2 = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]], dtype=np.int64)
A3 = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
A4 = np.array([[0, 1], [3, 4], [6, 7], [9, 10]], dtype=np.int64)
A5 = np.array([1.5, -2.0, 3.25, -4.0, 5.0, 0.5], dtype=np.float64)
A6 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.int64)
A7 = np.array([7, 8, 9], dtype=np.int64)
A8 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
A9 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)

R3_LOOPS_MATCH = {
    'loops: loop in else, nested if in loop [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(3):\n            if x[k] > 0:\n                s = s + x[k]\n    return s\n', (A0, False, False,)),
    'loops: loop in else, nested if in loop [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(3):\n            if x[k] > 0:\n                s = s + x[k]\n    return s\n', (A0, False, True,)),
    'loops: loop in else, nested if in loop [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(3):\n            if x[k] > 0:\n                s = s + x[k]\n    return s\n', (A0, True, False,)),
    'loops: loop in else, nested if in loop [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(3):\n            if x[k] > 0:\n                s = s + x[k]\n    return s\n', (A0, True, True,)),
    'loops: loop in else in loop [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    for j in range(2):\n        if c:\n            s = s + 100\n        else:\n            for k in range(2):\n                s = s + 1\n    return s\n', (A0, False, False,)),
    'loops: loop in else in loop [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    for j in range(2):\n        if c:\n            s = s + 100\n        else:\n            for k in range(2):\n                s = s + 1\n    return s\n', (A0, False, True,)),
    'loops: loop in else in loop [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    for j in range(2):\n        if c:\n            s = s + 100\n        else:\n            for k in range(2):\n                s = s + 1\n    return s\n', (A0, True, False,)),
    'loops: loop in else in loop [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    for j in range(2):\n        if c:\n            s = s + 100\n        else:\n            for k in range(2):\n                s = s + 1\n    return s\n', (A0, True, True,)),
    'loops: while in elif chain [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 1\n    elif d:\n        s = 2\n    else:\n        k = 0\n        while k < 3:\n            s = s + k\n            k = k + 1\n    return s\n', (A0, False, False,)),
    'loops: while in elif chain [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 1\n    elif d:\n        s = 2\n    else:\n        k = 0\n        while k < 3:\n            s = s + k\n            k = k + 1\n    return s\n', (A0, False, True,)),
    'loops: while in elif chain [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 1\n    elif d:\n        s = 2\n    else:\n        k = 0\n        while k < 3:\n            s = s + k\n            k = k + 1\n    return s\n', (A0, True, False,)),
    'loops: while in elif chain [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 1\n    elif d:\n        s = 2\n    else:\n        k = 0\n        while k < 3:\n            s = s + k\n            k = k + 1\n    return s\n', (A0, True, True,)),
    'loops: loop writes two names in else [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    t = 1\n    if c:\n        s = 5\n    else:\n        for k in range(3):\n            s = s + 1\n            t = t * 2\n    return s, t\n', (A0, False, False,)),
    'loops: loop writes two names in else [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    t = 1\n    if c:\n        s = 5\n    else:\n        for k in range(3):\n            s = s + 1\n            t = t * 2\n    return s, t\n', (A0, False, True,)),
    'loops: loop writes two names in else [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    t = 1\n    if c:\n        s = 5\n    else:\n        for k in range(3):\n            s = s + 1\n            t = t * 2\n    return s, t\n', (A0, True, False,)),
    'loops: loop writes two names in else [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    t = 1\n    if c:\n        s = 5\n    else:\n        for k in range(3):\n            s = s + 1\n            t = t * 2\n    return s, t\n', (A0, True, True,)),
    'loops: loop in if inside loop with break [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    for j in range(3):\n        if x[j] > 3:\n            break\n        if c:\n            for k in range(2):\n                s = s + 1\n        else:\n            s = s - 1\n    return s\n', (A0, False, False,)),
    'loops: loop in if inside loop with break [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    for j in range(3):\n        if x[j] > 3:\n            break\n        if c:\n            for k in range(2):\n                s = s + 1\n        else:\n            s = s - 1\n    return s\n', (A0, False, True,)),
    'loops: loop in if inside loop with break [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    for j in range(3):\n        if x[j] > 3:\n            break\n        if c:\n            for k in range(2):\n                s = s + 1\n        else:\n            s = s - 1\n    return s\n', (A0, True, False,)),
    'loops: loop in if inside loop with break [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    for j in range(3):\n        if x[j] > 3:\n            break\n        if c:\n            for k in range(2):\n                s = s + 1\n        else:\n            s = s - 1\n    return s\n', (A0, True, True,)),
    'loops: array rebind in loop in else [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = x * 1\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            t = t + 1\n    t[0] = 9\n    return t\n', (A0, False, False,)),
    'loops: array rebind in loop in else [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = x * 1\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            t = t + 1\n    t[0] = 9\n    return t\n', (A0, False, True,)),
    'loops: array rebind in loop in else [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = x * 1\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            t = t + 1\n    t[0] = 9\n    return t\n', (A0, True, False,)),
    'loops: array rebind in loop in else [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = x * 1\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            t = t + 1\n    t[0] = 9\n    return t\n', (A0, True, True,)),
    'loops: array write in loop in else via alias [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = x * 1\n    u = t\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            t[k] = 7\n    return t, u\n', (A0, False, False,)),
    'loops: array write in loop in else via alias [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = x * 1\n    u = t\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            t[k] = 7\n    return t, u\n', (A0, False, True,)),
    'loops: array write in loop in else via alias [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = x * 1\n    u = t\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            t[k] = 7\n    return t, u\n', (A0, True, False,)),
    'loops: array write in loop in else via alias [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = x * 1\n    u = t\n    if c:\n        t = x * 0\n    else:\n        for k in range(2):\n            t[k] = 7\n    return t, u\n', (A0, True, True,)),
    'loops: loop var visible after loop in else [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    k = -1\n    if c:\n        k = 100\n    else:\n        for k in range(3):\n            pass\n    return k\n', (A0, False, False,)),
    'loops: loop var visible after loop in else [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    k = -1\n    if c:\n        k = 100\n    else:\n        for k in range(3):\n            pass\n    return k\n', (A0, False, True,)),
    'loops: loop var visible after loop in else [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    k = -1\n    if c:\n        k = 100\n    else:\n        for k in range(3):\n            pass\n    return k\n', (A0, True, False,)),
    'loops: loop var visible after loop in else [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    k = -1\n    if c:\n        k = 100\n    else:\n        for k in range(3):\n            pass\n    return k\n', (A0, True, True,)),
    'loops: continue in loop in else [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(4):\n            if x[k] < 0:\n                continue\n            s = s + x[k]\n    return s\n', (A0, False, False,)),
    'loops: continue in loop in else [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(4):\n            if x[k] < 0:\n                continue\n            s = s + x[k]\n    return s\n', (A0, False, True,)),
    'loops: continue in loop in else [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(4):\n            if x[k] < 0:\n                continue\n            s = s + x[k]\n    return s\n', (A0, True, False,)),
    'loops: continue in loop in else [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(4):\n            if x[k] < 0:\n                continue\n            s = s + x[k]\n    return s\n', (A0, True, True,)),
    'loops: break in loop in else [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(4):\n            if x[k] < 0:\n                break\n            s = s + x[k]\n    return s\n', (A0, False, False,)),
    'loops: break in loop in else [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(4):\n            if x[k] < 0:\n                break\n            s = s + x[k]\n    return s\n', (A0, False, True,)),
    'loops: break in loop in else [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(4):\n            if x[k] < 0:\n                break\n            s = s + x[k]\n    return s\n', (A0, True, False,)),
    'loops: break in loop in else [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        s = 100\n    else:\n        for k in range(4):\n            if x[k] < 0:\n                break\n            s = s + x[k]\n    return s\n', (A0, True, True,)),
    'loops: 1D shapes differ across break [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = x * 1\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::2]\n    return m\n', (A0, False, False,)),
    'loops: 1D shapes differ across break [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = x * 1\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::2]\n    return m\n', (A0, False, True,)),
    'loops: 1D shapes differ across break [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = x * 1\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::2]\n    return m\n', (A0, True, False,)),
    'loops: 1D shapes differ across break [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = x * 1\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::2]\n    return m\n', (A0, True, True,)),
    'loops: same shapes across break [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = x * 1\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::-1] * 1\n    return m\n', (A0, False, False,)),
    'loops: same shapes across break [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = x * 1\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::-1] * 1\n    return m\n', (A0, False, True,)),
    'loops: same shapes across break [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = x * 1\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::-1] * 1\n    return m\n', (A0, True, False,)),
    'loops: same shapes across break [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = x * 1\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[::-1] * 1\n    return m\n', (A0, True, True,)),
    'loops: temp reused, branch then loop (python ok) [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        tmp = x[0]\n        s = s + tmp\n    for k in range(3):\n        if x[k] > 5:\n            break\n        tmp = x[k]\n        s = s + tmp\n    return s\n', (A0, False, False,)),
    'loops: temp reused, branch then loop (python ok) [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        tmp = x[0]\n        s = s + tmp\n    for k in range(3):\n        if x[k] > 5:\n            break\n        tmp = x[k]\n        s = s + tmp\n    return s\n', (A0, False, True,)),
    'loops: temp reused, branch then loop (python ok) [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        tmp = x[0]\n        s = s + tmp\n    for k in range(3):\n        if x[k] > 5:\n            break\n        tmp = x[k]\n        s = s + tmp\n    return s\n', (A0, True, False,)),
    'loops: temp reused, branch then loop (python ok) [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        tmp = x[0]\n        s = s + tmp\n    for k in range(3):\n        if x[k] > 5:\n            break\n        tmp = x[k]\n        s = s + tmp\n    return s\n', (A0, True, True,)),
    'loops: temp reused, branch then loop with continue [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        tmp = x[0]\n        s = s + tmp\n    for k in range(3):\n        if x[k] < 0:\n            continue\n        tmp = x[k]\n        s = s + tmp\n    return s\n', (A0, False, False,)),
    'loops: temp reused, branch then loop with continue [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        tmp = x[0]\n        s = s + tmp\n    for k in range(3):\n        if x[k] < 0:\n            continue\n        tmp = x[k]\n        s = s + tmp\n    return s\n', (A0, False, True,)),
    'loops: temp reused, branch then loop with continue [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        tmp = x[0]\n        s = s + tmp\n    for k in range(3):\n        if x[k] < 0:\n            continue\n        tmp = x[k]\n        s = s + tmp\n    return s\n', (A0, True, False,)),
    'loops: temp reused, branch then loop with continue [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        tmp = x[0]\n        s = s + tmp\n    for k in range(3):\n        if x[k] < 0:\n            continue\n        tmp = x[k]\n        s = s + tmp\n    return s\n', (A0, True, True,)),
    'loops: branch-only then loop rebind, read after [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    for k in range(3):\n        if x[k] > 3:\n            break\n        y = 2\n    return y\n', (A0, False, False,)),
    'loops: branch-only then loop rebind, read after [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    for k in range(3):\n        if x[k] > 3:\n            break\n        y = 2\n    return y\n', (A0, False, True,)),
    'loops: branch-only then loop rebind, read after [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    for k in range(3):\n        if x[k] > 3:\n            break\n        y = 2\n    return y\n', (A0, True, False,)),
    'loops: branch-only then loop rebind, read after [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    for k in range(3):\n        if x[k] > 3:\n            break\n        y = 2\n    return y\n', (A0, True, True,)),
    'loops: branch-only then plain loop rebind (no break) [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    for k in range(3):\n        y = k\n    return y\n', (A0, False, False,)),
    'loops: branch-only then plain loop rebind (no break) [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    for k in range(3):\n        y = k\n    return y\n', (A0, False, True,)),
    'loops: branch-only then plain loop rebind (no break) [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    for k in range(3):\n        y = k\n    return y\n', (A0, True, False,)),
    'loops: branch-only then plain loop rebind (no break) [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    for k in range(3):\n        y = k\n    return y\n', (A0, True, True,)),
}

R3_LOOPS_RAISES = {
    'loops: first bound after break, read after [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n    return y\n', (A0, False, False,)),
    'loops: first bound after break, read after [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n    return y\n', (A0, False, True,)),
    'loops: first bound after break, read after [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n    return y\n', (A0, True, False,)),
    'loops: first bound after break, read after [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n    return y\n', (A0, True, True,)),
    'loops: first bound after continue, read after [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > -10:\n            continue\n        y = x[k]\n    return y\n', (A0, False, False,)),
    'loops: first bound after continue, read after [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > -10:\n            continue\n        y = x[k]\n    return y\n', (A0, False, True,)),
    'loops: first bound after continue, read after [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > -10:\n            continue\n        y = x[k]\n    return y\n', (A0, True, False,)),
    'loops: first bound after continue, read after [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > -10:\n            continue\n        y = x[k]\n    return y\n', (A0, True, True,)),
    'loops: first bound in while after break [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    k = 0\n    while k < 3:\n        if x[k] > 0:\n            break\n        y = x[k]\n        k = k + 1\n    return y\n', (A0, False, False,)),
    'loops: first bound in while after break [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    k = 0\n    while k < 3:\n        if x[k] > 0:\n            break\n        y = x[k]\n        k = k + 1\n    return y\n', (A0, False, True,)),
    'loops: first bound in while after break [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    k = 0\n    while k < 3:\n        if x[k] > 0:\n            break\n        y = x[k]\n        k = k + 1\n    return y\n', (A0, True, False,)),
    'loops: first bound in while after break [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    k = 0\n    while k < 3:\n        if x[k] > 0:\n            break\n        y = x[k]\n        k = k + 1\n    return y\n', (A0, True, True,)),
}

R3_LOOPS_SHAPES = {
    'loops: scalar vs array across break [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = 0\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[1:3]\n    return m\n', (A0, False, False,)),
    'loops: scalar vs array across break [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = 0\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[1:3]\n    return m\n', (A0, False, True,)),
    'loops: scalar vs array across break [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = 0\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[1:3]\n    return m\n', (A0, True, False,)),
    'loops: scalar vs array across break [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    m = 0\n    for k in range(2):\n        if x[k] > 1:\n            break\n        m = x[1:3]\n    return m\n', (A0, True, True,)),
}

R3_LOOPS_UNSUPPORTED = {
    'loops: branch-only then loop rebind inside if [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        y = 1\n    for k in range(3):\n        if x[k] > 3:\n            break\n        if d:\n            y = 2\n    if c:\n        s = y\n    return s\n', (A0, False, False,)),
    'loops: branch-only then loop rebind inside if [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        y = 1\n    for k in range(3):\n        if x[k] > 3:\n            break\n        if d:\n            y = 2\n    if c:\n        s = y\n    return s\n', (A0, False, True,)),
    'loops: branch-only then loop rebind inside if [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        y = 1\n    for k in range(3):\n        if x[k] > 3:\n            break\n        if d:\n            y = 2\n    if c:\n        s = y\n    return s\n', (A0, True, False,)),
    'loops: branch-only then loop rebind inside if [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    s = 0\n    if c:\n        y = 1\n    for k in range(3):\n        if x[k] > 3:\n            break\n        if d:\n            y = 2\n    if c:\n        s = y\n    return s\n', (A0, True, True,)),
    'loops: first bound under if in loop [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > 100:\n            y = x[k]\n    return y\n', (A0, False, False,)),
    'loops: first bound under if in loop [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > 100:\n            y = x[k]\n    return y\n', (A0, False, True,)),
    'loops: first bound under if in loop [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > 100:\n            y = x[k]\n    return y\n', (A0, True, False,)),
    'loops: first bound under if in loop [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    for k in range(3):\n        if x[k] > 100:\n            y = x[k]\n    return y\n', (A0, True, True,)),
    'loops: first bound in if, read after [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    return y\n', (A0, False, False,)),
    'loops: first bound in if, read after [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    return y\n', (A0, False, True,)),
    'loops: first bound in if, read after [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    return y\n', (A0, True, False,)),
    'loops: first bound in if, read after [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    if c:\n        y = 1\n    return y\n', (A0, True, True,)),
    'loops: loop var of empty range [2]': ('def kernel(x: Integer[Array, "6"], n: Integer):\n    for k in range(0):\n        pass\n    return k\n', (A0, 2,)),
}

R3_MISC_MATCH = {
    'misc: unpack rows write': ('def kernel(m: Integer[Array, "3 4"]):\n    a, b, c = m\n    b[0] = 99\n    return m\n', (A1,)),
    'misc: unpack rows aug': ('def kernel(m: Integer[Array, "3 4"]):\n    a, b, c = m\n    a += 1\n    return m\n', (A1,)),
    'misc: unpack rows base write then read': ('def kernel(m: Integer[Array, "3 4"]):\n    a, b, c = m\n    m[1, 0] = 99\n    return b\n', (A1,)),
    'misc: unpack starred rows write': ('def kernel(m: Integer[Array, "3 4"]):\n    a, *rest = m\n    rest[0][0] = 99\n    return m\n', (A1,)),
    'misc: unpack slice rows write': ('def kernel(m: Integer[Array, "3 4"]):\n    a, b = m[:2]\n    b[1] = 99\n    return m\n', (A1,)),
    'misc: unpack 3D planes write': ('def kernel(t: Integer[Array, "2 3 4"]):\n    p, q = t\n    q[1, 1] = 99\n    return t\n', (A2,)),
    'misc: unpack via tuple(m)': ('def kernel(m: Integer[Array, "3 4"]):\n    a, b, c = tuple(m)\n    c[0] = 99\n    return m\n', (A1,)),
    'misc: unpack 1D elements (scalars)': ('def kernel(x: Integer[Array, "6"]):\n    a, b, c, d, e, f = x\n    a = 99\n    return x\n', (A3,)),
    'misc: for a, b in m.T write': ('def kernel(m: Integer[Array, "4 2"]):\n    for a, b in m:\n        a = 0\n    return m\n', (A4,)),
    'misc: dyn op= then view read': ('def kernel(x: Integer[Array, "6"]):\n    y = x[x > 15]\n    y *= 2\n    v = y[1:]\n    return v\n', (A3,)),
    'misc: list op= then element write': ('def kernel(x: Integer[Array, "6"]):\n    L = [x]\n    L += [x * 2]\n    a = L[0]\n    a[0] = 5\n    return x\n', (A3,)),
    'misc: broadcast_to copy is writable': ('def kernel(x: Integer[Array, "6"]):\n    b = np.broadcast_to(x, (2, 6)).copy()\n    b[0, 0] = 1\n    return b\n', (A3,)),
    'misc: imag of real then base write': ('def kernel(x: Float[Array, "6"]):\n    t = np.imag(x)\n    x[0] = 1.0\n    return t\n', (A5,)),
    'misc: imag of real, np.array copy writable': ('def kernel(x: Float[Array, "6"]):\n    t = np.array(np.imag(x))\n    t[0] = 1.0\n    return t\n', (A5,)),
    'misc: asarray float, write t, read t only': ('def kernel(x: Float[Array, "6"]):\n    t = np.asarray(x, dtype=float)\n    t[0] = 9.0\n    return t\n', (A5,)),
    'misc: asarray float, no writes': ('def kernel(x: Float[Array, "6"]):\n    t = np.asarray(x, dtype=float)\n    return t + x\n', (A5,)),
    'misc: asarray float, write x, read x': ('def kernel(x: Float[Array, "6"]):\n    t = np.asarray(x, dtype=float)\n    x[0] = 9.0\n    return x\n', (A5,)),
    'misc: asarray int to float (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x, dtype=float)\n    t[0] = 9.0\n    return x, t\n', (A3,)),
    'misc: array copy=False float': ('def kernel(x: Float[Array, "6"]):\n    t = np.array(x, copy=False)\n    t[0] = 9.0\n    return x\n', (A5,)),
    'misc: asarray bool same': ('def kernel(x: Integer[Array, "6"]):\n    b = x > 25\n    t = np.asarray(b, dtype=bool)\n    t[0] = True\n    return b\n', (A3,)),
    'misc: 2D merge same shape [True]': ('def kernel(m: Integer[Array, "3 4"], c: Boolean):\n    t = m * 1\n    if c:\n        t = m + 1\n    t[1, 2] = 0\n    return t\n', (A1, True,)),
    'misc: 2D merge transposed same shape [True]': ('def kernel(m: Integer[Array, "4 4"], c: Boolean):\n    t = m * 1\n    if c:\n        t = m.T * 1\n    t[1, 2] = 0\n    return t\n', (A6, True,)),
    'misc: 2D merge different shapes never read [True]': ('def kernel(m: Integer[Array, "3 4"], c: Boolean):\n    t = m * 1\n    if c:\n        t = m[:2] * 1\n    return m\n', (A1, True,)),
}

R3_MISC_READONLY = {
    'misc: broadcast_to view write': ('def kernel(x: Integer[Array, "6"]):\n    b = np.broadcast_to(x, (2, 6))\n    v = b[0]\n    v[0] = 1\n    return x\n', (A3,)),
    'misc: broadcast_to out=': ('def kernel(x: Integer[Array, "6"]):\n    b = np.broadcast_to(x, (2, 6))\n    np.add(b, 1, out=b)\n    return b\n', (A3,)),
    'misc: broadcast_to copyto': ('def kernel(x: Integer[Array, "6"]):\n    b = np.broadcast_to(x, (2, 6))\n    np.copyto(b, 0)\n    return b\n', (A3,)),
}

R3_MISC_STALE = {
    'misc: dyn op= with view': ('def kernel(x: Integer[Array, "6"]):\n    y = x[x > 15]\n    v = y[1:]\n    y += 1\n    return v\n', (A3,)),
    'misc: broadcast_to then base write read': ('def kernel(x: Integer[Array, "6"]):\n    b = np.broadcast_to(x, (2, 6))\n    x[0] = 1\n    return b\n', (A3,)),
    'misc: asarray float32 from float32': ('def kernel(x: Float[Array, "6"]):\n    y = x.astype(np.float32)\n    t = np.asarray(y, dtype=np.float32)\n    t[0] = 9.0\n    return y\n', (A5,)),
    'misc: asarray float64 of float32 (copy)': ('def kernel(x: Float[Array, "6"]):\n    y = x.astype(np.float32)\n    t = np.asarray(y, dtype=np.float64)\n    t[0] = 9.0\n    return y, t\n', (A5,)),
    'misc: astype float copy=False of float': ('def kernel(x: Float[Array, "6"]):\n    t = x.astype(float, copy=False)\n    t[0] = 9.0\n    return x\n', (A5,)),
}

R3_MISC_UNSUPPORTED = {
    'misc: dyn op= alias': ('def kernel(x: Integer[Array, "6"]):\n    y = x[x > 15]\n    z = y\n    y += 1\n    return z\n', (A3,)),
    'misc: dyn op= with flip': ('def kernel(x: Integer[Array, "6"]):\n    y = x[x > 15]\n    t = np.flip(y)\n    y += 1\n    return t\n', (A3,)),
    'misc: dyn op= in branch alias [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    y = x[x > 15]\n    z = y\n    if c:\n        y += 1\n    return z\n', (A3, True,)),
    'misc: list op= alias': ('def kernel(x: Integer[Array, "6"]):\n    L = [1, 2]\n    K = L\n    L += [3]\n    return len(K)\n', (A3,)),
    'misc: list op= of arrays alias': ('def kernel(x: Integer[Array, "6"]):\n    L = [x]\n    K = L\n    L += [x * 2]\n    return K[1]\n', (A3,)),
    'misc: array float dtype copy=False': ('def kernel(x: Float[Array, "6"]):\n    t = np.array(x, dtype=float, copy=False)\n    t[0] = 9.0\n    return x\n', (A5,)),
    'misc: asarray int32 of int64 (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x, dtype=np.int32)\n    t[0] = 9\n    return x, t\n', (A3,)),
}

R3_MORE_MATCH = {
    'more: branch-only temp, plain for (no break) [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    if c:\n        t = x[0]\n        s = s + t\n    for k in range(3):\n        t = x[k]\n        s = s + t\n    return s\n', (A0, False,)),
    'more: branch-only temp, plain for (no break) [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    if c:\n        t = x[0]\n        s = s + t\n    for k in range(3):\n        t = x[k]\n        s = s + t\n    return s\n', (A0, True,)),
    'more: branch-only array temp, plain for [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    if c:\n        t = x * 2\n        x = t\n    for k in range(2):\n        t = x + k\n    return t\n', (A0, False,)),
    'more: branch-only array temp, plain for [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    if c:\n        t = x * 2\n        x = t\n    for k in range(2):\n        t = x + k\n    return t\n', (A0, True,)),
    'more: both-arm binding, loop rebind with break [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    if c:\n        y = 1\n    else:\n        y = 2\n    for k in range(3):\n        if x[k] > 3:\n            break\n        y = y + 10\n    return y\n', (A0, False,)),
    'more: both-arm binding, loop rebind with break [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    if c:\n        y = 1\n    else:\n        y = 2\n    for k in range(3):\n        if x[k] > 3:\n            break\n        y = y + 10\n    return y\n', (A0, True,)),
    'more: branch-only, later while [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    if c:\n        y = 1\n    k = 0\n    while k < 2:\n        y = k\n        k = k + 1\n    return y\n', (A0, False,)),
    'more: branch-only, later while [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    if c:\n        y = 1\n    k = 0\n    while k < 2:\n        y = k\n        k = k + 1\n    return y\n', (A0, True,)),
    'more: branch-only in loop body if, rebind after if (no break) [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    s = 0\n    for k in range(3):\n        if x[k] > 0:\n            t = x[k]\n            s = s + t\n    for k in range(2):\n        t = k\n    return s + t\n', (A0, False,)),
    'more: 2D merge different shapes, overwritten before use [True]': ('def kernel(m: Integer[Array, "3 4"], c: Boolean):\n    t = m * 1\n    if c:\n        t = m[:2] * 1\n    t = m + 1\n    return t\n', (A1, True,)),
    'more: 2D merge different shapes, overwritten before use [False]': ('def kernel(m: Integer[Array, "3 4"], c: Boolean):\n    t = m * 1\n    if c:\n        t = m[:2] * 1\n    t = m + 1\n    return t\n', (A1, False,)),
    'more: 1D merge different lengths, read len [True]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    t = x * 1\n    if c:\n        t = x[:3] * 1\n    return len(t)\n', (A0, True,)),
    'more: 1D merge different lengths, read len [False]': ('def kernel(x: Integer[Array, "6"], c: Boolean):\n    t = x * 1\n    if c:\n        t = x[:3] * 1\n    return len(t)\n', (A0, False,)),
    'more: stale only read shape': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.flip(m)\n    m[0, 0] = 1\n    r, k = t.shape\n    return r * k\n', (A1,)),
    'more: stale then ndim': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.flip(m)\n    m[0, 0] = 1\n    return t.ndim\n', (A1,)),
}

R3_STALE_MATCH = {
    'stale: write c, read else [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    else:\n        return t\n    return x\n', (A3, False, False,)),
    'stale: write c, read else [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    else:\n        return t\n    return x\n', (A3, False, True,)),
    'stale: write c, read else [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    else:\n        return t\n    return x\n', (A3, True, False,)),
    'stale: write c, read else [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    else:\n        return t\n    return x\n', (A3, True, True,)),
    'stale: for row in m write': ('def kernel(m: Integer[Array, "3 4"]):\n    for row in m:\n        row[0] = 99\n    return m\n', (A1,)),
    'stale: for row in m aug': ('def kernel(m: Integer[Array, "3 4"]):\n    for row in m:\n        row += 1\n    return m\n', (A1,)),
    'stale: enumerate rows write': ('def kernel(m: Integer[Array, "3 4"]):\n    for i, row in enumerate(m):\n        row[1] = i\n    return m\n', (A1,)),
    'stale: list(m) row write': ('def kernel(m: Integer[Array, "3 4"]):\n    L = list(m)\n    r = L[1]\n    r[0] = 99\n    return m\n', (A1,)),
    'stale: zip rows write': ('def kernel(m: Integer[Array, "3 4"], y: Integer[Array, "3"]):\n    for row, v in zip(m, y):\n        row[0] = v\n    return m\n', (A1, A7,)),
    'stale: m write then old row read': ('def kernel(m: Integer[Array, "3 4"]):\n    rows = [r for r in m]\n    m[1, 0] = 99\n    return rows[1]\n', (A1,)),
    'stale: chip rebinding param': ('def f(a):\n    a = a * 2\n    a[0] = 0\n    return a\n\ndef kernel(x: Integer[Array, "6"]):\n    y = f(x)\n    return x, y\n', (A3,)),
    'stale: chip writes view of param': ('def f(a):\n    v = a[1:]\n    v[0] = 0\n\ndef kernel(x: Integer[Array, "6"]):\n    f(x)\n    return x\n', (A3,)),
    'stale: chip param alias of two args': ('def f(a, b):\n    a[0] = 5\n    return b[0]\n\ndef kernel(x: Integer[Array, "6"]):\n    return f(x, x)\n', (A3,)),
    'stale: chip param view and base': ('def f(a, b):\n    a[0] = 5\n    return b[1]\n\ndef kernel(x: Integer[Array, "6"]):\n    return f(x[1:], x)\n', (A3,)),
    'stale: list elem rebinding no effect': ('def kernel(x: Integer[Array, "6"]):\n    L = [x]\n    L[0] = x * 2\n    L[0][0] = 5\n    return x\n', (A3,)),
    'stale: list elem alias write': ('def kernel(x: Integer[Array, "6"]):\n    L = [x, x * 2]\n    a = L[0]\n    a[1] = 5\n    return x\n', (A3,)),
    'stale: tuple elem alias write': ('def kernel(x: Integer[Array, "6"]):\n    T = (x, 1)\n    a = T[0]\n    a[1] = 5\n    return x\n', (A3,)),
    'stale: list of views write via loop': ('def kernel(x: Integer[Array, "6"]):\n    L = [x[0:2], x[2:4]]\n    for v in L:\n        v[0] = 9\n    return x\n', (A3,)),
    'stale: stale shape read': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t.shape[0]\n', (A3,)),
    'stale: stale rebound then read': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    t = np.flip(x)\n    return t\n', (A3,)),
}

R3_STALE_STALE = {
    'stale: write c, read later if not c [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if not c:\n        s = t[5]\n    return s\n', (A3, False, False,)),
    'stale: write c, read later if not c [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if not c:\n        s = t[5]\n    return s\n', (A3, False, True,)),
    'stale: write c, read later if not c [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if not c:\n        s = t[5]\n    return s\n', (A3, True, False,)),
    'stale: write c, read later if not c [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if not c:\n        s = t[5]\n    return s\n', (A3, True, True,)),
    'stale: write c, read later if c [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if c:\n        s = t[5]\n    return s\n', (A3, False, False,)),
    'stale: write c, read later if c [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if c:\n        s = t[5]\n    return s\n', (A3, False, True,)),
    'stale: write c, read later if c [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if c:\n        s = t[5]\n    return s\n', (A3, True, False,)),
    'stale: write c, read later if c [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if c:\n        s = t[5]\n    return s\n', (A3, True, True,)),
    'stale: write c, read c==False [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if c == False:\n        s = t[5]\n    return s\n', (A3, False, False,)),
    'stale: write c, read c==False [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if c == False:\n        s = t[5]\n    return s\n', (A3, False, True,)),
    'stale: write c, read c==False [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if c == False:\n        s = t[5]\n    return s\n', (A3, True, False,)),
    'stale: write c, read c==False [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        x[0] = 99\n    if c == False:\n        s = t[5]\n    return s\n', (A3, True, True,)),
    'stale: write c and d, read not d [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        if d:\n            x[0] = 99\n    if not d:\n        s = t[5]\n    return s\n', (A3, False, False,)),
    'stale: write c and d, read not d [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        if d:\n            x[0] = 99\n    if not d:\n        s = t[5]\n    return s\n', (A3, False, True,)),
    'stale: write c and d, read not d [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        if d:\n            x[0] = 99\n    if not d:\n        s = t[5]\n    return s\n', (A3, True, False,)),
    'stale: write c and d, read not d [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c:\n        if d:\n            x[0] = 99\n    if not d:\n        s = t[5]\n    return s\n', (A3, True, True,)),
    'stale: write c or d, read not c [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c or d:\n        x[0] = 99\n    if not c:\n        s = t[5]\n    return s\n', (A3, False, False,)),
    'stale: write c or d, read not c [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c or d:\n        x[0] = 99\n    if not c:\n        s = t[5]\n    return s\n', (A3, False, True,)),
    'stale: write c or d, read not c [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c or d:\n        x[0] = 99\n    if not c:\n        s = t[5]\n    return s\n', (A3, True, False,)),
    'stale: write c or d, read not c [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    s = 0\n    if c or d:\n        x[0] = 99\n    if not c:\n        s = t[5]\n    return s\n', (A3, True, True,)),
    'stale: write c, read under not c via merge [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    u = x * 0\n    if not c:\n        u = t\n    return u\n', (A3, False, False,)),
    'stale: write c, read under not c via merge [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    u = x * 0\n    if not c:\n        u = t\n    return u\n', (A3, False, True,)),
    'stale: write c, read under not c via merge [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    u = x * 0\n    if not c:\n        u = t\n    return u\n', (A3, True, False,)),
    'stale: write c, read under not c via merge [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    if c:\n        x[0] = 99\n    u = x * 0\n    if not c:\n        u = t\n    return u\n', (A3, True, True,)),
    'stale: merge stale and fresh, read where fresh [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    s = 0\n    if not c:\n        s = u[5]\n    return s\n', (A3, False, False,)),
    'stale: merge stale and fresh, read where fresh [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    s = 0\n    if not c:\n        s = u[5]\n    return s\n', (A3, False, True,)),
    'stale: merge stale and fresh, read where fresh [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    s = 0\n    if not c:\n        s = u[5]\n    return s\n', (A3, True, False,)),
    'stale: merge stale and fresh, read where fresh [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    s = 0\n    if not c:\n        s = u[5]\n    return s\n', (A3, True, True,)),
    'stale: merge stale and fresh, read everywhere [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    return u\n', (A3, False, False,)),
    'stale: merge stale and fresh, read everywhere [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    return u\n', (A3, False, True,)),
    'stale: merge stale and fresh, read everywhere [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    return u\n', (A3, True, False,)),
    'stale: merge stale and fresh, read everywhere [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    return u\n', (A3, True, True,)),
    'stale: double merge carry [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    v = x * 2\n    if d:\n        v = u\n    s = 0\n    if not d:\n        s = v[5]\n    return s\n', (A3, False, False,)),
    'stale: double merge carry [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    v = x * 2\n    if d:\n        v = u\n    s = 0\n    if not d:\n        s = v[5]\n    return s\n', (A3, False, True,)),
    'stale: double merge carry [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    v = x * 2\n    if d:\n        v = u\n    s = 0\n    if not d:\n        s = v[5]\n    return s\n', (A3, True, False,)),
    'stale: double merge carry [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    v = x * 2\n    if d:\n        v = u\n    s = 0\n    if not d:\n        s = v[5]\n    return s\n', (A3, True, True,)),
    'stale: double merge carry read under d [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    v = x * 2\n    if d:\n        v = u\n    s = 0\n    if d:\n        s = v[5]\n    return s\n', (A3, False, False,)),
    'stale: double merge carry read under d [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    v = x * 2\n    if d:\n        v = u\n    s = 0\n    if d:\n        s = v[5]\n    return s\n', (A3, False, True,)),
    'stale: double merge carry read under d [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    v = x * 2\n    if d:\n        v = u\n    s = 0\n    if d:\n        s = v[5]\n    return s\n', (A3, True, False,)),
    'stale: double merge carry read under d [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    t = np.flip(x)\n    x[0] = 99\n    u = x * 1\n    if c:\n        u = t\n    v = x * 2\n    if d:\n        v = u\n    s = 0\n    if d:\n        s = v[5]\n    return s\n', (A3, True, True,)),
    'stale: alias write under d, read x [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    if d:\n        z[0] = 99\n    return x\n', (A3, False, False,)),
    'stale: alias write under d, read x [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    if d:\n        z[0] = 99\n    return x\n', (A3, False, True,)),
    'stale: alias write under d, read x [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    if d:\n        z[0] = 99\n    return x\n', (A3, True, False,)),
    'stale: alias write under d, read x [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    if d:\n        z[0] = 99\n    return x\n', (A3, True, True,)),
    'stale: alias write, read w under c [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    s = 0\n    if c:\n        s = w[0]\n    return s\n', (A3, False, False,)),
    'stale: alias write, read w under c [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    s = 0\n    if c:\n        s = w[0]\n    return s\n', (A3, False, True,)),
    'stale: alias write, read w under c [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    s = 0\n    if c:\n        s = w[0]\n    return s\n', (A3, True, False,)),
    'stale: alias write, read w under c [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    s = 0\n    if c:\n        s = w[0]\n    return s\n', (A3, True, True,)),
    'stale: alias write, read x under not c [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    s = 0\n    if not c:\n        s = x[0]\n    return s\n', (A3, False, False,)),
    'stale: alias write, read x under not c [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    s = 0\n    if not c:\n        s = x[0]\n    return s\n', (A3, False, True,)),
    'stale: alias write, read x under not c [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    s = 0\n    if not c:\n        s = x[0]\n    return s\n', (A3, True, False,)),
    'stale: alias write, read x under not c [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    z[0] = 99\n    s = 0\n    if not c:\n        s = x[0]\n    return s\n', (A3, True, True,)),
    'stale: alias write under c, read x under c [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n        z[0] = 99\n    return x\n', (A3, False, False,)),
    'stale: alias write under c, read x under c [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n        z[0] = 99\n    return x\n', (A3, False, True,)),
    'stale: alias write under c, read x under c [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n        z[0] = 99\n    return x\n', (A3, True, False,)),
    'stale: alias write under c, read x under c [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n        z[0] = 99\n    return x\n', (A3, True, True,)),
    'stale: write x then alias merge then read z [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    x[1] = 99\n    return z\n', (A3, False, False,)),
    'stale: write x then alias merge then read z [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    x[1] = 99\n    return z\n', (A3, False, True,)),
    'stale: write x then alias merge then read z [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    x[1] = 99\n    return z\n', (A3, True, False,)),
    'stale: write x then alias merge then read z [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    x[1] = 99\n    return z\n', (A3, True, True,)),
    'stale: write w, read z where c [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    w[1] = 99\n    s = 0\n    if c:\n        s = z[1]\n    return s\n', (A3, False, False,)),
    'stale: write w, read z where c [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    w[1] = 99\n    s = 0\n    if c:\n        s = z[1]\n    return s\n', (A3, False, True,)),
    'stale: write w, read z where c [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    w[1] = 99\n    s = 0\n    if c:\n        s = z[1]\n    return s\n', (A3, True, False,)),
    'stale: write w, read z where c [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    w[1] = 99\n    s = 0\n    if c:\n        s = z[1]\n    return s\n', (A3, True, True,)),
    'stale: write w, read z where not c [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    w[1] = 99\n    s = 0\n    if not c:\n        s = z[1]\n    return s\n', (A3, False, False,)),
    'stale: write w, read z where not c [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    w[1] = 99\n    s = 0\n    if not c:\n        s = z[1]\n    return s\n', (A3, False, True,)),
    'stale: write w, read z where not c [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    w[1] = 99\n    s = 0\n    if not c:\n        s = z[1]\n    return s\n', (A3, True, False,)),
    'stale: write w, read z where not c [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    w = x * 1\n    z = w\n    if c:\n        z = x\n    w[1] = 99\n    s = 0\n    if not c:\n        s = z[1]\n    return s\n', (A3, True, True,)),
    'stale: swap in loop with break [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    b = x * 2\n    for k in range(3):\n        if x[k] > 25:\n            break\n        a, b = b, a\n    a[0] = 7\n    return a, b\n', (A3, False, False,)),
    'stale: swap in loop with break [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    b = x * 2\n    for k in range(3):\n        if x[k] > 25:\n            break\n        a, b = b, a\n    a[0] = 7\n    return a, b\n', (A3, False, True,)),
    'stale: swap in loop with break [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    b = x * 2\n    for k in range(3):\n        if x[k] > 25:\n            break\n        a, b = b, a\n    a[0] = 7\n    return a, b\n', (A3, True, False,)),
    'stale: swap in loop with break [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    b = x * 2\n    for k in range(3):\n        if x[k] > 25:\n            break\n        a, b = b, a\n    a[0] = 7\n    return a, b\n', (A3, True, True,)),
    'stale: swap in loop then write b read a [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    b = x * 2\n    for k in range(3):\n        if x[k] > 25:\n            break\n        a, b = b, a\n    b[0] = 7\n    return a\n', (A3, False, False,)),
    'stale: swap in loop then write b read a [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    b = x * 2\n    for k in range(3):\n        if x[k] > 25:\n            break\n        a, b = b, a\n    b[0] = 7\n    return a\n', (A3, False, True,)),
    'stale: swap in loop then write b read a [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    b = x * 2\n    for k in range(3):\n        if x[k] > 25:\n            break\n        a, b = b, a\n    b[0] = 7\n    return a\n', (A3, True, False,)),
    'stale: swap in loop then write b read a [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    b = x * 2\n    for k in range(3):\n        if x[k] > 25:\n            break\n        a, b = b, a\n    b[0] = 7\n    return a\n', (A3, True, True,)),
    'stale: alias in loop then write [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    z = a\n    for k in range(3):\n        if x[k] > 15:\n            break\n        z = x\n    z[0] = 7\n    return a, x\n', (A3, False, False,)),
    'stale: alias in loop then write [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    z = a\n    for k in range(3):\n        if x[k] > 15:\n            break\n        z = x\n    z[0] = 7\n    return a, x\n', (A3, False, True,)),
    'stale: alias in loop then write [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    z = a\n    for k in range(3):\n        if x[k] > 15:\n            break\n        z = x\n    z[0] = 7\n    return a, x\n', (A3, True, False,)),
    'stale: alias in loop then write [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    z = a\n    for k in range(3):\n        if x[k] > 15:\n            break\n        z = x\n    z[0] = 7\n    return a, x\n', (A3, True, True,)),
    'stale: continue alias [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    z = a\n    for k in range(3):\n        if x[k] < 25:\n            continue\n        z = x\n    z[0] = 7\n    return a, x\n', (A3, False, False,)),
    'stale: continue alias [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    z = a\n    for k in range(3):\n        if x[k] < 25:\n            continue\n        z = x\n    z[0] = 7\n    return a, x\n', (A3, False, True,)),
    'stale: continue alias [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    z = a\n    for k in range(3):\n        if x[k] < 25:\n            continue\n        z = x\n    z[0] = 7\n    return a, x\n', (A3, True, False,)),
    'stale: continue alias [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    a = x * 1\n    z = a\n    for k in range(3):\n        if x[k] < 25:\n            continue\n        z = x\n    z[0] = 7\n    return a, x\n', (A3, True, True,)),
    'stale: iterate reversed rows write': ('def kernel(m: Integer[Array, "3 4"]):\n    for row in m[::-1]:\n        row[0] = 99\n    return m\n', (A1,)),
    'stale: chip returns flip, caller reads after base write': ('def f(a):\n    return np.flip(a)\n\ndef kernel(x: Integer[Array, "6"]):\n    t = f(x)\n    x[0] = 1\n    return t\n', (A3,)),
    'stale: chip writes base under cond, caller reads flip': ('def f(a, c):\n    if c:\n        a[0] = 1\n\ndef kernel(x: Integer[Array, "6"], c: Boolean):\n    t = np.flip(x)\n    f(x, c)\n    return t\n', (A3,)),
    'stale: chip param flip and base': ('def f(a, b):\n    a[0] = 5\n    return b[5]\n\ndef kernel(x: Integer[Array, "6"]):\n    return f(np.flip(x), x)\n', (A3,)),
    'stale: merged list of arrays write [False, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    y = x * 2\n    if c:\n        L = [x, y]\n    else:\n        L = [y, x]\n    a = L[0]\n    a[0] = 5\n    return x, y\n', (A3, False, False,)),
    'stale: merged list of arrays write [False, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    y = x * 2\n    if c:\n        L = [x, y]\n    else:\n        L = [y, x]\n    a = L[0]\n    a[0] = 5\n    return x, y\n', (A3, False, True,)),
    'stale: merged list of arrays write [True, False]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    y = x * 2\n    if c:\n        L = [x, y]\n    else:\n        L = [y, x]\n    a = L[0]\n    a[0] = 5\n    return x, y\n', (A3, True, False,)),
    'stale: merged list of arrays write [True, True]': ('def kernel(x: Integer[Array, "6"], c: Boolean, d: Boolean):\n    y = x * 2\n    if c:\n        L = [x, y]\n    else:\n        L = [y, x]\n    a = L[0]\n    a[0] = 5\n    return x, y\n', (A3, True, True,)),
    'stale: list of flips write base': ('def kernel(x: Integer[Array, "6"]):\n    L = [np.flip(x), x * 1]\n    x[0] = 9\n    return L[1]\n', (A3,)),
    'stale: stale len read': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return len(t)\n', (A3,)),
    'stale: stale overwritten fully then read': ('def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    t[:] = 0\n    return t\n', (A3,)),
}

R3_STALE_UNSUPPORTED = {
    'stale: list append then write': ('def kernel(x: Integer[Array, "6"]):\n    L = []\n    L.append(x)\n    a = L[0]\n    a[2] = 5\n    return x\n', (A3,)),
}

R3_VIEWS_MATCH = {
    'views: np.real of real base write': ('def kernel(x: Float[Array, "6"]):\n    r = np.real(x)\n    x[0] = 9.0\n    return r\n', (A9,)),
    'views: x.T base write (1d)': ('def kernel(x: Integer[Array, "6"]):\n    t = x.T\n    x[0] = 9\n    return t\n', (A3,)),
    'views: np.transpose base write': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.transpose(m)\n    m[0, 1] = 99\n    return t\n', (A1,)),
    'views: np.squeeze base write': ('def kernel(m: Integer[Array, "3 4"]):\n    t = np.squeeze(m[1:2])\n    m[1, 1] = 99\n    return t\n', (A1,)),
    'views: expand_dims base write': ('def kernel(x: Integer[Array, "6"]):\n    t = np.expand_dims(x, 0)\n    x[1] = 99\n    return t\n', (A3,)),
    'views: np.reshape fn base write': ('def kernel(x: Integer[Array, "6"]):\n    t = np.reshape(x, (2, 3))\n    x[1] = 99\n    return t\n', (A3,)),
    'views: ravel of column (copy in numpy) then write t': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m[:, 1].ravel()\n    t[0] = 99\n    return m, t\n', (A1,)),
    'views: flatten of view (copy)': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m[:, 1].flatten()\n    t[0] = 99\n    return m, t\n', (A1,)),
    'views: np.roll (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.roll(x, 1)\n    t[0] = 99\n    return x, t\n', (A3,)),
    'views: np.sort (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.sort(x)\n    t[0] = 99\n    return x, t\n', (A3,)),
    'views: np.where (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.where(x > 25, x, 0)\n    t[0] = 99\n    return x, t\n', (A3,)),
    'views: np.concatenate (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = np.concatenate([x, x])\n    t[0] = 99\n    return x, t\n', (A3,)),
    'views: x[[0,1]] fancy (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = x[[0, 1]]\n    t[0] = 99\n    return x, t\n', (A3,)),
    'views: x[x>25] mask (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = x[x > 25]\n    t[0] = 99\n    return x, t\n', (A3,)),
    'views: x + 0 (copy)': ('def kernel(x: Integer[Array, "6"]):\n    t = x + 0\n    t[0] = 99\n    return x, t\n', (A3,)),
    'views: np.copy then base write': ('def kernel(x: Integer[Array, "6"]):\n    t = np.copy(x)\n    x[0] = 99\n    return t\n', (A3,)),
    'views: list(x) then base write': ('def kernel(x: Integer[Array, "6"]):\n    t = list(x)\n    x[0] = 99\n    return t[0]\n', (A3,)),
    'views: x.tolist then base write': ('def kernel(x: Integer[Array, "6"]):\n    t = x.tolist()\n    x[0] = 99\n    return t[0]\n', (A3,)),
    'views: np.split pieces base write': ('def kernel(x: Integer[Array, "6"]):\n    a, b = np.split(x, 2)\n    x[4] = 99\n    return b\n', (A3,)),
    'views: np.hsplit 2D base write': ('def kernel(m: Integer[Array, "3 4"]):\n    a, b = np.hsplit(m, 2)\n    m[0, 3] = 99\n    return b\n', (A1,)),
    'views: np.vsplit 2D write piece': ('def kernel(m: Integer[Array, "3 4"]):\n    a, b, c = np.vsplit(m, 3)\n    b[0, 0] = 99\n    return m\n', (A1,)),
    'views: np.array_split base write': ('def kernel(x: Integer[Array, "6"]):\n    p = np.array_split(x, 3)\n    x[5] = 99\n    return p[2]\n', (A3,)),
}

R3_VIEWS_READONLY = {
    'views: np.diag write (numpy read-only)': ('def kernel(m: Integer[Array, "3 3"]):\n    d = np.diag(m)\n    d[0] = 1\n    return d\n', (A8,)),
}

R3_VIEWS_STALE = {
    'views: broadcast_to 2D base write': ('def kernel(m: Integer[Array, "3 3"]):\n    b = np.broadcast_to(m[0], (3, 3))\n    m[0, 1] = 7\n    return b\n', (A8,)),
    'views: np.diag 2D base write read': ('def kernel(m: Integer[Array, "3 3"]):\n    d = np.diag(m)\n    m[1, 1] = 99\n    return d\n', (A8,)),
    'views: atleast_3d base write': ('def kernel(x: Integer[Array, "6"]):\n    t = np.atleast_3d(x)\n    x[1] = 99\n    return t\n', (A3,)),
    'views: x.reshape(-1) of 2D col (copy) base write': ('def kernel(m: Integer[Array, "3 4"]):\n    t = m[:, 1].reshape(-1)\n    m[1, 1] = 99\n    return t\n', (A1,)),
}

R3_VIEWS_UNSUPPORTED = {
    'views: np.diagonal base write': ('def kernel(m: Integer[Array, "3 3"]):\n    d = np.diagonal(m)\n    m[1, 1] = 99\n    return d\n', (A8,)),
    'views: m.diagonal() base write': ('def kernel(m: Integer[Array, "3 3"]):\n    d = m.diagonal()\n    m[1, 1] = 99\n    return d\n', (A8,)),
}

# ── More cases per finding ───────────────────────────────────────────────

X6 = np.array([3, -1, 4, -6, 5, 9], dtype=np.int64)
X6_NEGATIVE_FIRST = np.array([-3, -1, 4, -6, 5, 9], dtype=np.int64)
X6_POSITIVE = np.array([3, 1, 4, 6, 5, 9], dtype=np.int64)
M34 = np.arange(12, dtype=np.int64).reshape(3, 4)
H = 'def kernel(x: Integer[Array, "6"], m: Integer[Array, "3 4"], c: Boolean):\n'

UNPACKING = {
    "rows, write a row": H + '    a, b, c2 = m\n    b[1] = 99\n    return m\n',
    "rows, write the base": H + '    a, b, c2 = m\n    m[2, 3] = 99\n    return a, b, c2\n',
    "rows, op= on a row": H + '    a, b, c2 = m\n    c2 += 5\n    return m\n',
    "rows under a condition": H + '    a, b, c2 = m\n    if c:\n        b[0] = 99\n    return m\n',
    "starred rest, write through it": H + '    a, *rest = m\n    rest[1][2] = 99\n    return m\n',
    "transposed rows": H + '    a, b, c2, d = m.T\n    b[0] = 99\n    return m\n',
    "rows of a list-held array": H + '    L = [m]\n    L[0][1][2] = 99\n    return m\n',
}

VIEW_FUNCTIONS = {
    "broadcast_to, base write": H + '    b = np.broadcast_to(x, (2, 6))\n    x[0] = 1\n    return b\n',
    "broadcast_to of a row, base write": H + '    b = np.broadcast_to(m[1], (2, 4))\n    m[1, 1] = 1\n    return b\n',
    "diag of a matrix, base write": H + '    d = np.diag(m[:, :3])\n    m[1, 1] = 99\n    return d\n',
}
VIEW_FUNCTIONS_UNWRITTEN = {
    "broadcast_to, read only": H + '    b = np.broadcast_to(x, (2, 6))\n    return b + 1\n',
    "diag of a matrix, read only": H + '    d = np.diag(m[:, :3])\n    return d * 2, m\n',
    "diag of a vector is a new array": H + '    d = np.diag(x[:3])\n    d[0, 1] = 7\n    return d, x\n',
}
READ_ONLY_RESULTS = {
    "diag of a matrix": H + '    d = np.diag(m[:, :3])\n    d[0] = 1\n    return d\n',
    "broadcast_to": H + '    b = np.broadcast_to(x, (2, 6))\n    b[1, 1] = 1\n    return b\n',
}

# B4: (source, inputs on which Python returns, inputs on which it raises).
LB = 'def kernel(x: Integer[Array, "6"], c: Boolean):\n'
UNBOUND = {
    "break before the first binding": LB + '    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n    return y\n',
    "continue before the first binding": LB + '    for k in range(3):\n        if x[k] > 0:\n            continue\n        y = x[k]\n    return y\n',
    "while with break": LB + '    k = 0\n    while k < 3:\n        if x[k] > 0:\n            break\n        y = x[k]\n        k = k + 1\n    return y\n',
    "read inside the loop and after": LB + '    s = 0\n    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n        s = s + y\n    return s, y\n',
    "nested loops": LB + '    for k in range(2):\n        if x[k] > 0:\n            break\n        for j in range(2):\n            y = x[j + 2]\n    return y\n',
    "inside a branch": LB + '    s = 0\n    if c:\n        for k in range(3):\n            if x[k] > 0:\n                break\n            y = x[k]\n        s = y\n    return s\n',
    "method call on it": LB + '    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k:k + 2]\n    return y.sum()\n',
    "subscript write to it": LB + '    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k:k + 2] * 1\n    y[0] = 5\n    return y\n',
}
UNBOUND_BOUND_INPUTS = [(X6_NEGATIVE_FIRST, True), (X6_NEGATIVE_FIRST, False)]
UNBOUND_UNBOUND_INPUTS = [(X6_POSITIVE, True)]
# A name bound before the loop is bound everywhere; no assertion.
BOUND = {
    "bound before the loop": LB + '    y = 0\n    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n    return y\n',
    "rebound after the loop": LB + '    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n    y = 7\n    return y\n',
    "plain loop": LB + '    for k in range(3):\n        y = x[k]\n    return y\n',
}

# FR1: a name bound in one branch, then in a loop.
FR1 = {
    "plain for": LB + '    s = 0\n    if c:\n        t = x[0]\n        s = s + t\n    for k in range(3):\n        t = x[k]\n        s = s + t\n    return s, t\n',
    "while": LB + '    s = 0\n    if c:\n        t = x[0]\n    k = 0\n    while k < 3:\n        t = x[k]\n        s = s + t\n        k = k + 1\n    return s, t\n',
    "array temp": LB + '    s = 0\n    if c:\n        t = x * 2\n        s = s + t[0]\n    for k in range(2):\n        t = x * k\n        s = s + t[1]\n    return s, t\n',
}
# With a `break` before the loop's assignment, Python keeps the branch's
# value on the paths that skip it; Zinnia drops that binding at the branch
# join, so those paths have no proof (sound, not complete).
FR1_BREAK = LB + '    if c:\n        t = x[0]\n    for k in range(3):\n        if x[k] > 0:\n            break\n        t = x[k]\n    return t\n'

# FR2: arrays of different shapes under one name, used or not.
FR2_UNUSED = {
    "rebound before use": H + '    t = m * 1\n    if c:\n        t = m[:2] * 1\n    t = m + 1\n    return t\n',
    "array or number, rebound": H + '    t = m * 1\n    if c:\n        t = 0\n    t = x\n    return t\n',
    "loop that may break, rebound": H + '    t = m * 1\n    for k in range(2):\n        if x[k] > 3:\n            break\n        t = np.transpose(t)\n    t = x * 2\n    return t\n',
}
FR2_USED = {
    "read": H + '    t = m * 1\n    if c:\n        t = m[:2] * 1\n    return t.sum()\n',
    "shape": H + '    t = m * 1\n    if c:\n        t = m[:2] * 1\n    return t.shape[0]\n',
    "write": H + '    t = m * 1\n    if c:\n        t = m[:2] * 1\n    t[0, 0] = 5\n    return x\n',
    "method": H + '    t = m * 1\n    if c:\n        t = m[:2] * 1\n    u = t.T\n    return x\n',
    "array or number": H + '    t = m * 1\n    if c:\n        t = 0\n    return t\n',
}

# FR3: metadata of a stale array.
FR3 = {
    "shape": H + '    t = np.flip(m)\n    m[0, 0] = 1\n    r, k = t.shape\n    return r * k, m\n',
    "ndim": H + '    t = np.flip(m)\n    m[0, 0] = 1\n    return t.ndim, m\n',
    "size": H + '    t = np.flip(m)\n    m[0, 0] = 1\n    return t.size, m\n',
    "shape of an expression": H + '    t = np.flip(m)\n    return np.flip(m).shape[1] + t[0, 0]\n',
}


def _cases(*tables):
    return [pytest.param(source, args, id=name) for table in tables for name, (source, args) in table.items()]


def assert_no_backend_proves(source, *args):
    """No backend proves `source` on `args` (Python raises on them): the
    program is rejected, or the mock rejects the witness, halo2-base has no
    verifying proof and Noir's execution fails."""
    try:
        program = compile_kernel(source)
    except ZinniaException:
        return
    satisfied, _, _ = run_mock_layout(program, *args)
    assert not satisfied, "the mock accepts an input on which Python raises"
    try:
        proof = program.prove(*args, backend="halo2-base")
        assert not program.verify(proof), "halo2-base proves an input on which Python raises"
    except AssertionError:
        raise
    except Exception:  # noqa: BLE001  (no proof)
        pass
    if nargo_available():
        import json
        from zinnia._zinnia_core import execute_noir_circuit
        from zinnia.exec.input_parser import build_circuit_inputs

        witness = build_circuit_inputs(program.program_inputs, args)
        with pytest.raises(Exception):
            execute_noir_circuit(program._compiled_ir, json.dumps(witness), {}, json.dumps(NOIR_PARAMS))


# `np.sort` of a static array is planned as a routed sort network, which
# only halo2-base emits (`np.sort` on Noir is outside this review).
HALO2_ONLY = {"views: np.sort (copy)"}


@pytest.mark.parametrize("source, args", _cases(R3_LOOPS_MATCH, R3_MISC_MATCH, R3_MORE_MATCH, R3_STALE_MATCH, R3_VIEWS_MATCH))
def test_the_reproducer_matches_numpy(source, args, request):
    noir = request.node.callspec.id not in HALO2_ONLY
    assert_all_backends_match_numpy(source, *args, noir=noir)


@pytest.mark.parametrize("source, args", _cases(R3_LOOPS_RAISES))
def test_no_backend_proves_an_input_on_which_python_raises(source, args):
    with pytest.raises(Exception):
        numpy_outputs(source, *args)
    assert_no_backend_proves(source, *args)


@pytest.mark.parametrize("source, args", _cases(R3_MISC_STALE, R3_STALE_STALE, R3_VIEWS_STALE))
def test_reading_a_stale_array_is_rejected(source, args):
    with pytest.raises(ZinniaException, match=STALE):
        compile_kernel(source)


@pytest.mark.parametrize("source, args", _cases(R3_MISC_READONLY, R3_VIEWS_READONLY))
def test_a_write_to_a_read_only_result_is_rejected(source, args):
    with pytest.raises(ZinniaException, match=READ_ONLY):
        compile_kernel(source)


@pytest.mark.parametrize("source, args", _cases(R3_LOOPS_SHAPES))
def test_using_a_name_bound_to_different_shapes_is_rejected(source, args):
    with pytest.raises(ZinniaException, match=SHAPES):
        compile_kernel(source)


@pytest.mark.parametrize("source, args", _cases(R3_LOOPS_UNSUPPORTED, R3_MISC_UNSUPPORTED, R3_STALE_UNSUPPORTED, R3_VIEWS_UNSUPPORTED))
def test_an_unsupported_reproducer_is_rejected_or_correct(source, args):
    try:
        compile_kernel(source)
    except ZinniaException:
        return
    assert_all_backends_match_numpy(source, *args)


@pytest.mark.parametrize("name", list(UNPACKING))
@pytest.mark.parametrize("c", [False, True])
def test_unpacking_gives_views_of_the_rows(name, c):
    assert_all_backends_match_numpy(UNPACKING[name], X6, M34, c)


@pytest.mark.parametrize("name", list(VIEW_FUNCTIONS))
def test_a_view_function_result_is_linked_to_its_argument(name):
    with pytest.raises(ZinniaException, match=STALE):
        compile_kernel(VIEW_FUNCTIONS[name])


@pytest.mark.parametrize("name", list(VIEW_FUNCTIONS_UNWRITTEN))
def test_a_view_function_result_nobody_writes_is_fine(name):
    assert_all_backends_match_numpy(VIEW_FUNCTIONS_UNWRITTEN[name], X6, M34, True)


@pytest.mark.parametrize("name", list(READ_ONLY_RESULTS))
def test_a_view_function_result_is_read_only(name):
    with pytest.raises(ZinniaException, match=READ_ONLY):
        compile_kernel(READ_ONLY_RESULTS[name])


@pytest.mark.parametrize("name", list(UNBOUND))
@pytest.mark.parametrize("inputs", range(len(UNBOUND_BOUND_INPUTS)))
def test_a_name_bound_on_the_path_reads_its_value(name, inputs):
    assert_all_backends_match_numpy(UNBOUND[name], *UNBOUND_BOUND_INPUTS[inputs])


@pytest.mark.parametrize("name", list(UNBOUND))
def test_a_name_unbound_on_the_path_has_no_proof(name):
    for args in UNBOUND_UNBOUND_INPUTS:
        with pytest.raises(Exception):
            numpy_outputs(UNBOUND[name], *args)
        assert_no_backend_proves(UNBOUND[name], *args)


@pytest.mark.parametrize("name", list(BOUND))
@pytest.mark.parametrize("x", [X6, X6_NEGATIVE_FIRST])
def test_a_name_bound_on_every_path_reads_its_value(name, x):
    assert_all_backends_match_numpy(BOUND[name], x, True)


def _asserts(source):
    return sum(s["ir_instance"]["__class__"] == "AssertIR" for s in compile_kernel(source).get_ir_stmts())


def test_the_bound_assertion_is_emitted_only_where_needed():
    # One assertion for the read after the loop; none for a name bound
    # before the loop, and none for a read in the iteration that bound it.
    unbound = UNBOUND["break before the first binding"]
    bound = BOUND["bound before the loop"]
    assert _asserts(unbound) == _asserts(bound) + 1
    inside = LB + '    s = 0\n    for k in range(3):\n        if x[k] > 0:\n            break\n        y = x[k]\n        s = s + y\n    return s\n'
    inside_bound = inside.replace("    s = 0\n", "    s = 0\n    y = 0\n")
    assert _asserts(inside) == _asserts(inside_bound)


@pytest.mark.parametrize("name", list(FR1))
@pytest.mark.parametrize("c", [False, True])
def test_a_name_bound_in_a_branch_then_in_a_loop_compiles(name, c):
    assert_all_backends_match_numpy(FR1[name], X6, c)


def test_a_branch_binding_skipped_by_a_loop_break_has_no_proof_where_python_keeps_it():
    # c and x[0] > 0: Python returns the branch's x[0]; Zinnia has no proof.
    assert_no_backend_proves(FR1_BREAK, X6, True)
    # The loop binds `t` in its first iteration: the value is the loop's.
    assert_all_backends_match_numpy(FR1_BREAK, X6_NEGATIVE_FIRST, True)
    assert_all_backends_match_numpy(FR1_BREAK, X6_NEGATIVE_FIRST, False)


@pytest.mark.parametrize("name", list(FR2_UNUSED))
@pytest.mark.parametrize("c", [False, True])
def test_an_unused_merge_of_different_shapes_compiles(name, c):
    assert_all_backends_match_numpy(FR2_UNUSED[name], X6, M34, c)


@pytest.mark.parametrize("name", list(FR2_USED))
def test_a_used_merge_of_different_shapes_is_rejected(name):
    with pytest.raises(ZinniaException, match=SHAPES):
        compile_kernel(FR2_USED[name])


@pytest.mark.parametrize("name", list(FR3))
def test_the_metadata_of_a_stale_array_is_available(name):
    assert_all_backends_match_numpy(FR3[name], X6, M34, True)


def test_reading_the_cells_of_a_stale_array_is_still_rejected():
    source = H + '    t = np.flip(m)\n    m[0, 0] = 1\n    return t.shape[0] + t[0, 0]\n'
    with pytest.raises(ZinniaException, match=STALE):
        compile_kernel(source)
