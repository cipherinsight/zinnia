"""Candidate tasks, independent oracles, and ordinary implementation alternatives.

The alternatives are a coverage inventory, not participant starter solutions.
Shapes in PROFILES are public project parameters; array contents are witnesses.
"""

import ast
from dataclasses import dataclass
from textwrap import dedent

import numpy as np


@dataclass(frozen=True)
class Task:
    name: str
    title: str
    profiles: tuple[dict, ...]
    variants: dict[str, str]


def body(source):
    return dedent(source).strip()


TASKS = [
    Task("calibrate", "Calibrate and clip readings", ({"N": 6}, {"N": 1}, {"N": 32}), {
        "clip": "return np.clip(2 * x + 5, 0, 255)",
        "clip_keywords": "return np.clip(2 * x + 5, a_min=0, a_max=255)",
        "clip_method": "return (2 * x + 5).clip(0, 255)",
        "clip_named_method": "y = 2 * x + 5\nreturn y.clip(min=0, max=255)",
        "clip_out": "y = 2 * x + 5\nnp.clip(y, 0, 255, out=y)\nreturn y",
        "mask_assignment": "y = 2 * x + 5\ny[y < 0] = 0\ny[y > 255] = 255\nreturn y",
        "minmax": "return np.minimum(np.maximum(2 * x + 5, 0), 255)",
        "where": "y = 2 * x + 5\nreturn np.where(y < 0, 0, np.where(y > 255, 255, y))",
        "comprehension": "return np.array([min(255, max(0, 2 * v + 5)) for v in x])",
        "append": body("""
            result = []
            for v in x:
                result.append(min(255, max(0, 2 * v + 5)))
            return np.asarray(result)
        """),
        "indexed_branches": body("""
            result = np.zeros_like(x)
            for i in range(len(x)):
                v = 2 * x[i] + 5
                if v < 0:
                    v = 0
                elif v > 255:
                    v = 255
                result[i] = v
            return result
        """),
        "while": body("""
            result = np.zeros_like(x)
            i = 0
            while i < len(x):
                result[i] = min(255, max(0, 2 * x[i] + 5))
                i += 1
            return result
        """),
        # Audit regressions (eval-redo X5, X6, X1): positional ufunc `out`,
        # np.putmask with repeated values, and a helper that writes into its
        # array argument under a private branch.
        "positional_out": "y = 2 * x + 5\nnp.maximum(y, 0, y)\nnp.minimum(y, 255, y)\nreturn y",
        "putmask": "y = 2 * x + 5\nnp.putmask(y, y < 0, 0)\nnp.putmask(y, y > 255, [255, 255])\nreturn y",
        "helper_mutation": body("""
            def set_at(out, i, value):
                out[i] = value
            result = 2 * x + 5
            for i in range(len(x)):
                if result[i] < 0:
                    set_at(result, i, 0)
                elif result[i] > 255:
                    set_at(result, i, 255)
            return result
        """),
        # Gate H7: an early return under a private condition, and a local
        # helper with its own returns (the checker rewrites only the
        # variant's returns and keeps them as returns).
        "early_return": body("""
            y = 2 * x + 5
            if np.all(y >= 0) and np.all(y <= 255):
                return y
            return np.clip(y, 0, 255)
        """),
        "helper_returns": body("""
            def clamp(v):
                if v < 0:
                    return 0
                if v > 255:
                    return 255
                return v
            return np.array([clamp(2 * v + 5) for v in x])
        """),
    }),
    Task("background", "Subtract channel background", ({"T": 3, "C": 2}, {"T": 1, "C": 1}, {"T": 16, "C": 4}), {
        "function": "return x - np.min(x, axis=0)",
        "method": "return x - x.min(axis=0)",
        "keepdims": "return x - np.amin(x, axis=0, keepdims=True)",
        "transpose": "return (x.T - np.min(x.T, axis=1).reshape(-1, 1)).T",
        "column_slices": body("""
            out = np.zeros_like(x)
            for c in range(x.shape[1]):
                out[:, c] = x[:, c] - min(x[:, c])
            return out
        """),
        "comprehension": "return np.array([[x[t, c] - min(x[:, c]) for c in range(x.shape[1])] for t in range(x.shape[0])])",
        "nested_loops": body("""
            out = np.zeros_like(x)
            for c in range(x.shape[1]):
                low = x[0, c]
                for t in range(1, x.shape[0]):
                    if x[t, c] < low:
                        low = x[t, c]
                for t in range(x.shape[0]):
                    out[t, c] = x[t, c] - low
            return out
        """),
    }),
    Task("threshold_energy", "Energy above a per-channel noise floor", ({"N": 3, "C": 4}, {"N": 1, "C": 1}, {"N": 16, "C": 4}), {
        "mask": "return np.sum((x > floor) * x * x, axis=1)",
        "where": "return np.where(x > floor, x * x, 0).sum(axis=1)",
        "filtered_rows": body("""
            out = []
            for row in x:
                active = row[row > floor]
                out.append(np.sum(active * active))
            return np.array(out)
        """),
        "generator": "return np.array([sum(v * v for v, f in zip(row, floor) if v > f) for row in x])",
        "conditional_expression": "return np.array([sum(row[c] * row[c] if row[c] > floor[c] else 0 for c in range(len(floor))) for row in x])",
        "filtered_comprehension": "return np.array([sum([row[c] ** 2 for c in range(len(floor)) if row[c] > floor[c]]) for row in x])",
        # Audit regression (eval-redo X9): a filtered element that calls a
        # lifted helper runs only for kept candidates.
        "filtered_helper": "square = lambda v: v * v\nreturn np.array([sum([square(row[c]) for c in range(len(floor)) if row[c] > floor[c]]) for row in x])",
        "mask_assignment": "squares = x * x\nsquares[x <= floor] = 0\nreturn squares.sum(axis=1)",
        "filtered_iteration": body("""
            result = []
            for row in x:
                total = 0
                active = row[row > floor]
                for v in active:
                    total += v * v
                result.append(total)
            return np.asarray(result)
        """),
        "nested_loops": body("""
            out = np.zeros(x.shape[0], dtype=int)
            for n in range(x.shape[0]):
                for c in range(x.shape[1]):
                    if x[n, c] > floor[c]:
                        out[n] += x[n, c] * x[n, c]
            return out
        """),
        "append_filtered": body("""
            out = []
            for row in x:
                active = []
                for c in range(len(floor)):
                    if row[c] > floor[c]:
                        active.append(row[c] * row[c])
                out.append(sum(active))
            return np.array(out)
        """),
    }),
    Task("distances", "Pairwise squared distances", ({"N": 3, "M": 2, "D": 3}, {"N": 1, "M": 1, "D": 1}, {"N": 8, "M": 6, "D": 3}), {
        "broadcast": "delta = x[:, None, :] - y[None, :, :]\nreturn np.sum(delta * delta, axis=2)",
        "power": "return ((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2).sum(axis=-1)",
        "dot_identity": "return np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2 * np.dot(x, y.T)",
        "matmul_identity": "return (x * x).sum(axis=1, keepdims=True) + (y * y).sum(axis=1) - 2 * (x @ y.T)",
        "einsum": "delta = x[:, None, :] - y[None, :, :]\nreturn np.einsum('nmd,nmd->nm', delta, delta)",
        "einsum_ellipsis": "delta = x[:, None, :] - y[None, :, :]\nreturn np.einsum('...d,...d->...', delta, delta, optimize=True)",
        "tensordot_identity": "return np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2 * np.tensordot(x, y, axes=([1], [1]))",
        "comprehension": "return np.array([[sum((a - b) ** 2 for a, b in zip(row, other)) for other in y] for row in x])",
        "nested_loops": body("""
            out = np.zeros((x.shape[0], y.shape[0]), dtype=int)
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    total = 0
                    for d in range(x.shape[1]):
                        diff = x[i, d] - y[j, d]
                        total += diff * diff
                    out[i, j] = total
            return out
        """),
    }),
    Task("pool", "Sum non-overlapping image blocks", ({"B": 2, "H": 4, "W": 6, "C": 2}, {"B": 1, "H": 2, "W": 2, "C": 1}, {"B": 2, "H": 8, "W": 8, "C": 3}), {
        "reshape_tuple_axes": "b, h, w, c = x.shape\nreturn x.reshape(b, h // 2, 2, w // 2, 2, c).sum(axis=(2, 4))",
        "reshape_sequential_axes": "b, h, w, c = x.shape\ny = np.reshape(x, (b, h // 2, 2, w // 2, 2, c))\nreturn np.sum(np.sum(y, axis=4), axis=2)",
        "strided_slices": "return x[:, 0::2, 0::2, :] + x[:, 0::2, 1::2, :] + x[:, 1::2, 0::2, :] + x[:, 1::2, 1::2, :]",
        "stack": "return np.stack([x[:, 0::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 0::2, :], x[:, 1::2, 1::2, :]], axis=0).sum(axis=0)",
        "einsum": "b, h, w, c = x.shape\ny = x.reshape(b, h // 2, 2, w // 2, 2, c)\nreturn np.einsum('biujvc->bijc', y)",
        "block_slices": body("""
            b, h, w, c = x.shape
            out = np.zeros((b, h // 2, w // 2, c), dtype=int)
            for batch in range(b):
                for i in range(h // 2):
                    for j in range(w // 2):
                        out[batch, i, j, :] = x[batch, 2*i:2*i+2, 2*j:2*j+2, :].sum(axis=(0, 1))
            return out
        """),
        "nested_loops": body("""
            b, h, w, c = x.shape
            out = np.zeros((b, h // 2, w // 2, c), dtype=int)
            for batch in range(b):
                for i in range(h // 2):
                    for j in range(w // 2):
                        for channel in range(c):
                            total = 0
                            for u in range(2):
                                for v in range(2):
                                    total += x[batch, 2*i+u, 2*j+v, channel]
                            out[batch, i, j, channel] = total
            return out
        """),
    }),
    Task("smooth", "Repeated spatial smoothing", ({"H": 4, "W": 5, "R": 3}, {"H": 3, "W": 3, "R": 1}, {"H": 2, "W": 3, "R": 2}, {"H": 8, "W": 8, "R": 3}), {
        "slices": body("""
            a = x.copy()
            for step in range({R}):
                out = a.copy()
                out[1:-1, 1:-1] = (4*a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]) // 8
                a = out
            return a
        """),
        "np_copy": body("""
            a = np.copy(x)
            for step in range({R}):
                out = np.copy(a)
                for i in range(1, a.shape[0] - 1):
                    for j in range(1, a.shape[1] - 1):
                        out[i, j] = (4*a[i, j] + a[i-1, j] + a[i+1, j] + a[i, j-1] + a[i, j+1]) // 8
                a = out
            return a
        """),
        "nested_loops": body("""
            a = x.copy()
            for step in range({R}):
                out = a.copy()
                for i in range(1, a.shape[0] - 1):
                    for j in range(1, a.shape[1] - 1):
                        out[i, j] = (4*a[i, j] + a[i-1, j] + a[i+1, j] + a[i, j-1] + a[i, j+1]) // 8
                a = out
            return a
        """),
        "while": body("""
            a = x.copy()
            step = 0
            while step < {R}:
                out = a.copy()
                for i in range(1, a.shape[0] - 1):
                    for j in range(1, a.shape[1] - 1):
                        out[i, j] = (4*a[i, j] + a[i-1, j] + a[i+1, j] + a[i, j-1] + a[i, j+1]) // 8
                a = out
                step += 1
            return a
        """),
        "comprehension": body("""
            a = x.copy()
            for step in range({R}):
                a = np.array([[a[i, j] if i == 0 or j == 0 or i == a.shape[0]-1 or j == a.shape[1]-1 else (4*a[i,j]+a[i-1,j]+a[i+1,j]+a[i,j-1]+a[i,j+1]) // 8 for j in range(a.shape[1])] for i in range(a.shape[0])])
            return a
        """),
        "buffer_assignment": body("""
            a = x.copy()
            out = x.copy()
            for step in range({R}):
                out[1:-1, 1:-1] = (4*a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]) // 8
                a[:, :] = out
            return a
        """),
    }),
    # Audit regressions (eval-redo X1, X4): writes whose target row is a
    # private value, through a chained subscript and through a helper called
    # under a private branch.
    Task("scatter_add", "Add positive amounts into keyed buckets", ({"N": 5}, {"N": 1}, {"N": 16}), {
        "single_step": body("""
            out = np.zeros(4, dtype=int)
            for i in range(x.shape[0]):
                if x[i, 1] > 0:
                    out[x[i, 0] % 4] += x[i, 1]
            return out
        """),
        "chained_private_row": body("""
            grid = np.zeros((4, 1), dtype=int)
            for i in range(x.shape[0]):
                if x[i, 1] > 0:
                    grid[x[i, 0] % 4][0] += x[i, 1]
            return grid[:, 0]
        """),
        "helper_under_branch": body("""
            def add_to(out, key, amount):
                out[key] += amount
            out = np.zeros(4, dtype=int)
            for i in range(x.shape[0]):
                if x[i, 1] > 0:
                    add_to(out, x[i, 0] % 4, x[i, 1])
            return out
        """),
        "masked_sums": body("""
            keys = x[:, 0] % 4
            amounts = np.where(x[:, 1] > 0, x[:, 1], 0)
            return np.array([np.sum(np.where(keys == k, amounts, 0)) for k in range(4)])
        """),
    }),
]


def chained_indices(text):
    """Also exercise ordinary a[i][j] reads and writes in the loop solutions."""
    class Chain(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if not isinstance(node.slice, ast.Tuple):
                return node
            value = node.value
            for i, index in enumerate(node.slice.elts):
                context = node.ctx if i == len(node.slice.elts) - 1 else ast.Load()
                value = ast.Subscript(value=value, slice=index, ctx=context)
            return ast.copy_location(value, node)

    return ast.unparse(ast.fix_missing_locations(Chain().visit(ast.parse(text))))


for task in TASKS:
    if task.name in ("background", "distances", "pool", "smooth"):
        task.variants["chained_indexing"] = chained_indices(task.variants["nested_loops"])


def signatures(task, p):
    if task.name == "calibrate":
        shapes = {"x": (p["N"],)}
        out = (p["N"],)
    elif task.name == "background":
        shapes = {"x": (p["T"], p["C"])}
        out = shapes["x"]
    elif task.name == "threshold_energy":
        shapes = {"x": (p["N"], p["C"]), "floor": (p["C"],)}
        out = (p["N"],)
    elif task.name == "distances":
        shapes = {"x": (p["N"], p["D"]), "y": (p["M"], p["D"])}
        out = (p["N"], p["M"])
    elif task.name == "pool":
        shapes = {"x": tuple(p[k] for k in ("B", "H", "W", "C"))}
        out = (p["B"], p["H"] // 2, p["W"] // 2, p["C"])
    elif task.name == "scatter_add":
        shapes = {"x": (p["N"], 2)}
        out = (4,)
    else:
        shapes = {"x": (p["H"], p["W"])}
        out = shapes["x"]
    return shapes, out


def source(task, variant, profile):
    shapes, _ = signatures(task, profile)
    args = ", ".join(f'{name}: Integer[Array, "{" ".join(map(str, shape))}"]' for name, shape in shapes.items())
    lines = task.variants[variant].format(**profile).splitlines()
    return f"def kernel({args}):\n" + "\n".join("    " + line for line in lines) + "\n"


def fixtures(task, profile):
    shapes, _ = signatures(task, profile)
    rng = np.random.default_rng(20260921)
    cases = []
    for mode in range(7):
        values = []
        for name, shape in shapes.items():
            if mode == 0:
                a = np.zeros(shape, dtype=np.int64)
            elif mode == 1:
                a = np.full(shape, 255, dtype=np.int64)
            elif mode == 2:
                a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) * 7
            elif mode == 3:
                a = np.full(shape, -10, dtype=np.int64)
            elif mode == 5:
                a = np.resize(np.array([-512, -511, 4095, 4096], dtype=np.int64), shape)
            elif mode == 6:
                a = rng.integers(-512, 4097, size=shape, dtype=np.int64)
            else:
                a = rng.integers(-20, 256, size=shape, dtype=np.int64)
            if task.name == "smooth":
                a = np.abs(a)
            if name == "floor":
                if mode in (0, 1, 3, 4):
                    a = np.full(shape, {0: 0, 1: 254, 3: -11, 4: 300}[mode], dtype=np.int64)
                elif mode == 2:
                    a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) * 11
                elif mode == 5:
                    a = np.resize(np.array([-511, -512, 4096, 4095], dtype=np.int64), shape)
            if task.name == "calibrate" and mode == 2:
                a = np.resize(np.array([-3, -2, 0, 124, 125, 126], dtype=np.int64), shape)
            values.append(a)
        cases.append(values)
    return cases


def oracle(task, profile, values):
    """Scalar reference algorithms independent of the vectorized alternatives."""
    x = values[0]
    _, shape = signatures(task, profile)
    result = np.zeros(shape, dtype=np.int64)
    if task.name == "calibrate":
        for i in range(len(x)):
            result[i] = min(255, max(0, 2 * int(x[i]) + 5))
    elif task.name == "background":
        for c in range(x.shape[1]):
            low = min(int(x[t, c]) for t in range(x.shape[0]))
            for t in range(x.shape[0]):
                result[t, c] = int(x[t, c]) - low
    elif task.name == "threshold_energy":
        floor = values[1]
        for i in range(x.shape[0]):
            result[i] = sum(int(x[i, c]) ** 2 for c in range(x.shape[1]) if x[i, c] > floor[c])
    elif task.name == "distances":
        y = values[1]
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                result[i, j] = sum((int(a) - int(b)) ** 2 for a, b in zip(x[i], y[j]))
    elif task.name == "pool":
        for b, i, j, c in np.ndindex(shape):
            result[b, i, j, c] = sum(int(x[b, 2*i+u, 2*j+v, c]) for u in range(2) for v in range(2))
    elif task.name == "scatter_add":
        for key, amount in x.tolist():
            if amount > 0:
                result[key % 4] += amount
    else:
        result = x.copy()
        for _ in range(profile["R"]):
            old = result.copy()
            for i in range(1, x.shape[0]-1):
                for j in range(1, x.shape[1]-1):
                    result[i, j] = (4*int(old[i,j])+int(old[i-1,j])+int(old[i+1,j])+int(old[i,j-1])+int(old[i,j+1])) // 8
    return result
