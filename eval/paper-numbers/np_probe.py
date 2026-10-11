"""Count the NumPy functions, ndarray methods and math functions the compiler accepts.

A name counts as supported if at least one small program calling it compiles (halo2-base
target, no proving). Candidates: every public callable of numpy, numpy.linalg, math and every
public ndarray method; templates call it on 1-D/2-D float and integer arrays and scalars with
0-2 extra arguments. Run with the npb-hw worktree's venv (compiler 3ebe0558).
"""
import json, math, sys, multiprocessing as mp
import numpy as np

TEMPLATES = [
    "r = {f}(a)", "r = {f}(a, b)", "r = {f}(a, 1)", "r = {f}(a, 0)", "r = {f}(m)", "r = {f}(m, m)",
    "r = {f}(m, axis=0)", "r = {f}(a, 1, 2)", "r = {f}(x)", "r = {f}(x, y)", "r = {f}(i)",
    "r = {f}(i, j)", "r = {f}(4)", "r = {f}((2, 3))", "r = {f}(a > 0)", "r = {f}(a > 0, a, b)",
    "r = {f}(ia)", "r = {f}(ia, ib)", "r = {f}(a, [1, 2])", "r = {f}(m, 2)", "r = {f}(a, a > 0)",
    "r = {f}(a > 0, b > 0)", "r = {f}(sq)", "r = {f}(sq, a[:4])", "r = {f}('ij,jk->ik', m, mt)",
    "r = {f}(ia, 3)", "r = {f}(a, ia)", "r = {f}(a, 2, 0)", "r = {f}(m, 1)", "r = {f}(a, 50)",
    "r = {f}(a, (2, 3))", "r = {f}([a, b])", "r = {f}((a, b))", "r = {f}(a, ib)", "r = {f}(4, 4)",
]
METHOD_TEMPLATES = ["r = a.{f}()", "r = m.{f}()", "r = m.{f}(0)", "r = a.{f}(1)", "r = ia.{f}()",
                    "r = m.{f}(3, 2)", "r = a.{f}(int)", "r = a.{f}(a)", "r = m.{f}(1, 0)",
                    "r = a.{f}(ia)", "r = a.{f}(0, 1)", "r = a.{f}"]
HEADER = (
          "def probe(a: Float[Array, '6'], b: Float[Array, '6'], m: Float[Array, '2 3'],"
          " mt: Float[Array, '3 2'], sq: Float[Array, '4 4'], ia: Integer[Array, '6'],"
          " ib: Integer[Array, '6'], x: Float, y: Float, i: Integer, j: Integer):\n")


def compiles(body):
    from zinnia import ZKCircuit
    src = HEADER + "    " + body + "\n    return r\n"
    try:
        ZKCircuit("probe", src).compile()
        return True
    except Exception:
        return False


def probe(item):
    kind, name, qual = item
    for t in (METHOD_TEMPLATES if kind == "method" else TEMPLATES):
        if compiles(t.format(f=qual)):
            return item, t.format(f=qual)
    return item, None


if __name__ == "__main__":
    items = []
    items += [("np", n, f"np.{n}") for n in dir(np) if not n.startswith("_")
              and callable(getattr(np, n)) and not isinstance(getattr(np, n), type)]
    items += [("linalg", n, f"np.linalg.{n}") for n in dir(np.linalg) if not n.startswith("_")
              and callable(getattr(np.linalg, n)) and not isinstance(getattr(np.linalg, n), type)]
    items += [("math", n, f"math.{n}") for n in dir(math) if not n.startswith("_")
              and callable(getattr(math, n))]
    items += [("method", n, n) for n in dir(np.ndarray) if not n.startswith("_")]
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        results = pool.map(probe, items, chunksize=1)
    ok = {}
    for (kind, name, _), t in results:
        if t:
            ok.setdefault(kind, []).append((name, t))
    json.dump(ok, open(sys.argv[2] if len(sys.argv) > 2 else "np_probe.json", "w"), indent=1)
    for kind, names in ok.items():
        print(kind, len(names), sorted(n for n, _ in names))
