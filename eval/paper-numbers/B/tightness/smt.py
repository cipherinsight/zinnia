"""Method (c): z3 models of the size expressions where a construction does not obviously
settle the maximum: the arrays below capacity (periodic_dist) and the programs whose arrays
cannot all be full on one input (0490, tax, periodic_dist).

For each question there are two queries: an Optimize (OMT) maximisation, whose model is
turned into a program input and measured with the NumPy reference (common.Program.sizes),
and a plain satisfiability check that "size >= optimum + 1" is unsatisfiable, which proves
the optimum (if the OMT query times out, the bound checked is the construction's size). The models are over the reals (z3 `Real`), with every value inside the Float
domain |v| < 2^31; integers are z3 `Int` in the int64 range. Writes smt.json.
"""
import json
import time

import z3

import common

LIMIT = common.FLOAT_LIMIT
TIMEOUT_MS = 900_000


def ind(b):
    return z3.If(b, 1, 0)


def fval(m, v):
    r = m.eval(v, model_completion=True)
    return float(r.as_fraction()) if z3.is_rational_value(r) else float(r.as_decimal(20).rstrip('?'))


def maximise(constraints, objective, claimed=None):
    """OMT maximum and its model, then a satisfiability check of objective >= bound + 1,
    where bound is the OMT optimum (or, if the OMT query times out, the claimed maximum
    from the construction); "unsat" proves the bound."""
    t = time.time()
    o = z3.Optimize()
    o.set(timeout=TIMEOUT_MS)
    o.add(*constraints)
    h = o.maximize(objective)
    res = o.check()
    best = h.value().as_long() if res == z3.sat else None
    model = o.model() if res == z3.sat else None
    bound = best if best is not None else claimed
    proof = None
    if bound is not None:
        s = z3.Solver()
        s.set(timeout=TIMEOUT_MS)
        s.add(*constraints, objective >= bound + 1)
        proof = str(s.check())
    return dict(omt=str(res), optimum=best, checked_bound=bound, above_bound=proof,
                seconds=round(time.time() - t, 2)), model


# ---- pythran_periodic_dist: one axis (the three axes are independent and identical) ----
def periodic(n, l_constraint, which, claimed):
    x = [z3.Real(f'x{i}') for i in range(n)]
    L = z3.Int('L')
    h = z3.ToReal(L) / 2
    cons = [z3.And(-LIMIT <= xi, xi <= LIMIT) for xi in x] + [L >= common.INT_MIN, L <= common.INT_MAX]
    # Symmetry breaking: relabelling the particles permutes the entries of dx, so every
    # count below is unchanged and the points may be taken in increasing order.
    cons += [x[i] <= x[i + 1] for i in range(n - 1)]
    if l_constraint == 'L>0':
        cons.append(L > 0)
    m1 = m2 = 0
    for i in range(n):
        for j in range(n):
            d = x[j] - x[i]                       # dx[i, j] = xtemp[i, j] - xtemp.T[i, j] = x[j] - x[i]
            first = d > h                         # dx[dx > L / 2]
            d2 = z3.If(first, d - z3.ToReal(L), d)  # dx[dx > L / 2] = dx[dx > L / 2] - L
            second = d2 < -h                      # dx[dx < -L / 2], on the updated dx
            m1 = m1 + ind(first)
            m2 = m2 + ind(second)
    obj = {'first': m1, 'second': m2, 'both': m1 + m2}[which]
    r, model = maximise(cons, obj, claimed)
    if model is not None:
        r['input'] = dict(x=[fval(model, xi) for xi in x], L=model.eval(L, model_completion=True).as_long())
    return r


def periodic_check(r):
    """Measure a one-axis model on the real program (all three axes set to it)."""
    P = common.Program('pythran__pythran_periodic_dist__dist')
    v = dict(x=r['input']['x'], y=r['input']['x'], z=r['input']['x'], L=r['input']['L'],
             periodicX=True, periodicY=True, periodicZ=True)
    s = P.sizes(v)
    return None if s is None else {k: s[k] for k in s if k.startswith(('19:', '20:'))}


# ---- case0490: one row (the five rows are independent and identical) ----
def row0490(which):
    n = 50
    t = [z3.Real(f't{i}') for i in range(n)]
    t1, t2 = z3.Int('t1'), z3.Int('t2')
    cons = [z3.And(-LIMIT <= v, v <= LIMIT) for v in t] + [z3.And(common.INT_MIN <= k, k <= common.INT_MAX) for k in (t1, t2)]
    cons += [t[i] <= t[i + 1] for i in range(n - 1)]   # symmetry breaking: counts ignore order
    lt = sum(ind(v < z3.ToReal(t1)) for v in t)                   # np.where(temp < t1)
    ge = sum(ind(v >= z3.ToReal(t2)) for v in t)                  # np.where(temp >= t2)
    mid = sum(ind(z3.And(v >= z3.ToReal(t1), v < z3.ToReal(t2))) for v in t)  # mask result and its + 5
    obj = {'lt': lt, 'ge': ge, 'mid': mid, 'row_total': lt + ge + 2 * mid}[which]
    r, model = maximise(cons, obj)
    if model is not None:
        r['input'] = dict(temp=[fval(model, v) for v in t], t1=model.eval(t1, model_completion=True).as_long(),
                          t2=model.eval(t2, model_completion=True).as_long())
    return r


def row0490_check(r):
    P = common.Program('ds1000_numpy__case0490__kernel')
    v = dict(arr=[r['input']['temp']] * 5, n1=[r['input']['t1']] * 5, n2=[r['input']['t2']] * 5)
    s = P.sizes(v)
    return None if s is None else dict(total=sum(s.values()), row0={k: s[k] for k in s if '#0' in k})


# ---- pythran_tax ----
def tax():
    n = 10
    d = [z3.Real(f'd{i}') for i in range(n)]
    cons = [z3.And(-LIMIT <= v, v <= LIMIT) for v in d] + [sum(d) != 0]   # np.sum(d) is a divisor
    cons += [d[i] <= d[i + 1] for i in range(n - 1)]   # symmetry breaking: counts ignore order
    c1 = sum(ind(v > 256303) for v in d)
    c2 = sum(ind(z3.And(v > 54057, v <= 256303)) for v in d)
    c3 = sum(ind(z3.And(v > 13769, v <= 54057)) for v in d)
    c4 = sum(ind(z3.And(v > 8820, v <= 13769)) for v in d)
    # Arrays per bracket in the inventory: 3 (seg1), 3 (seg2), 6 (seg3 chain), 5 (seg4 chain).
    r, model = maximise(cons, 3 * c1 + 3 * c2 + 6 * c3 + 5 * c4)
    if model is not None:
        r['input'] = dict(d=[fval(model, v) for v in d])
    return r


def tax_check(r):
    P = common.Program('pythran__pythran_tax__tax')
    s = P.sizes(dict(d=r['input']['d']))
    return None if s is None else dict(total=sum(s.values()))


def main():
    out = {}
    for which in ('lt', 'ge', 'mid', 'row_total'):
        key = f'case0490 one row max {which}'
        r = row0490(which)
        r['numpy'] = row0490_check(r) if 'input' in r else None
        out[key] = r
        print(key, {k: r[k] for k in ('omt', 'optimum', 'above_bound', 'seconds')}, flush=True)
    (common.HERE / 'smt.json').write_text(json.dumps(out, indent=1))
    r = tax()
    r['numpy'] = tax_check(r) if 'input' in r else None
    out['tax max total'] = r
    print('tax', {k: r[k] for k in ('omt', 'optimum', 'above_bound', 'seconds')}, r['numpy'], flush=True)
    # Claimed maxima from the constructions and the argument in REPORT.md: the first mask
    # n^2 for L < 0 and n(n-1)/2 for L > 0, the second mask n(n-1)/2, both together n^2
    # for L < 0 and n(n-1) for L > 0. Small n first (fast), then the program's n = 16.
    for n in (4, 6, 8, 16):
        half = n * (n - 1) // 2
        for lc in ('any L', 'L>0'):
            for which, claimed in (('first', n * n if lc == 'any L' else half), ('second', half),
                                   ('both', n * n if lc == 'any L' else 2 * half)):
                key = f'periodic_dist n={n} {lc} max {which}'
                r = periodic(n, lc, which, claimed)
                r['claimed'] = claimed
                r['numpy'] = periodic_check(r) if 'input' in r and n == 16 else None
                out[key] = r
                print(key, {k: r[k] for k in ('omt', 'optimum', 'claimed', 'above_bound', 'seconds')}, flush=True)
                (common.HERE / 'smt.json').write_text(json.dumps(out, indent=1))
    (common.HERE / 'smt.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
