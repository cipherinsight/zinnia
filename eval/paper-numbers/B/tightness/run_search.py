"""Methods (a) and (b): find, for every inventoried array, a valid input maximising its size,
and for every program a single valid input maximising the total size of its arrays.

Usage: run_search.py <program id> [restrict]   (writes search/<program id>[.<restriction>].json)
       run_search.py      (all 36 programs and the restricted variants, 8 processes, 16 GB cap each)

Candidates, each measured with the NumPy reference (common.Program.sizes), which also
rejects inputs outside the value domain or on which the program raises:
  fixture    the two fixture inputs A and B (the paper's measurement);
  construct  method (a), constructions.py;
  sweep      fixture A with one parameter replaced by each generic pattern;
  random     random combinations of generic patterns, one per parameter;
  climb      hill climbing from the best candidate so far, one run per objective (each
             array's size, and the total), accepting non-decreasing moves.
The generic patterns know only each parameter's dtype and shape, never the program: domain
extremes, constants (zeros, ones, ...), increasing/decreasing ramps, alternations, uniform
random values at several scales, small-range random integers (to create ties) and the
fixture values themselves (permuted, negated).
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common
from constructions import CONSTRUCTIONS

OUT = common.HERE / 'search'
SEED = 20261013
N_RANDOM = 20000
N_CLIMB = 3000

# Alternative, narrower input domains (not declared by the program), searched separately:
# periodic_dist's box length L is a plain Integer, so L <= 0 is valid; physically L > 0.
RESTRICTIONS = {'pythran__pythran_periodic_dist__dist': ('L>0', lambda v: v['L'] > 0)}

INTS = sorted({common.INT_MIN, -2**31, -1000, -100, -50, -25, -20, -16, -10, *range(-9, 10),
               10, 12, 16, 20, 23, 25, 50, 72, 100, 1000, 2**31, common.INT_MAX})
FL = common.FLOAT_LIMIT
FLOATS = sorted({-FL, -1e9, -1e6, -1e5, -1e4, -1e3, -100.0, -10.0, -2.0, -1.0, -0.5, 0.0,
                 0.5, 1.0, 2.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e9, FL})


class Space:
    """Generic random inputs for one parameter, from its dtype and shape only."""

    def __init__(self, param, fixture_values, rng):
        self.p, self.rng = param, rng
        self.shape = tuple(param.shape)
        self.fix = [np.asarray(v) for v in fixture_values]
        self.pool = INTS if param.dtype == 'int' else FLOATS

    def scalar(self):
        d = self.p.dtype
        if d == 'bool':
            return bool(self.rng.random() < 0.5)
        if d == 'complex':
            return [self.pick(FLOATS), self.pick(FLOATS)]
        r = self.rng.random()
        if r < 0.7:
            return self.pick(self.pool)
        if r < 0.85 and self.fix:
            return self.pick(self.fix).item()
        k = self.pick([3, 10, 100, 10**4])
        return int(self.rng.integers(-k, k + 1)) if d == 'int' else float(self.rng.uniform(-k, k))

    def pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def patterns(self):
        """Every deterministic pattern (for the sweep)."""
        if not self.shape:
            vals = [True, False] if self.p.dtype == 'bool' else list(self.pool)
            if self.p.dtype == 'complex':
                vals = [[a, b] for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0)]
            return vals
        out = []
        n = int(np.prod(self.shape))
        base = [True, False] if self.p.dtype == 'bool' else self.pool
        for v in base:
            out.append(self._shape([v] * n))
        if self.p.dtype != 'bool':
            for step in (1, 2, 10, 1000):
                out.append(self._shape([i * step for i in range(n)]))
                out.append(self._shape([(n - 1 - i) * step for i in range(n)]))
                out.append(self._shape([(i - n // 2) * step for i in range(n)]))
            out.append(self._shape([1 if i % 2 == 0 else 2 for i in range(n)]))
            out.append(self._shape([1 if i % 2 == 0 else -1 for i in range(n)]))
        for f in self.fix:
            out.append(self._json(f))
        return out

    def array(self):
        """A random array value."""
        n = int(np.prod(self.shape))
        d = self.p.dtype
        r = self.rng.random()
        if d == 'bool':
            q = self.rng.random()
            return self._shape([bool(x) for x in self.rng.random(n) < q])
        if r < 0.35:
            return self.pick(self.patterns())
        if r < 0.5:
            vals = [self.pick(self.pool) for _ in range(n)]
        elif r < 0.65:
            k = self.pick([1, 2, 3])
            vals = self.rng.integers(-k, k + 1, n).tolist()
        elif r < 0.85:
            s = self.pick([1, 10, 100, 10**4, 10**6])
            vals = (self.rng.integers(-s, s + 1, n).tolist() if d == 'int'
                    else self.rng.uniform(-s, s, n).tolist())
        elif self.fix:
            f = self.fix[self.rng.integers(len(self.fix))].copy()
            flat = f.reshape(-1, 2) if d == 'complex' else f.reshape(-1)
            idx = self.rng.permutation(len(flat))
            flat = flat[idx] * (-1 if self.rng.random() < 0.3 else 1)
            return self._json(flat.reshape(f.shape))
        else:
            vals = [self.pick(self.pool) for _ in range(n)]
        if d == 'complex':
            im = [self.pick([0.0, 1.0, -1.0, self.pick(FLOATS)]) for _ in range(n)]
            vals = [[float(a), float(b)] for a, b in zip(vals, im)]
            return np.array(vals).reshape(self.shape + (2,)).tolist()
        return self._shape(vals)

    def value(self):
        return self.array() if self.shape else self.scalar()

    def mutate(self, v):
        """Change one element (or replace the whole value)."""
        if not self.shape or self.rng.random() < 0.3:
            return self.value()
        a = np.array(v, dtype=object)
        flat = a.reshape(-1, 2) if self.p.dtype == 'complex' else a.reshape(-1)
        i = int(self.rng.integers(len(flat)))
        if self.p.dtype == 'complex':
            flat[i] = [float(self.pick(FLOATS)), float(self.pick(FLOATS))]
        elif self.p.dtype == 'bool':
            flat[i] = not flat[i]
        elif self.rng.random() < 0.5:
            flat[i] = self.pick(self.pool)
        else:
            j = int(self.rng.integers(len(flat)))
            flat[i] = flat[j] + (self.pick([-1, 1]) if self.rng.random() < 0.5 else 0)
        return self._json(a)

    def _shape(self, vals):
        cast = {'int': int, 'float': float, 'bool': bool, 'complex': lambda x: x}[self.p.dtype]
        if self.p.dtype == 'complex':
            vals = [[float(x), 0.0] for x in vals]
            return np.array(vals).reshape(self.shape + (2,)).tolist()
        return np.array([cast(x) for x in vals], dtype=object).reshape(self.shape).tolist()

    def _json(self, a):
        a = np.asarray(a)
        if self.p.dtype == 'complex' and np.iscomplexobj(a):
            a = np.stack([a.real, a.imag], axis=-1)
        cast = {'int': int, 'float': float, 'bool': bool, 'complex': float}[self.p.dtype]
        return np.vectorize(cast, otypes=[object])(a).tolist() if a.ndim else cast(a.item())


def search(pid, restrict=None):
    allowed = RESTRICTIONS[pid][1] if restrict else (lambda v: True)
    P = common.Program(pid)
    name = P.entry.program
    rng = np.random.default_rng(SEED)
    fixture_inputs = [P.fixture_input('A'), P.fixture_input('B')]
    spaces = {p.name: Space(p, [np.asarray(x[p.name]) for x in fixture_inputs], rng) for p in P.params}
    ids = [r['array'] for r in P.rows]
    cap = {r['array']: common.capacity(r) for r in P.rows}
    best = {a: dict(size=-1) for a in ids}           # per array
    found = {a: {} for a in ids}                      # per array: method -> best size by that method
    total_best = dict(total=-1)
    total_found = {}                                  # method -> best total by that method
    evaluations = dict(valid=0, invalid=0)

    generic = []                                       # (input, sizes) of generic candidates

    def consider(values, method, label):
        s = P.sizes(values) if allowed(values) else None
        if s is None:
            evaluations['invalid'] += 1
            return None
        evaluations['valid'] += 1
        if method in ('sweep', 'random'):
            generic.append((values, s))
        for a in ids:
            found[a][method] = max(found[a].get(method, -1), s[a])
            if s[a] > best[a]['size']:
                best[a] = dict(size=s[a], method=method, label=label, input=common.to_json(values))
        t = sum(s.values())
        total_found[method] = max(total_found.get(method, -1), t)
        if t > total_best['total']:
            total_best.update(total=t, sizes=s, method=method, label=label, input=common.to_json(values))
        return s

    for f, v in zip('AB', fixture_inputs):
        consider(v, 'fixture', f)
    for label, over, why in CONSTRUCTIONS.get(name, []):
        values = {**fixture_inputs[0], **over}
        if not allowed(values):
            continue
        if consider(values, 'construct', label) is None:
            raise SystemExit(f'{pid}: construction {label} is not a valid input')
    for p in P.params:
        for v in spaces[p.name].patterns():
            consider({**fixture_inputs[0], p.name: v}, 'sweep', f'{p.name} = generic pattern')
    for i in range(N_RANDOM):
        consider({p.name: spaces[p.name].value() for p in P.params}, 'random', f'random sample {i}')

    # Hill climbing, one run per objective, from the best generic (sweep/random) candidate
    # for that objective, so that the climb never starts from a construction.
    def climb(objective):
        cur, s = max(generic, key=lambda x: objective(x[1]))
        cur_val = objective(s)
        for _ in range(N_CLIMB):
            p = P.params[int(rng.integers(len(P.params)))]
            cand = {**cur, p.name: spaces[p.name].mutate(cur[p.name])}
            s = consider(cand, 'climb', 'hill climbing')
            if s is not None and objective(s) >= cur_val:
                cur, cur_val = cand, objective(s)

    for a in ids:
        climb(lambda s, a=a: s[a])
    climb(lambda s: sum(s.values()))

    out = dict(program=pid, restriction=RESTRICTIONS[pid][0] if restrict else None, seed=SEED, evaluations=evaluations, total_capacity=sum(cap.values()),
               simultaneous=dict(total_best, by_method=total_found), arrays=[])
    for r in P.rows:
        a = r['array']
        out['arrays'].append(dict(array=a, expression=r['expression'], capacity=cap[a],
                                  length_A=r['lengths']['A'], length_B=r['lengths']['B'],
                                  max_size=best[a]['size'], by_method=found[a],
                                  construct=found[a].get('construct'),
                                  generic=max(found[a].get(m, -1) for m in ('sweep', 'random', 'climb')),
                                  best_method=best[a].get('method'), best_label=best[a].get('label'),
                                  best_input=best[a].get('input')))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        pid, restrict = sys.argv[1], len(sys.argv) > 2
        r = search(pid, restrict)
        suffix = '.' + RESTRICTIONS[pid][0] if restrict else ''
        (OUT / (pid + suffix + '.json')).write_text(json.dumps(r, indent=1))
        return

    def run(job):
        pid, extra = job
        cmd = ['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=16G',
               sys.executable, '-B', __file__, pid, *extra]
        with (OUT / (pid + ''.join('.' + x for x in extra) + '.log')).open('w') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=log, env=os.environ).returncode
        print(pid, 'exit', rc, flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [(p, []) for p in common.PROGRAM_IDS] + [(p, ['restrict']) for p in RESTRICTIONS]
        rcs = list(pool.map(run, jobs))
    sys.exit(max(rcs))


if __name__ == '__main__':
    main()
