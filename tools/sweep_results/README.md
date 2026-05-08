# Benchmark sweep results

Persisted output from `tools/sweep_benchmarks.py`. These JSONs are the canonical
baseline that future SMT-resolver tuning iterations diff against.

## Files

| File | Mode | Workers | Wall | PASS / TIMEOUT / FAIL |
| --- | --- | ---: | ---: | --- |
| `baseline_off.json` | off | 6 | 77 s | 104 / 3 / 74 |
| `p3_layered_on.json` | on (no mitigations) | 6 | 2376 s | 105 / 2 / 74 |
| `p5_round1_serial_off.json` | off | 1 | 151 s | 105 / 2 / 74 |
| `p5_round1_serial_on.json` | on (100 ms timeout, 4096 formula cap) | 1 | 150 s | 105 / 2 / 74 |
| `p4_round1_serial_off.json` | off, P4 consumers wired | 1 | 150 s | **104 / 2 / 75** |
| `p4_round1_serial_on.json` | on, P4 consumers wired | 1 | 211 s | **104 / 2 / 75** |
| `p4_round1_5_serial_off.json` | off, round-1.5 visit_while fast-path | 1 | 152 s | 104 / 2 / 75 |
| `p4_round1_5_serial_on.json` | on, round-1.5 visit_while fast-path | 1 | 211 s | 104 / 2 / 75 |

## P4 round 1 — net negative

P4 round 1 wired two consumers of the resolver:
- **While-loop early termination** (`fd1e9e5`) — call `resolve_bool(guard)` after each unrolled iteration in `visit_while`; break on `Some(false)`.
- **AlwaysSatisfiedElimination upgrade** (`ea17474`) — drop assertions when the resolver proves them; raise a compile-time error when proven false.

The expected wins did not materialise:

- **0 of 5 headline sort benchmarks moved** — `insertion_sort` still TIMEOUT, others still FAIL on unrelated issues. The resolver returns `None` for guards that read array elements (`list2[j-1] > save`); range and SMT layers can't reason about heap reads without symbolic representation of array contents (out of scope for this epic).
- **−1 PASS regression** — `factorize_naive/factorize_naive.py` flipped PASS → FAIL because it uses Python's `assert False, "unreachable"` idiom at the end of `while True:`. After Zinnia's bounded unrolling, the assertion's path condition is feasible, so AlwaysSatisfiedElimination's new `Some(false)` arm correctly proves it unsatisfiable and emits a hard error. Pre-P4 the assertion was kept and would have fired at runtime.
- **+41 % aggregate compile-time slowdown on-mode** (149 s → 211 s). Hot benchmarks: `mulmod` 0.08 s → 12.5 s (155×), `guerre` 1.3 s → 47 s (35×), `primes_sieve2` 0.07 s → 1.05 s (14×). The cost is per-iteration `resolve_bool` calls in tight while-loops × `loop_limit` (~1000) iterations × N nested loops. Even when `resolve_bool` quickly returns `None` via the static-val fast path, the call-site overhead dominates these benchmarks' previously sub-second compiles.
- **off-mode also regressed** to 104/2/75 because the AlwaysSatisfiedElimination upgrade fires regardless of `smt_enable` (the constant-fold path catches the `assert False` literal before the resolver is consulted).

The P4 commits stay reachable in git history. Without further work, they're a strict regression — and the user-visible compile-time hit is real. P4 should be revisited when (a) a more powerful resolver or path-condition refinement lifts the sort-benchmark guards, and/or (b) the per-iteration `resolve_bool` overhead is mitigated (e.g., skip when the guard hasn't changed since the last iteration, or only fire when range analysis already produced bounds).

## What the data shows

P3 reported a +1 coverage win at 25× aggregate slowdown and halted. P5 round 1
re-measured with telemetry and serial workers and found:

- **Serial off vs serial on: ratio 0.99×, 0 status movers, 0 benchmarks slow >2×, top slowdown 1.05× (noise).** All P5 targets met (median ≤+20%, p95 ≤+50%, worst case ≤2×).
- **The "P3 25× slowdown" was process-pool contention, not real cost.** Comparing parallel-off vs serial-off for the same compiler binary (no SMT involved): `guerre` 15.5 s → 1.4 s, `perm` 52.7 s → 8.0 s, `grayscott` TIMEOUT → 5.3 s PASS. The contention was 6 workers fighting for CPU + cache, amplified by the CPU-bound compiler.
- **The +1 coverage win (`grayscott` TIMEOUT→PASS in P3) was also a contention artifact.** It passes in serial-off-mode at 5.3 s without the resolver flip. Under contention-free measurement, off and on produce identical bucket counts.
- **Telemetry shows the SMT layer is currently cold across the suite.** `tools/sweep_results/profile_*.txt` (committed in `3ceccdf`) capture the per-benchmark counters: every `require_static_int` query resolves via the `static_val` fast path. Range and SMT layers are wired but not yet exercised — the wins will materialise as future call-site adoption (P3 follow-ups) or P4 consumers (`resolve_bool` for while-loops, `resolve_max` for recursion bounds) start asking the resolver non-trivial questions.

The defensive mitigations in `c75f3fb` (100 ms timeout, 4096-statement formula
cap) cost nothing on the cold path but bound the future worst case if a
consumer reaches the SMT layer with a hard formula.

## Regenerating

```bash
# Off-mode baseline (serial — avoids contention skew)
python tools/sweep_benchmarks.py --smt off --workers 1 --timeout 60 \
    --out tools/sweep_results/p5_round1_serial_off.json

# On-mode (serial — avoids contention skew)
python tools/sweep_benchmarks.py --smt on --workers 1 --timeout 60 \
    --out tools/sweep_results/p5_round1_serial_on.json

# Both, with delta table
python tools/sweep_benchmarks.py --smt both --workers 1 --timeout 60
```

Each pass takes ~150 s serial. **Avoid `--workers >1` for measurement runs** —
the parallel mode is fine for "did anything panic?" smoke tests but skews
compile-time numbers via CPU/cache contention; the P3 baseline data
(`baseline_off.json`, `p3_layered_on.json`, captured at workers=6) is kept here
only as a record of how the contention manifested.

## Schema

```json
{
  "label": "off | on",
  "total": 181,
  "pass": <int>, "timeout": <int>, "fail": <int>,
  "by_exception": { "<ExceptionClassName>": <count>, ... },
  "outcomes": {
    "<benchmark_relpath>": {
      "status": "PASS | TIMEOUT | FAIL",
      "detail": "<exception class>" | null,
      "elapsed": <seconds>,
      "raw": { ... per-circuit details ... }
    }
  }
}
```

## When to regenerate

- After any change to `Resolver` / `SmtResolver` / `RangeResolver` /
  `LayeredResolver` semantics or performance.
- After any change to `IRGenConfig::smt_enable` plumbing.
- After any change to default `smt_query_timeout_ms` / `smt_max_formula_size`.
- Before / after any P5 mitigation so the delta is documented in the commit
  message. Always use `--workers 1` for these.
