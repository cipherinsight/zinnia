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
