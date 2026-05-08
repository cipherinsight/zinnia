# Benchmark sweep results

Persisted output from `tools/sweep_benchmarks.py`. These JSONs are the canonical
baseline that future SMT-resolver tuning iterations (P5 and beyond) diff against.

## Files

| File | Mode | Result |
| --- | --- | --- |
| `baseline_off.json` | `ZINNIA_SMT_ENABLE=0` (today's default) | 104 PASS / 3 TIMEOUT / 74 FAIL |
| `p3_layered_on.json` | `ZINNIA_SMT_ENABLE=1` (`LayeredResolver::range_then_smt`) | 105 PASS / 2 TIMEOUT / 74 FAIL |

Net: **+1 PASS** (`grayscott` flips TIMEOUT→PASS; took 1284 s, ~21× the 60 s budget).
**0 regressions.** **~25× aggregate compile-time slowdown** across the suite —
the halt trigger that kept `smt_enable` off-by-default after P3.

Captured `2026-05-08` against compiler commit `694efa0` (P2 LayeredResolver) +
the P3 wiring (`7b5b188`).

## Regenerating

```bash
# Baseline
python tools/sweep_benchmarks.py --smt off --timeout 60 \
    --out tools/sweep_results/baseline_off.json

# Layered resolver on
python tools/sweep_benchmarks.py --smt on --timeout 60 \
    --out tools/sweep_results/p3_layered_on.json

# Both, with delta table
python tools/sweep_benchmarks.py --smt both --timeout 60
```

Each pass takes ~80 s in `off` mode and ~40 min in `on` mode (process-pool of
`cpu_count() // 2`). The wall-clock asymmetry is the cost being paid; P5
profiling and tuning targets it.

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
- Before / after any P5 mitigation (timeout tightening, formula-size pruning,
  per-`SiteKind` SMT-skip wiring) so the delta is documented in the commit
  message.
