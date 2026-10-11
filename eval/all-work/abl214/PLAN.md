# Ablation over the 214 RQ1 programs (sizes only, no proving)

## Compiler and programs

Compiler: `wt/literal`, branch all-merged, 6fa27f8e. The remote checkout `$RUN/zinnia-literal`
is at a65574b5, an ancestor of 6fa27f8e. The 10 commits between them only add zkVM/Cairo
baselines: `git diff a65574b5 6fa27f8e` is empty for `src`, `zinnia`, `benchmarking/ablations`,
`benchmarking/harness`, `noir_gates.py`, `noir_fpc` and the case sets. Its extension was built
at a65574b5, clean, with both backends (`target/zinnia-harness-build.json`). Use it as is.

Programs (`run.py` writes them to `OUT/programs.json`):
- 132 DS-1000 cases from `rq1-ds-accepted.txt`, all `perf` in `ds1000_perf.json`;
- 82 NPBench/Pythran `perf` cases from `npbench_pythran_perf.json`;
- total 214. RQ2 compares 172 of them: it drops 36 copy-only programs
  (`data_movement_only`) and 6 constant-output ones (`CONSTANT_OUTPUT` in
  `fig/data/rq2-perf.py`). The two groups do not overlap.

## Variants in the current compiler

Each variant is selected by environment variables that the compiler reads itself
(`docs/ablations.md`). Unset, they change nothing. `run.py` removes every inherited
`ZINNIA_ABLATE_*` / `ZINNIA_FORCE_*` variable before setting the variant's own.

| variant | environment | what it disables | paper wording (4115cecb study) |
|---|---|---|---|
| `final` | none | nothing (the reference) | |
| `abl_joint` | `ZINNIA_ABLATE_JOINT=1` | joint selection: each operator group picks its construction alone, priced as if it were the only group | "each construction on its own" |
| `fix_selects` | `ZINNIA_FORCE_CONSTRUCTION=@selects` | selection: one fixed construction per family (select chains, plain compute) | "every access by a chain of selects" |
| `fix_delegate` | `ZINNIA_FORCE_CONSTRUCTION=@delegate` | selection for arrays: every runtime-index access goes through the backend's memory (Noir ROM/RAM, halo2-base table read / RAM trace) | "what a conventional compiler does" |
| `abl_layer_facts` | `ZINNIA_ABLATE_LAYER_FACTS=1` | facts in construction selection: no free facts, program checks neither dropped nor used | "no facts in construction selection" |
| `abl_array_facts` | `ZINNIA_ABLATE_ARRAY_FACTS=1` | array facts everywhere (frontend store and the layer's array claims) | fact ablations |
| `abl_smt` | `ZINNIA_ABLATE_SMT=1` | every Z3 value query answers Unknown (the layer's cost optimisation stays on) | fact ablations |
| `abl_check` | `ZINNIA_ABLATE_CHECK_ELIM=1` | the two passes that drop program checks | fact ablations |
| `abl_all_layer_facts` (opt-in) | `LAYER_FACTS=1` + `ARRAY_FACTS=1` | both fact switches of the layer | combination never measured before |

Not included: the oracle (`oracle.py`, one family forced at a time) and the older
`ZINNIA_SMT_ENABLE=0` / `ZINNIA_REQ_DISABLE=1`, which `docs/ablations.md` lists as
"not ablations of the fact layer".

Expected failure: `abl_smt` rejects `ds1000_case0389` (proving the divisor non-zero needs
Z3). The smoke test reproduced this. `analyze.py` lists the failure separately and leaves
the program out of that variant's geomean.

## Per-program measurement

Run in the checkout, with the variant's environment variables set:

```
WT/.venv/bin/python WT/benchmarking/ablations/capped.py 64G 3600 \
  WT/.venv/bin/python abl214/child.py WT KERNEL FIXTURE FUNCTION {halo2|noir}
```

- `child.py` compiles once with `compile_kernel`.
- halo2: `halo2_base_cell_stats` (witness-free, no keygen). `size = advice + lookup` at the
  `k` that `estimate_params` picks. It also records `k`, `poly_cost` and the IR digest.
- Noir: `emit_structural` on input B, then `nargo compile` + `bb gates`
  (`--scheme ultra_honk --verifier_target noir-recursive`, as `noir_gates.py` runs it).
  `size = circuit_size`.
- Each result also stores the layer report's summary: constructions chosen per family,
  native-memory groups and fallbacks.

**Difference from RQ2's halo2 size.** In `cell_stats`, `advice` includes the advice cells of
lookup tables and RAM traces. RQ2 (`rq2-perf.py`, `size`) parses the harness fingerprint, which
counts the base builder's advice only. RQ2's Zinnia halo2 sizes are therefore smaller wherever a
table read or RAM trace is chosen:
- 0319: 14,833 vs 15,984 here;
- 0450: 17,745 vs 18,304;
- insertion_sort (no table): 3,810 vs 3,809.

The ablations use the full count, as the earlier ablations did. The full count is also the
fair one for `fix_delegate`, which selects tables and RAM traces. Noir gates match RQ2 exactly.

## Runner and analysis

- `run.py`: one job per (variant, backend, program), 8 × 2 × 214 = 3,424 jobs.
  - Writes one JSON per job, atomically, to `OUT/results/<variant>/<backend>/<program>.json`.
  - Resumable: a rerun skips every existing result. `--retry-errors` redoes the failed ones.
  - Runs a thread pool of `--jobs` children with no core pinning, at `nice 10`.
  - Starts a child only while MemAvailable > `--min-free`. Each child has an RSS cap
    (`--mem-cap`) and a timeout.
  - Runs every `final` job first, then the other variants, slowest program first.
  - Writes `manifest-*.json`: commit, extension SHA-256, build record, nargo/bb versions.
- `analyze.py OUT [--json summary.json] [--metric poly_cost]`, per variant and backend:
  - geomean of variant/final over the 214 and over the 172 (with n);
  - programs that change, grow or shrink, and those whose IR changes but not the size;
  - the largest increases, by name;
  - failures on either side.

## Cost

Local measurement, `final` only, 428 jobs on 8 parallel jobs:
- 0 errors, 205 s wall, 1,091 job-seconds;
- peak child RSS 5.6 GB (scattering, Noir);
- the slowest jobs: resnet (140 / 148 s), vadv (90 / 97 s), ccopy (62 / 71 s), scattering
  (40 / 103 s) (halo2 / Noir).

All 8 variants come to about 9k job-seconds. At 56 jobs on the remote that is roughly
5–15 min wall, bounded by the slowest single compiles. `fix_selects` is the heavy one: at
4115cecb, scattering on Noir exceeded a 24 G cap, which is why the cap here is 64 G. If it
fails anyway, rerun just that job with `--variants fix_selects --programs scattering...
--mem-cap 200G --retry-errors`.

## Launch (only after the zk20 / zk42 timed epochs finish)

```
eval-redo/all-work/abl214/launch-remote.sh          # refuses while harness runs are active
eval-redo/tools/remote.sh log abl214
eval-redo/tools/remote.sh pull /home/zhantong/zinnia-eval-run/abl214-out eval-redo/all-work/abl214/
```

The remote writes `abl214-out/tables.md` and `abl214-out/summary.json` when it finishes.

## Local smoke test (`smoke-local/`)

Run with `final` + `abl_joint` on 0319, 0450 and Pythran insertion_sort, both backends:
- 12/12 jobs succeeded in 3 s.
- A rerun skipped all 12 (resume works).
- `abl_smt` on 0389 recorded the expected rejection.
- A child forced over a 50 M cap recorded "over the memory cap"; `--retry-errors` then
  remeasured it.
- `abl_joint` on halo2 gave 0319 ×1.93 and 0450 ×1.68 (4115cecb: ×1.92 / ×1.68);
  insertion_sort and every Noir result were unchanged.
- `cost-final-local.log` is the 428-job local run used for the cost estimate.
