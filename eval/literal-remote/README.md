# Literal-translation baselines: remote results (2026-10-09)

Remote: shuaicpu6 docker (`zhantong-dev`), checkout `$RUN/zinnia-literal`.

## Citable: three-way timing `tw1`
- Code: branch `literal` = `main` f4d05ef5 (compiler 3ebe0558 from npb-hw; optimized baselines from npb-hw;
  literal baselines per `benchmarking/baselines/LITERAL_BASELINES.md`, arithmetic matched to the optimized ones).
- Systems: zinnia_halo2, halo2_hw, halo2_lit, zinnia_noir, noir_hw, noir_lit, all six in the same session per case.
- 3 repeats, 8 pinned cores and `--threads 8` per job; DS 30G, NPB 64G cap (watchdog). 179/179 ok for every system,
  no case needed the 200G pass.
- `results-tw/` epochs (8 DS + 8 NPB shards; keys/proofs `work/` not copied).
- `analysis/three-way-summary.md`: all pairs, prove/end-to-end/verify, by corpus. `analysis/tw1-ds.json`,
  `analysis/tw1-npb.json`: per-case ratios (eval-redo/tools/pooled.py).
- Script: `three-way.sh` (stages D, N, B). Logs: `logs/`.

## Superseded
- `results/` lit1-ds: literal-only DS timing against the OLD frozen compiler 4115cecb (Zinnia from fin sessions,
  not same-session). `analysis/timing.py`, `analysis/timing-ds.json`. Kept for the record; do not cite.
- `smoke/`: 1-repeat smoke epochs.

## Circuit sizes: `analysis/sizes-summary.md`, per case `analysis/sizes-tw1.json` (from the tw1 setup records, final code).
Older local pre-fix counts: eval-redo/literal-work/merged/zinnia-vs-literal.md (do not cite).
