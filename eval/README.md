# Evaluation data and tools (OOPSLA 2027 R1)

The remote results, scripts and analyses that the paper's numbers and figures are computed from
(figure scripts: `zinnia-paper/oopsla-2027-r1/fig/data/*.py`). Only data on that chain is kept;
per-run log directories are omitted.
Final compiler: branch `sizefix-cairo`, commit 2569c9ed.

| Directory | Contents |
|---|---|
| `literal-remote/fz-run/` | Epoch fz (2026-10-11, 2569c9ed): launcher `fz.sh`, acceptance (`accept-merged.json`), compile timing (`ct.jsonl`), size ablation (`abl214-out/`) |
| `literal-remote/results-tw/` | Remote proving runs: `2026-10-11-fz-*` (all 214 programs, six circuit systems, 3 samples), earlier `tw1`/`aa1`/`aa3` (fixed-point error), `zk20` (zkVM sandwich for 20 programs) |
| `literal-remote/zk42/` | Same-session zkVM sandwich for the 42 copy-only and constant-output programs |
| `remote-results/` | `fin-zd`/`fin-zn` (Zinnia medians the pooled zkVM ratios divide), `ff1-zkvm`/`ff1-npb-zkvm` (SP1, RISC Zero and DS-1000 Cairo runs), `g16` and `g16-rq2-ratios.json` (Groth16-wrapped zkVMs) |
| `final-analysis/` | Pooled zkVM ratios (`fin/zd-pooled.json`, `fin/zn-pooled.json`) |
| `cairo-work/results/cz78/` | Cairo sandwich epochs cz78 and cz10b; runner in `cairo-work/remote/` |
| `all-work/circ327/` | CirC acceptance on the 327 RQ1 programs |
| `all-work/abl214/` | Size-ablation runner (`run.py`, `child.py`, `analyze.py`) |
| `all-work/rq1-ds-accepted.txt` | DS-1000 programs accepted in RQ1 |
| `post-freeze/` | Acceptance results used for the RQ1 taxonomy; remote acceptance and compile-timing scripts used by `fz.sh` |
| `paper-numbers/` | Analyses behind text numbers: A (accepted programs), B (cost of bounds, `arrays.csv`), C (SMT statistics), D (running example), NumPy coverage probe |
| `study-analysis/` | User study (RQ3) task data and analysis |
| `tools/` | Remote launch, pooling, memory cap (`memrun.sh`) and assembly scripts |

Nothing here depends on files outside this repository.
