# Archived benchmark material

Nothing here is used for new results, and no current code reads it. It is kept so earlier
claims can be traced to the files that produced them.

- `2026-03-submission/`: the runners, datasets (LeetCode, ML, crypto, early DS-1000) and
  `results-*.json` behind the OOPSLA 2026 submission, copied from the machine that
  produced them (shuaicpu6, `~/zinnia/benchmarking`, repository commit 7f9e74a of
  2026-03-17). Its `plotter_for_paper.py` and `table_generator.py` made the paper's figures
  and tables.
- `2026-06-agent-sweeps/`: every result, comparison and survey file written at the top of
  `benchmarking/` between May and September 2026 (agent sweeps, four-way comparisons,
  halo2/Noir comparison reports), plus the unused CirC port classification `circ_ports/`.
  These numbers predate the October 2026 soundness audit (see `eval-redo/audit/` outside
  the repository) and must not be compared with new results.
- `2026-06-agent-sweeps/runners/`: the runners, validators and analysis scripts that wrote
  those files, retired on 2026-10-06 in favour of `benchmarking/harness/` (audit
  `eval-redo/audit/harness.md`, section 4.3):
  - `run_zinnia_halo2.py`, `run_zinnia_noir.py` -> `harness/adapters/zinnia_halo2.py`,
    `zinnia_noir.py` (no legacy `halo2` backend, no keygen subtraction, no `/tmp` mtime lookup);
  - `run_halo2_baselines.py` -> `harness/adapters/contract.py` (`halo2_hw`); its
    `INPUT_BUILDERS` and abandoned-case list are superseded by the per-case `fixture.json`
    files and the case sets of the dataset workstream; `run_noir_baselines.py` -> `harness/adapters/noir_hw.py` (its per-case
    `WRITERS` are replaced by one fixture-driven writer);
  - `audit_bindings.py` -> `harness/outputs.py` (ordered comparison, real backends, tamper
    check); `ensure_release_build.py` -> `harness/build.py`;
  - `validate_contracts.py`, `validate_contracts_parallel.py`, `validate_ds1000_numpy.py` ->
    `harness/acceptance/validate.py`;
  - `combine_halo2_results.py`, `analyze_noir_vs_zinnia.py`, `analyze_halo2_slowdowns.py`,
    `run_zinnia_backend_probe.py`, `summarize_backend_probe.py`, `render_contract_review.py`:
    no successor (`harness/analysis/` computes summaries from an epoch directory; process
    isolation is part of the protocol).
  They are kept unmodified and no longer run from here (their paths point at the old layout).
