# Archived benchmark material

Nothing here is read by current code or used for new results. It is kept so earlier
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
