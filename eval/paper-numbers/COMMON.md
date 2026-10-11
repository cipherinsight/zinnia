# Common rules for paper-number tasks (2026-10-09)

You compute numbers for the evaluation section of a paper about Zinnia, a compiler from NumPy
programs to zk-SNARK circuits (halo2-base and Noir backends).

- Final compiler: commit 3ebe0558, built in the worktree /home/zhantong/Projects/zinnia-project/wt/npb-hw
  (use its `.venv/bin/python`; `from zinnia import ZKCircuit`; `ZKCircuit(name, source).compile()` where
  `source` is the function text WITHOUT import lines, as the harness passes it; see
  benchmarking/harness for how the harness compiles kernels and reads fixtures, and reuse that code).
- Benchmarks: wt/npb-hw/benchmarking/datasets (ds1000, ds1000_numpy, npbench, pythran, case_sets/).
  The acceptance account of the paper (214 of 327 accepted, compiler 4115cecb, remote run):
  /home/zhantong/Projects/zinnia-project/eval-redo/final-analysis/fin/remote/post-freeze/data-fin/accept_final/results.json
- NEVER generate proofs or keys, never run timing. Compile, mock-run and count only. Cap memory: run heavy
  Python under `systemd-run --user --scope -q -p MemoryMax=16G ...`, at most 8 parallel processes.
- Do not edit anything in wt/ or zinnia-paper/. Write scripts and outputs only under your task folder
  /home/zhantong/Projects/zinnia-project/eval-redo/paper-numbers/<task>/.
- Every number you report must be reproducible by a script in your folder. Report in
  <task>/REPORT.md: the definition you used, the numbers, the per-program list behind each count,
  caveats, and the command to reproduce. Be precise; do not guess. If a definition is ambiguous,
  pick the most natural one, state it, and give the alternative's number too.
