# Benchmarks

- `datasets/`: the programs.
  - `ds1000/ds1000_case<NNNN>/`: the cross-system performance set (Zinnia `kernel.py` plus
    handwritten halo2-base, Noir, Cairo, SP1 and RISC Zero baselines).
  - `ds1000_numpy/`: the DS-1000 NumPy problems (original reference solutions), for the
    acceptance account; `prep/` holds the task classification.
  - `npbench/`, `pythran/`: upstream NPBench and Pythran kernels (see `datasets/FILTERING.md`),
    for the acceptance account; `contracts/` holds their generated contract drafts.
- `baselines/`: rules and support code for the handwritten baselines (`CONVENTIONS.md`,
  `NOIR_BASELINES.md`).
- `noir_fpc/`: the Q32 fixed-point Noir library shared by the handwritten Noir baselines and
  Zinnia's Noir backend (embedded by `src/prove/noir/mod.rs`, so it stays here).
- `archive/`: results and runners that are kept for traceability only (see its README).
- The `*.py` scripts at this level are the pre-redo runners and validators; they are being
  replaced by a single harness.
