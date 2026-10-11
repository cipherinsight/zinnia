# Brief: CirC acceptance study on the RQ1 corpus (one batch per agent)

## Goal

RQ1 of the paper compiles 327 unmodified NumPy programs with Zinnia (214 accepted). A reviewer
asked how many of these programs a prior SMT-backed circuit compiler, CirC, can compile. You
measure that for your batch by actually running CirC, not by judgement. The result must be
reproducible and fair to CirC.

## Tools

- CirC binary (built, do not rebuild): `~/.cache/zkcc-reconfirm/circ-stock/target/release/examples/circ`
  (`circ --language zsharp <file.zok> r1cs --action count`; see `circ --help`, `circ <f> r1cs --help`).
  Interpreter for checking outputs: `zxi` in the same directory (inputs as `.pin`/`.vin` files; see
  examples under `~/.cache/zkcc-reconfirm/circ-stock/src/examples/ZoKrates/pf/`). Source:
  `~/.cache/zkcc-reconfirm/circ-stock/src` (Z# frontend in `src/front/zsharp`, its stdlib, examples).
- Programs: `wt/literal/<kernel>` (paths in your batch file are relative to
  /home/zhantong/Projects/zinnia-project/wt/literal). Each has a fixture (fixture.json next to the
  kernel, or as the acceptance harness finds it: see wt/literal/benchmarking/ for how
  `eval-redo/post-freeze/data-bcc/accept_final/results.json` was produced) with test inputs A and B.
  Python with NumPy: /home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python.

## Per program: two attempts

Write a Z# (ZoKrates dialect CirC accepts) program for the kernel function, private inputs, public
outputs, the same input domain as the kernel's signature.

1. **As written**: a statement-by-statement translation of the kernel. Keep every array shape,
   loop bound and index exactly as data-dependent as in the source. NumPy calls become the obvious
   Z# loops (vectorized expressions become loops over the static shape). Reals become fixed-point
   integers (scale 2^32, as all other baselines in this project). If a construct cannot be written
   in Z# without changing what the program computes (an array whose length depends on values, a loop
   whose trip count depends on values with no syntactic bound, a dynamic-length output), record the
   attempt as **not expressible** with the construct and the source line. Do not invent a bound.
2. **With minimal rewrite** (only if attempt 1 is not expressible or CirC rejects it): the smallest
   change a Z# programmer would make while computing the same results on every valid input: pick a
   capacity or loop bound that the declared input domain implies (state the argument in one
   sentence), add a mask or guard, pad a dynamic-length output. Do not hand-write a sorting network,
   permutation check, lookup argument, fixed-point division/sqrt/exp/log or any other gadget: if the
   program needs one, record **needs gadget: <which>** instead. Record every rewrite you made.

Compile each attempt with CirC to R1CS under a 300 s timeout and 8 GB memory cap (as Zinnia's
acceptance run; use /home/zhantong/Projects/zinnia-project/eval-redo/tools/memrun.sh with
MEMRUN_MAX=8G and `timeout 300`). Record: accepted / rejected with CirC's error message (first lines)
/ timeout / out of memory, and the constraint count when accepted.

For every accepted attempt, check correctness: run zxi on inputs A and B from the fixture (encode
reals in the same fixed point) and compare with NumPy within 1e-6 relative + absolute tolerance after
decoding. An accepted attempt whose outputs do not match counts as **wrong**, not accepted; fix your
translation if the mismatch is your error.

## Categories (one per program, in this precedence)

- `accepted_as_written`
- `accepted_with_rewrite` (list the rewrites)
- `not_expressible: dynamic shape` / `dynamic loop` / `dynamic index or scatter` (no bound from the domain)
- `needs_gadget: <sort|search|division|sqrt|exp|log|...>`
- `unsupported_feature: <construct>` (Z#/CirC lacks it, e.g. floats with no fixed-point route, strings)
- `circ_error` (CirC rejects or crashes on a faithful translation; quote the error)
- `timeout_or_oom`

## Fairness

Do not look at Zinnia's IR, circuits or sizes. Do not use Zinnia's outcome to decide yours; the
batch file lists it only so the aggregate can be compared. Translate the source as a careful Z#
programmer would; when in doubt, choose the reading that lets CirC succeed. Never weaken a program
to make it compile (dropping outputs, narrowing the domain beyond what the signature declares).

## Files

Work only in /home/zhantong/Projects/zinnia-project/eval-redo/all-work/circ327/<batch>/:
- `<program>/as_written.zok`, `<program>/rewrite.zok` (if any), inputs `.pin` files, logs;
- `<program>/result.json`: {program, category, as_written: {status, error, constraints},
  rewrite: {status, error, constraints, rewrites: [...]} or null, correctness: {A, B}, notes};
- `REPORT.md`: one table row per program (program, category, constraints, one-line reason), counts per
  category, and anything you could not verify.
Do not edit anything outside your batch directory. TMPDIR under your batch directory (/tmp is a small
tmpfs). Never use the remote machine. End REPORT.md with BATCH_DONE on its own line.
