# Literal-translation baselines

Every performance case (the 100 DS-1000 cases of `datasets/case_sets/ds1000_perf.json`
and the 79 NPBench/Pythran entries of `datasets/case_sets/npbench_pythran_perf.json`
with status `perf`) has a best-effort *optimized* handwritten baseline in halo2-base
and in Noir. This file defines a second baseline for each case and system: the
*literal translation*. The comparison it supports is "Zinnia versus what a developer
gets by translating the Python program into the circuit language line by line, without
optimizing", next to the existing "Zinnia versus an expert who optimizes".

A literal baseline must be **fair** and **unoptimized**:

- *Fair* means it proves the same statement as the optimized baseline, with the same
  arithmetic, and is sound. Every rule of `CONVENTIONS.md` sections 1, 2 and 4 (the
  statement, numbers, soundness) and of `NPB_FAIRNESS.md` "Required comparison contract"
  items 1, 2, 3 and 5 still holds. It uses the shared arithmetic gadgets of
  `halo2_hw/src/q32*` and `noir_hw` so that its rounding, ranges and transcendental
  approximations are exactly those of the optimized baselines.
- *Unoptimized* means it is a direct translation of `kernel.py`: each Python statement
  and each NumPy call becomes the circuit code that computes it, in the source's order,
  with no restructuring. Section 2 below makes this precise. It replaces
  `CONVENTIONS.md` section 3 ("Idiomatic and cheapest sound") for literal baselines.

Writing an optimized circuit by habit is the main way to get this wrong. When in doubt
between the direct form and a cheaper form, write the direct form. A literal baseline
that is *smaller* than the optimized one for the same case is a red flag; report it
(section 5).

## 1. Files and tools

A literal baseline sits next to the optimized one and carries the tag `literal` before
the file suffix:

| set | halo2-base | Noir |
|---|---|---|
| DS-1000 | `datasets/ds1000/<case>/target.literal.rs` | `datasets/ds1000/<case>/main.literal.nr` |
| NPBench/Pythran | `datasets/{npbench,pythran}/<program>/target.<function>.literal.rs` | `.../main.<function>.literal.nr` |

The environment variable `HW_VARIANT=literal` makes every tool read these files instead
of the optimized ones: the `halo2_hw` build (`build.rs`; module names and case ids stay
the same, so build into its own target directory `target-literal`), `hw_check.check`,
`hw_check.lint`, `hw_check.sizes`, `hw_check.prover_toml` and the Noir driver.
`Prover.<...>.literal.toml` is generated; never edit or commit it by hand.

Run from `benchmarking/baselines` with the worktree's virtual environment:

```bash
export HW_VARIANT=literal
PY=../../.venv/bin/python
MEMRUN=/home/zhantong/Projects/zinnia-project/eval-redo/tools/memrun.sh
E=pythran_lu__factorMatrix0           # or a DS-1000 number such as 0300

# Build only the entries you are working on (dev-fast while iterating).
(cd halo2_hw && HALO2_HW_CASES=$E CARGO_TARGET_DIR=target-literal CARGO_BUILD_JOBS=3 \
    MEMRUN_MAX=8G $MEMRUN cargo build --profile dev-fast)

# The check without proofs: halo2 mock with A and B, cells, out-of-domain input rejected;
# Noir gates, witness solving with A and B, out-of-domain input rejected. Outputs are
# compared with the NumPy reference under the case set's tolerance.
$PY -m hw_check.check $E --no-prove --halo2-bin halo2_hw/target-literal/dev-fast/halo2_hw

$PY -m hw_check.lint $E               # static rules and (NPBench/Pythran) size constants
```

**Never run `hw_check.check` without `--no-prove`, and never run `setup`, `prove` or
`verify`.** No key generation and no proofs on this machine (`NPB_FAIRNESS.md`). Do not
run Zinnia, read its IR or emitted code, or look at its circuit sizes while writing a
baseline.

Size constants: NPBench/Pythran baselines follow `NPB_BASELINES.md` section 3 (every
shape-derived number is a named constant with its formula at the top of the file);
`hw_check.lint` checks it. DS-1000 baselines may use plain constants like their
optimized counterparts.

## 2. What "literal" means

The source is `kernel.py` (the entry function and the helpers it calls). Translate it
under these rules.

**L1. One statement, one piece of circuit code, in source order.** Each assignment,
expression and NumPy call is translated where it occurs. Python helper functions become
circuit functions called at the same places. A Python loop becomes a loop over the same
range in the same order (the circuit unrolls it). Statements are not merged, reordered,
split, or moved out of loops.

**L2. Intermediate arrays are computed in full.** When the source builds an array
(`d = a - b`, `s = np.sort(x)`, `m = x > 0`, `c = np.cumsum(x)`), the baseline computes
every element of it, even if later code reads only part of it (`np.sort(x)[0]` sorts the
whole array; `(a * b).sum()` forms every product, then sums). Values the source never
computes are not computed either.

**L3. No rewriting of the arithmetic.** No common-subexpression elimination across
source expressions (an expression written twice is computed twice), no
reassociation, no algebraic identities (`x*x + y*y` compared with `r*r` instead of
`sqrt(...) < r`, `exp(a)*exp(b)` as `exp(a+b)`), no strength reduction, no hoisting of
loop-invariant work, no symmetry or "only half the matrix" tricks, no fusing of
consecutive NumPy calls. Python-level constants (shapes, literals, `len(x)`) are
compile-time numbers, as in the optimized baselines. Arithmetic on those numbers can be
done at compile time.

**L4. Data-dependent control flow becomes select.** An `if`/`else` (or `np.where`,
a ternary, a conditional `break`/`continue`) whose condition depends on input values
computes both branches and selects, per assigned variable. A loop whose trip count or
exit depends on input values (`while`, `break`, `range(n)` with private `n`) runs to
the bound fixed by the shapes, with an "active" flag that gates every effect of each
iteration. Conditions are evaluated each time the source evaluates them.

**L5. NumPy calls use the textbook in-circuit algorithm.** A library call has no source
code to copy, so it is computed by the straightforward algorithm a developer would write
in the circuit language, computing the result inside the circuit:

| source | literal translation |
|---|---|
| element-wise ops, broadcasting | a loop over the broadcast shape, one gadget call per element |
| `sum`, `prod`, `mean`, `cumsum`, `min`/`max` (reductions) | a running accumulator in index order (`max`/`min`: pairwise `max`/`min` gadget along the scan) |
| `np.dot`, `@`, `tensordot`, `einsum`, `correlate`, `convolve` | the defining sum, term by term in index order; the single rescale of `CONVENTIONS.md` section 2 applies only to these one-call reductions (it is the arithmetic contract, not an optimization); a Python loop accumulating products rescales each product |
| `argmax`, `argmin` | a linear scan keeping (best value, best index), first occurrence wins |
| `np.sort`, `argsort`, `sorted` | a plain compare-and-swap sort over all elements: bubble sort or selection sort written as fixed loops (n(n-1)/2 compare-and-swaps), carrying the index for `argsort`, stable |
| `np.unique`, `np.searchsorted`, `np.median`, `np.percentile`, `bincount`, `histogram` | the textbook definition built from the rows above (sort, then adjacent compares; a scan; counting by a loop of equality tests) |
| `x[mask]`, `np.nonzero`, `np.where(cond)`, `argwhere` | a running count of kept elements, then for each output slot `j` the sum over `i` of `[kept_i and count_i == j] * x_i` (an equality test per pair); the length is public as in the optimized baselines |
| `a[i]` with a private index `i` | the library's standard indexing gadget (`q.index`, Noir `a[noir_hw::index_pos(i, len)]`): this *is* the direct form of an indexed read in each language |
| `a[i] = v` with a private index `i` | for every position `k`, `a[k] = select(i == k, v, a[k])` |
| `/`, `//`, `%`, `np.sqrt`, `np.floor`, `int()`, `astype`, `exp`, `log`, `sin`, `cos`, `tanh`, `arctan2`, `**` | the shared gadget for that operation (`CONVENTIONS.md` section 5 tables, `q32/transcendental.rs`, `q32/canonical.rs`, `noir_hw/src/transcendental.nr`); these hints are part of the arithmetic, not an optimization |
| comparisons `<`, `<=`, `>`, `>=` | one comparison gadget per source comparison: halo2 `q.bound` on each operand, then `q.lt`/..., at the point of the comparison; Noir `noir_hw::lt`/`le`/`gt`/`ge` (or `window` per operand at the point of the comparison) |

**L6. No hints of your own.** The baseline file contains no `load_witness`, no
`unconstrained` function and no `unsafe { ... }` hint call of its own. The only witnesses
that are not computed by constraints are those inside the shared arithmetic gadgets of
the L5 row for `/`, `sqrt`, transcendentals and so on. In particular, do **not** use these
optimized gadgets, which compute by guess-and-check or reuse proofs:

- halo2: `q.sort`, `q.max_of`, `q.min_of`, `q.select_kept`, `q.compact`,
  `q.div_by_abs`, `q.index_i64`, `q.index_onehot`, `q.index_hits`, `q.index_unchecked`,
  `q.load_*_bounded` (and any reuse of an `I64` from one source operation in another);
- Noir: `max_of`, `min_of`, `select_kept`, `compact`, `compact_rows`, `div_by_abs`,
  `pick`, `array_index` with a hinted position, and reuse of a `window` result across
  source operations.

**L7. Statement plumbing is the same as the optimized baseline.** Private inputs,
public outputs and their order, constant outputs left out of the instance, the
`// @outputs:` line in Noir, the input-domain check of every read input element once
(`CONVENTIONS.md` 1.6; this is part of the statement, so it is not "reuse"), the
treatment of data-dependent-length outputs. You may read the optimized baseline of the
same case to copy *this plumbing* and the API usage. Never copy its computation: derive
the computation from `kernel.py` under L1-L6.

**What is still allowed.** Each shared gadget may be internally efficient (that is the
library, not the program). Python-level constant folding (shapes, literal constants).
Constant indices read wires directly (`a[3]`). Exact integer arithmetic for integers
(`CONVENTIONS.md` section 3, last bullet, is a matter of semantics). The circuit
compiler's own optimizations (Noir's SSA passes) are part of the toolchain.

## 3. Semantics must match

Literal does not relax correctness: the same outputs as the NumPy reference on inputs A
and B within the case set's tolerance, the same rejection of out-of-domain inputs, the
same tie-breaking (`argmax` first occurrence, stable sorts), the same behaviour where
NumPy raises (no satisfying witness). The arithmetic contract of `NPB_FAIRNESS.md` item
2 applies unchanged; the checker's NumPy comparison does not replace it.

## 4. Each file's header

Start each literal baseline with a short comment that says, in plain words: the case,
"literal translation of kernel.py (LITERAL_BASELINES.md)", and one line per place where
the translation needed a choice (which textbook algorithm for a library call, how a
data-dependent loop is bounded, any `kernel.py` line whose direct form is not obvious).
Then mirror `kernel.py`: a comment with the source line before the code that translates
it is welcome where it helps a reviewer match the two.

## 5. Report

Each authoring batch writes `LITERAL_BATCH<k>.md` next to this file, with one row per
case: case id, halo2 cells (advice + lookup, as the checker prints) and Noir gates for
the literal baseline, the same two numbers for the optimized baseline (from the same
`--no-prove` check run with `HW_VARIANT` unset, or "not run"), the check verdicts, and
notes. Flag every case where the literal baseline is not larger than the optimized one,
and say why (the case is genuinely direct already, or the optimized baseline is not
optimal, or a rule above was broken). List cases that could not be finished, with the
reason, instead of leaving a partial file that fails the check.
