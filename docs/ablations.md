# Ablation switches (measurement only)

These environment variables turn off one part of the compiler, so the evaluation can
report what each part contributes. Each is read by the compiler itself (set it in the
environment of the process that compiles). Unset (for the `ZINNIA_ABLATE_*` switches:
set to anything but `1`), they change nothing: the compiled IR is identical to a build
without them (checked on the 100 DS-1000 and 79 NPBench/Pythran perf cases, both
targets, IR and `main.nr` digests).

## The use of facts

| Variable | What it turns off | Where |
|---|---|---|
| `ZINNIA_ABLATE_SMT=1` | Every Z3 query that reasons about values answers `Unknown` without calling Z3: the IR generator's resolver and `prove()` (bounds, preconditions, index checks, strategy choices), the shape-parameter bounds, and the SMT tier of `AlwaysSatisfiedElimination`. Each caller treats `Unknown` as "no information", so what is left is constants, intervals over the IR (`RangeResolver`) and the scans of interval facts (`ask_bounds`). The construction selection of the lowering layer (a Z3 optimization over costs, not reasoning about values) stays on. | `optim::z3_budget::{check, check_assuming, check_optimize}` |
| `ZINNIA_ABLATE_LAYER_FACTS=1` | The array-lowering layer gets no free facts (no bounds, no gadget postconditions) and ignores the program's checks: no check is dropped as implied by an earlier one or by a construction's postcondition, and no check justifies a construction's precondition. Every precondition fact gets a check node (the selection may still use another construction's postcondition when that is cheaper). | `lower::emit::lower`, `lower::facts::FreeFacts::none` |
| `ZINNIA_ABLATE_CHECK_ELIM=1` | The two optimizer passes that drop program checks, `AlwaysSatisfiedElimination` and `AssertSubsumption`, are not run. Checks the IR generator never emitted (discharged at IR generation by `prove()` or by bounds) are not affected; that is `ZINNIA_ABLATE_SMT`'s territory. | `lib.rs::compile_circuit` |

Related, older switches (not ablations of the fact layer):

- `ZINNIA_SMT_ENABLE=0` replaces only the IR generator's resolver (`require_static_int`
  and the other resolver chokepoints) with the static-only resolver, which also loses the
  interval tier. It does not affect `prove()`, the shape-parameter bounds, or
  `AlwaysSatisfiedElimination` (which builds its own resolver).
- `ZINNIA_REQ_DISABLE=1` makes `prove()` answer `Unknown`, so every op precondition is
  checked in the circuit and every strategy takes its default; the interval fallbacks of
  the index checks still apply.

## The array-lowering layer's selection

The array-lowering layer (`src/lower/`) chooses, for every operator group, a
construction (an implementation node) among the ones its families offer, in one joint
optimization over the whole program (`emit::select_members`). Two switches replace that
choice, so the evaluation can measure what deciding jointly, and deciding at all, is
worth.

| Variable | What it changes | Where |
|---|---|---|
| `ZINNIA_ABLATE_JOINT=1` | Local selection: each group chooses its construction on its own, priced as if it were the only group of the program (definition below). Only the decision changes: the chosen constructions are emitted by the normal machinery, so a shared value two groups both chose is built once, a check a chosen construction proves is dropped, and the verifier challenge is laid out once. | `lower::local`, `lower::emit::lower_with`, `Context::sharing_groups` |
| `ZINNIA_FORCE_CONSTRUCTION=<list>` | No selection: for every group, the first listed construction that applies to it (its candidates produce every live lane of the group) is the only one offered (with its helper, the multiplexer's index bits). The list holds construction names (`Construction::name`) and named configurations `@delegate`, `@selects` (below). A group no listed construction applies to, or whose forced construction cannot be selected together with the rest of the program, keeps the layer's choice (a *fallback*; the layer report counts them). | `lower::forcing` |
| `ZINNIA_LAYER_REPORT=<path>` | Not an ablation: appends one JSON line per lowering round to `<path>`, with each group's family, the constructions applicable to it, the forced one and the chosen ones. The compiled program does not depend on it. | `lower::report` |

### Local selection (`ZINNIA_ABLATE_JOINT=1`)

The compiler's selection pays every shared node once: a value several groups use (the
bits of an index, the one-hot of a scan or of an index wrap, the cells of a written
array), a fact one construction establishes for another, a program check one
construction discharges for several, a program-wide facility (the verifier challenge,
the RAM trace), and the costs some prices divide among the groups that may share them
(the equality tests of reads at one index, the fixed columns of the lookup tables). A
construction that is costly alone can therefore be the cheapest choice in the program.
The local selection removes exactly that credit:

1. **Decisions, one group at a time, priced alone.** The groups decide in group order
   (the order of their first lanes in the program). A group's decision is the cheapest
   selection that justifies its own requirements with every node at its full cost:
   what an earlier group selected is not free, and a price that the compiler divides
   among the groups that may share a cost is the price for this group alone
   (`Context::sharing_groups`). A group's requirements are its live lanes and the
   program checks its constructions can establish or rely on (the checks the compiler
   makes requirements of, attached to every group whose candidates mention them).
2. **Node kinds.** Free facts stay free. A program check stays what it is in the
   compiler: a group may rely on it, keeping it at its cost, or discharge it by its own
   postcondition; a check two groups both count is counted by each. A check node (the
   framework's check of an unproven precondition) is paid by the group that needs it.
   A program-wide facility is paid in full by every group that uses it. A shared value
   is paid by every group that uses it.
3. **Commitment.** Some constructions are sound only together with a particular
   construction of another group (a read through the one-hot of an index wrap relies on
   the wrap's lanes being the ones the program checks; the overlay of a written array
   reads the wrap's one-hot). So a decision is committed: every lane a selected node
   produces is produced by that node. A later group may use a node that produces a
   committed lane only if it is that lane's producer or a *variant* of it (the same
   construction producing a strict superset of its requirements, such as the same wrap
   also exposing its one-hot), which then replaces it. A node that produces lanes of
   several groups (the RAM trace of a written array produces the reads of its versions,
   which are element groups of their own) commits all of them; a group whose lanes are
   all committed has nothing left to decide. This is how written arrays (memory groups,
   their versions and overlays) are handled: whichever of the array's groups decides
   first chooses its representation, paying for it in full.
4. **Completion.** With every lane's producer fixed, the program checks, check nodes,
   shared values and program-wide facilities the decided constructions need are
   selected by the compiler's selection among the nodes that produce no lane.

### Fixed configurations (`ZINNIA_FORCE_CONSTRUCTION=@name`)

One fixed construction per family (`OpKind::family`), whatever it costs
(`lower::forcing::CONFIGURATIONS`). A family not listed keeps the layer's choice.

| Family | `@delegate` | `@selects` |
|---|---|---|
| element (a read at a runtime index) | `memory: element by table read` (Noir: a native array read through bb's ROM; halo2-base: a dynamic-lookup table read) | `gather: element by a select chain` |
| memory (an array written at runtime) | `memory: native RAM` (Noir: bb's RAM) / `memory: RAM trace` (halo2-base) | `memory: overlay of selects` |
| wrap (NumPy's negative index) | layer's choice | `memory: wrap by pinned sign` |
| scan (first / last set flag, argmax of flags) | layer's choice | `gather: scan by a select chain` |
| sort | layer's choice | `sort: network` |
| searchsorted | layer's choice | `searchsorted: count` |
| linear (matmul, correlation) | layer's choice | `linear: direct product sums` |
| is_permutation | layer's choice | `permutation: hit counts` |
| mask write | layer's choice | `mask write: selects` |
| insert, window, permutation scatter, corrcoef | layer's choice | layer's choice |

`@delegate` is what a conventional circuit compiler does with array code: every access
at a runtime index goes through the backend's memory. On Noir the array is a native
array, so Noir's compiler and bb lower the access, not Zinnia; the halo2-base
counterpart (table read, RAM trace) is reported as an ablation only. It fixes the
families that decide how an array is represented and accessed; the other families keep
the layer's choice, because their alternatives differ in computing a result against
checking it (sort, searchsorted, linear, is_permutation), have no memory-based
alternative (wrap, scan, window, permutation scatter, insert, mask write), or have no
choice (corrcoef: the input type decides). `@selects` is the simplest uniform lowering:
select chains and plain compute everywhere a family has such a construction (insert's
two constructions are two forms of the same position one-hot, neither a select).

A RAM construction applies only to an array whose versions form one chain read in
program order; a written array that is not keeps the layer's choice (a fallback). An
element read of a written array lowered by the RAM is produced by the RAM node (it is
not a fallback).

### Measurement scripts (`benchmarking/ablations/`)

* `measure.py --config C`: per perf case and target, halo2-base cells and `k`, Noir
  `bb gates` (`--gates`), IR and `main.nr` digests, and the layer report's summary
  (per family the constructions chosen, the groups lowered through the backend's memory,
  the fallbacks of a fixed configuration). `compare.py` compares two of its outputs.
* `oracle.py`: for every family with two or more applicable constructions in a program,
  the program compiled once per construction forced (one family at a time); the
  compiler's choice against the best and the worst forced construction.
* `soundness.py --config C`: the harness protocol with one proof per case (keys from
  input A, a proof of input B equal to NumPy's outputs, verification, a tampered public
  value rejected).
* `run_configs.sh WT OUT CONFIG [JOBS]`: the remote runner (configurations `final`,
  `abl_joint`, `fix_delegate`, `fix_selects`, `oracle`, `noir_hw`).

