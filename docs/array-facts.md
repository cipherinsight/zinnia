# Array facts: facts about whole arrays

What every element of an array, or of a part of it, satisfies, kept once per array
instead of once per element, and how the compiler uses it. This is the port of the
`array-facts` branch (35bebe45, designed on the memory-segment frontend) onto the
array-SSA frontend (`docs/array-ssa.md`). Code: `src/optim/array_facts/` (the store and
the value domain), `src/builder/array_facts.rs` (stating, deriving and querying facts),
`src/builder/bound_queries.rs` (the query tiers), and the sources and consumers listed
below.

## 1. Terms

- **Cell**: one scalar element of an array; in the IR, the statement whose value it is
  (or a compile-time constant).
- **Storage**: the compile-time identity of the cells some Python names reach
  (`builder/store.rs`). An array and every view of it (a slice, a transpose, a
  contiguous reshape, an alias `b = a`, an element of a list holding it) share one
  storage. A storage is in one of two states: `Cells` (the frontend knows every cell as
  an SSA value; no IR exists for the array) or `Version` (after a write at a runtime
  position, the cells are those of an IR array version).
- **Array value**: what a storage holds at one moment: its list of cells, or its
  version statement (`ArrayInit` or `ArrayWrite`). Both are immutable: a write gives
  the storage a *new* array value.
- **Segmentation** (FunArray, P. Cousot, R. Cousot, F. Logozzo, POPL 2011): an
  array's positions cut into consecutive segments `{b0} E1 {b1} ... Ek {bk}` with
  `b0 = 0`, `bk = n`, each with the element facts `Ej` every cell in it satisfies.
  A bound is a constant or a runtime integer plus a constant (`count - 1`); a segment
  whose bounds are not known to differ is possibly empty, and a position whose place
  relative to a symbolic bound is unknown gets the join of both neighbours' facts.
- **Element facts** (`ElemFacts`): a range `lo <= a[p] <= hi` (ends constant or
  symbolic, either may be absent), an order along positions (`<=`, `<`, `>=`, `>`),
  and an affine formula `a[p] = base + step * p`. A segmentation may also state that
  the array is a **permutation** of `0..n`.
- **Array fact**: a segmentation stated about one array value.
- **Guard** (`FactGuard`): where a fact holds. `Always` (on every path) or
  `When(atoms)`: where every listed Boolean IR value is true, the conjuncts of an
  execution predicate (`builder/guard.rs`). The scalar facts of the `FactStack` use
  the same guards.
- **Context** (`Where`): what a derivation or a query may use. `Everywhere`: only
  what holds on every path (constants, intervals over the IR, the fact stack's
  unconditional facts, array facts stated `Always`). `Here`: everything the current
  execution guard makes visible.
- **Source**: where an array fact comes from. **Consumer**: a decision that reads one.
- **Trust rule** (`PAPER-NOTES.md` section 5): a fact may be stated only if a
  constraint enforces it on every accepted witness where its guard holds, or it holds
  by the definition of the cells.

## 2. What an array fact is about, and why a write expires it

An array fact describes one array value, never a variable, a value id or a storage.

- **Cell lists.** A source states a segmentation over a list of cells
  (`ArrayFacts::state`). The store indexes, for each runtime cell, the lists it is in
  and its position. The facts of an array are found from its *current* cells: the
  facts of every stated list of which they are a run of consecutive positions, in
  order or reversed (`list_facts`), and each cell's own facts. Constants carry their
  own facts by inspection (`constant_facts`: range, order, formula; float constants
  only a non-strict order, since rounding to the fixed-point grid may tie them).
- **Versions.** An `ArrayInit`'s facts are those of its cells (computed when first
  asked, per context). An `ArrayWrite`'s facts are derived from the version it writes
  when the write is built (section 4).

Every query (`view_facts`, `array_range`, `array_has_order`, the read rule) looks up
the storage's current array value at the time of the query. A write through *any*
name of the array (the array itself, an alias, a view, a container element: all share
the storage) changes the storage's value: a write at a constant address of a `Cells`
storage replaces one cell (`c[a] := x`, or `select(p, x, c[a])` under a runtime
predicate), a write at a runtime position makes a new version. The next query sees the
new value, whose facts are its own. Facts about the old value stay true (about the old
cells), and a view that is still a run of the old cells keeps them, correctly.

**Copies NumPy would share.** Where Zinnia copies what NumPy returns as a view
(`np.flip`, a runtime-bounded slice, ...), the two storages are linked
(`builder/stale.rs`), and a write to either one makes the other stale. A stale storage,
on any path, has no facts (`storage_is_current`); reading its cells is a compile error
anyway.

This replaces every value-id keyed array fact of the base (AF-1: a `forall` bound
outlived `a[0] = x`; the `is_sorted` fact outlived `y = x; y[0] = 100`): the
`array_in_range` predicate and its write-clock records, the `forall` name scan, the
storage cell bounds of `np.nonzero`, the known-permutation list, and the `is_sorted`
records and contracts.

## 3. Where facts hold

A fact derived from what is known where the code runs holds there. Every stated fact
carries a guard, and a query sees a fact only when the fact's atoms are among the
query's context's (`holds_in`): a fact stated under `if c:` is used under `c` and
under `c and d`, never on every path or under another condition, and nothing guarded
is used where the code never runs.

- **Definitional facts** (an operator's semantics, a check at entry on every path)
  are stated `Always` (`state_defined_facts`).
- **Derived facts** (a transfer of what is known of the operands) are derived twice
  (`state_derived_facts`): from what holds on every path, stated `Always`; and, under
  a runtime guard, from what holds where the code runs, stated with that guard when
  it gives more. Each derivation uses only its own context: a scalar bound from the
  fact stack's guard frames or a branch's implications is a `Here` bound
  (`derive_bounds_from_facts`); `Everywhere` uses `derive_bounds_from_unconditional_facts`
  (the scopes' facts, no implication) and intervals over the IR.
- **Symbolic bounds** are compared by an oracle of the same context (`Oracle`).

## 4. The rules: read, write, merge

**Read.** A read `ArrayRead(v, access, positions)` yields a cell of `v` at an address
the access can name (an `ArrayRead` with a position outside its axis has no witness;
positions are guarded into range where the code does not run). Its range is the
facts of `v` over those addresses (`read_range`; the addresses narrowed by the
positions' intervals over the IR). A read folded to a cell (`builder/arrays.rs`) is
that cell, with its own facts. This is how an array fact becomes a fact about a
scalar, and it covers every gather of known cells at a checked index (`np.take`,
fancy indexing, `read_cells`), each an `ArrayRead` of an `ArrayInit`.

**Write.** (`derive_write_facts`, `derive_select_cell_facts`)

- A write on every path at a constant address of a version replaces that cell: the
  segment holding it splits into the run before it, the cell (with the written value's
  range) and the run after it; the runs keep their facts (`Segmentation::strong_write`,
  FunArray's split). A constant write to a `Cells` storage replaces the cell itself.
- Any other write (a runtime position, or a runtime predicate) may change any cell it
  can reach, and only to the written value: every segment the write may reach has its
  range joined with the written value's, and loses its order and formula; the
  permutation is dropped (`Segmentation::weak_write`). Under a runtime predicate a
  write to a known cell stores `select(p, x, c)`, whose range is the join of `x`'s and
  `c`'s.
- Why the result holds wherever the old version's facts do: where the write does not
  happen the cells are the old ones; where it happens every cell but the written one
  is old, and the written one is the value, whose range is taken in the write's own
  context (the write happens only where its predicate, hence the current guard,
  holds).

**Merge.** A branch join merges two arrays of one shape cell by cell under one
condition `c` (`helpers::value_ops::select_value`): the merged array is the
then-array or the else-array, as a whole. What holds for both holds for it
(`Segmentation::either`): the ranges' join, an order both sides have (a choice
position by position, as `np.where`, keeps no order), a permutation both are, a
formula both share (`state_merged_facts`). Each side's facts are those that hold at
the join.

## 5. Sources and why each fact holds

| source | code | fact | why it holds |
|---|---|---|---|
| `@requires(forall(a, in_range(lo, hi)))`, `lt_bound(B)`, `is_nonneg`, `eq_const(C)` | `ir_gen/array_fact_sources.rs` | range on the input's cells (`lo`, `hi`, `B`, `C` literals or integer inputs: symbolic) | the predicate's witness emitter has just asserted it for every element, under the current execution predicate (the stated guard) |
| `@requires(is_sorted(a))` | same | the input's cells non-decreasing (row-major) | every consecutive pair is asserted `<=` |
| `@requires(is_permutation(p))` | `IRBuilder::record_permutation` | permutation of `0..n` | the check of `lower/permutation.rs`, which the layer keeps |
| Boolean input arrays | `state_boolean_input_facts` | `[0, 1]` | each element's domain check `x (x - 1) = 0`, emitted on every path when the input is read |
| element-wise `+ - *`, negation, `// c` (`c > 0`) of integer arrays | `helpers/static_array/elementwise.rs::state_elementwise_facts` | interval transfer of the operands' ranges (derived) | exact integer arithmetic over the field (i128 interval ends; an end outside i64 is dropped); Python floor division |
| `np.minimum`, `np.maximum`, `np.clip` | `ops/np_like/mod.rs::state_minmax_facts` | min/max transfer (derived) | each cell is a select on an exact comparison of one cell of each operand |
| `np.where(c, x, y)` | `state_selected_from` | join of `x`'s and `y`'s ranges (derived) | each cell is `select(c_i, x_i, y_i)` |
| `np.sort` / `np.argsort` lines | `helpers/static_array/sorting.rs::state_sorted_line_facts` | sort: non-decreasing (definitional) and in the inputs' range (derived); argsort: permutation of `0..n` (definitional) | every sort construction's lemma: the lanes are NumPy's stable sort of the line on every accepted witness, and the `Sort` lanes have no execution predicate |
| a sorted lane at a runtime rank | `IRBuilder::ir_sorted_at_runtime_rank` | in the inputs' range (derived) | the lane is one of the sorted lanes (no witness for a rank outside) |
| `np.unique` | `ops/static_ndarray_ops/indexing.rs::state_unique_facts` | values `{0} < {count} [0,0] {n}` and in the inputs' range (derived); inverse in `[0, count - 1]` | computed in-circuit from comparisons, selects and sums, no hint: the cell of rank `k < count` is the unique first occurrence with `k` distinct values below; the first occurrence of `x_i`'s value is counted in `count` but not in `rank[i]` |
| `np.nonzero` / `np.flatnonzero` | `state_nonzero_facts` | `{0} [0, extent - 1] order {count} [0,0] {n}` (1-D increasing; rows of 2-D non-decreasing) | the compaction network (`compact_by_mask`) computes lane `j` as the coordinate of the `(j+1)`-th set cell or the fill 0, with selects and products only |
| `np.argwhere` | `np_argwhere` | every packed cell in `[0, max extent - 1]` | as `np.nonzero` |
| boolean-mask read | `helpers/static_array/mask.rs` | the input's range joined with 0 (derived) | each output cell is an input cell or the padding 0 (compaction) |
| boolean-mask write | `deposit_mask_write_post` | the old range joined with the value's (derived) | each new cell is its old value or (an element of) the written one |
| runtime-shaped `array op scalar` | `ops/dyn_ndarray/binary.rs` | interval transfer, on the output's storage (derived) | output cell `k` is `op(cell k, scalar)` for every capacity cell |
| flattened `np.cumsum` of cells in `[lo, hi]`, `lo >= 0` | `ops/static_ndarray_ops/reductions.rs::state_cumsum_facts` | non-decreasing, `[lo, n * hi]` (derived: rests on the input's range) | partial sums of non-negative integers, exact in the field |
| `np.linspace` with a runtime length, `start <= stop` | `constructors.rs::np_linspace_bounded` | non-decreasing over the capacity | cell `i` is `start + i * step`, `step >= 0`, the integer-times-fixed-point product exact; the integer cast does not decrease |
| `np.arange` | constructors | (nothing stated) | its cells are constants: range, strict order and formula are read off them |
| `min` / `max` reduction | `reductions.rs::propagate_forall_bounds_to_reduction_output` | the input's range (derived), also as scalar facts for `prove` | the output is one of the elements |
| branch merges | `state_merged_facts` | `Segmentation::either` (derived) | section 4 |

## 6. Consumers and why each use is sound

Every consumer asks in context `Here`, so it uses only facts that hold where the code
it emits runs; the checks it drops would have been emitted under that same execution
predicate.

| consumer | code | decision | soundness |
|---|---|---|---|
| bound tiers | `cheap_bounds`, `ask_bounds` | an integer's bounds: the meet of the scalar tiers and its array facts | both hold where the code runs, so their intersection does |
| index wrap | `helpers/static_array/base.rs::normalize_dynamic_index`, `array_ops/indexing.rs` | skip the wrap of an index proved non-negative; use the cheap sign test for a narrow one | a decision query (section 7): proved, not assumed |
| index and write checks | `optim/resolver/discharge.rs` (`probe_in_range`, `provably_in_range`, `discharge_index_in_range`), `static_array/write.rs` | drop `lo <= i < hi` when proved | as above |
| index against a runtime length | `discharge_index_below_value` (`index_facts_below`) | drop `i < len` when the index's array facts bound it by `len - 1` (np.unique's inverse against the count) | the fact's symbolic end is the same statement as the length |
| Boolean form of `& | ^` | `value_ops::is_zero_or_one` | an operand proved in `[0, 1]` | decision query |
| operator preconditions | `IRBuilder::discharge_requires` (`cheap_tiers_prove`) | a precondition comparing integers with constants (a divisor `!= 0`, an exponent `>= 0`) proved by the cheap tiers needs no check | the tiers' bounds hold where the operator runs |
| non-empty reduction, integer power | `aggregation.rs::require_non_empty`, `ir_pow_i` | skip the check when proved | decision query |
| permutation scatter | `lower/gather` via `is_known_permutation` | `inv[p] = v` without a permutation check when `p`'s cells are a stated permutation (an asserted `is_permutation`, an argsort line) | the permutation holds of exactly those cells in that order (`list_facts` requires the whole list) |
| boundary reads of a sorted runtime-shaped array | `aggregation.rs` (`live_prefix_has_order`) | `min = x[0]`, `max = x[len - 1]`, `argmin = 0`, `argmax` = first position of the last value | the live prefix lies in an ordered segment of the array's current contents (a segment over the whole capacity holds it: the live length never exceeds the capacity, and a read past it has no witness) |
| layer: argsort permutation | `lower/facts.rs::is_argsort_line` | `Claim::Permutation(p, 0..n)` free when `p` is the `n` argsort lanes of one sort instance, every rank once | the sort construction's lemma, read off the graph (no fact table survives the passes' renumbering) |

## 7. The query tiers

Two kinds of question are asked of the facts (`builder/bound_queries.rs`):

- **Decision queries** (`proves_at_least`, `proves_at_most`, `proves_within`): "does
  `v >= c` hold where the code runs?". The tiers, cheapest first, each tried only when
  the cheaper ones fail: constants; intervals over the IR (`RangeResolver`) and the
  scalar facts (the fact stack, one step of arithmetic over facts, the bounded walk);
  the array facts of the arrays `v` is a cell of or is read from (and one step of
  arithmetic over them: `x + c`, `x - c`, `x * c`); then one satisfiability check of the
  violation `v < c` (`Resolver::proves_bound_with_stmts`). A check that only tightens a
  bound a cheaper tier gives gets a tenth of the step budget (`Effort::Tighten`). An
  unknown answer is never read as false.
- **Bound queries** (`ask_bounds`, at sizing sites: a loop's trip count, a capacity):
  the interval engine, the SMT optimizer when the intervals give no answer, the scalar
  facts, then met with the array facts.

Array facts never reach the solver: a segmentation already states what holds for
every cell, so no quantified formula is built. The SMT resolver remembers each
statement's extremes and decisions (a statement's formula and the program's
preconditions are fixed once they exist).

## 8. Not done, and differences from the old branch

- **Zeros/ones content facts** (`forall_eq_const`, relayed through ~50 view operators,
  `optim/resolver/relays.rs`) are still value-id facts; branch `fact-staleness` makes
  them expire on writes. They are a candidate for the store (a range `[c, c]`), to be
  reconciled when that branch lands.
- **Runtime-shaped merges**: a merge of two runtime-shaped arrays gets no merged
  facts (only static arrays of one shape do).
- **Float ranges** are not facts (they would be about the fixed-point encoding, and
  their consumers, `sqrt`/division guards, are free in the backends); float lists
  carry order only.
- **No widening** (loops are unrolled), and one expression per segment bound.
- The old branch's per-storage write history (`StorageFacts`) is replaced by the
  version facts; its `read_ranges` snapshot is the version a read names.

## 9. Interface to the lowering layer (proposal)

The lowering layer (branch `fact-rules`) states array claims over lists of IR values
(`ElementsWithin`, `NonDecreasing`, `StrictlyIncreasing`, `Distinct`, `CountBound`,
`Permutation`) and establishes them from definitions read off the graph (sort lanes,
argsort) and from the program's checks. The frontend's store knows more (facts derived
by transfer through operators, `np.unique`'s and `np.nonzero`'s segmentations, the
`@requires` element predicates, the facts of written versions), but it lives only in
the IR builder. A fact the store proves can reach the layer as an IR statement:

- `IR::ArrayFact { claim }` with operands `[guard, x_0, ..., x_{n-1}]` (or a scope
  `Everywhere` without the guard operand): "on every accepted witness where `guard`
  holds, `claim` holds of the values `x_k` in this order"; `claim` carries the
  segmentation's vocabulary for the layer's claims (a range with constant or operand
  ends, an order, a permutation of `0..n`, a cut at an operand).
- Emitted by the builder when a fact is stated (`state_array_facts`,
  `state_version`), with the stated guard's conjunction as the guard operand. A fact is
  about values, which never change, so it needs no expiry in the IR: the facts of an
  array after a write are about other values (the new cells, or the new version).
- Not a use: dead-code elimination drops an `ArrayFact` whose operands are otherwise
  dead and never keeps an operand alive for it; `DuplicateCodeElimination` and
  `ConstantFold` rewrite its operands like any statement's. The layer collects them
  into `Fact { claim, scope }` and removes them; no backend sees one.

Not built: no measured case needs a fact the layer cannot already establish, and the
interface is the layer's to define (`docs/array-facts-lowering.md` on `fact-rules`).
On merge, the layer's `array_facts::definitional` subsumes `FreeFacts::is_argsort_line`
(the argsort permutation), which should then be removed.

## 10. Tests

- `src/optim/array_facts/tests.rs`: domain, segmentations (cuts, slices, the write
  rules, merges), the store (runs, AF-1, guards), constants.
- `src/builder/array_facts.rs` (tests): writes through any name, runtime and
  conditional writes, branch-local facts, merges, stale copies, permutations.
- `src/optim/resolver/tests.rs`: decision queries and remembered extremes.
- `src/lower/facts.rs`: the argsort-line permutation.
- `testing/lang/test_array_facts.py`: every source and consumer above, and the
  adversarial cases (writes through aliases, views, strided and reshaped views,
  containers, `out=`, augmented and mask writes; writes under branches, in loops with
  an early exit; facts learned in a branch used after it; written argsort, unique and
  nonzero results), each on the mock against NumPy, with halo2-base and Noir proving
  the mock's values, and the violating inputs without a proof.
- `testing/lang/test_sorted_reductions.py`, `test_stale_array_bounds.py`: the
  boundary reads and the AF-1 reproducers.
