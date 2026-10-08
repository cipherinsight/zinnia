# Array facts in the lowering layer: claims, sources, and the interface for the IR generator

The array-lowering layer (`src/lower/`) chooses, per array operator, a construction whose
preconditions are justified and whose cost is lowest. Some constructions rely on facts about
whole arrays: a sort of an input that is already in order is the input itself, a range check of
a narrow range makes every comparison of the value narrow, a permutation's values need no index
check. This page describes how such facts reach the layer, and how the IR generator states the
facts it knows.

Code: `src/lower/array_facts.rs` (claims, implication, bases, definitional facts, the stated
facts), `src/lower/checks.rs` (program checks read as facts, merged range checks),
`src/lower/simplify.rs` (statements the facts decide), `src/lower/emit.rs` (composition nodes,
stated-fact nodes), `IRBuilder::ir_array_fact` (`src/builder.rs`).

## 1. Terms

- A **cell** is one scalar element of an array: in the IR, the statement whose value it is.
- A **list** is a sequence of values, cells or integer constants (`ListId`, interned per
  lowering).
- A **claim** is what the layer can state and check (`lower::graph::Claim`); a **fact** is a claim
  with a scope (everywhere, or where a Boolean holds).
- A **value's code** is the integer the backends compare: the integer itself, or a float's Q32
  fixed-point code `round(v * 2^32)`. Codes are ordered as the floats, so every order and range
  claim below is a claim about codes.
- A claim is **established** when every witness the circuit accepts satisfies it. Only an
  established claim may justify a construction (the trust rule).

## 2. The claims

| claim | meaning |
|---|---|
| `ElementsWithin { list, lo, hi }` | every value's code in `[lo, hi]`; `None` is an open end. `NonNegativeElements(list)` is `ElementsWithin { list, lo: Some(0), hi: None }` |
| `NonDecreasing(list)` | `list[0] <= list[1] <= ...` |
| `StrictlyIncreasing(list)` | `list[0] < list[1] < ...` |
| `Distinct(list)` | no two values equal |
| `CountBound { list, max_nonzero }` | at most `max_nonzero` values are nonzero |
| `Permutation(list, [0, ..., n-1])` | the values are a permutation of `0..n` |
| `Ordered { x, y, strict }` | `x <= y` (`x < y`): the scalar claim the order claims decompose into |
| `Holds(c)` | the Boolean statement `c` is true: what `Assert(c)` states |

Implication (`array_facts::implies`) is syntactic: a claim about a list implies the same claim
about a part of it (`ElementsWithin`, `Distinct`: any sublist; the orders: any subsequence in list
order), the scalar claims about its values (`ElementsWithin` implies `Within`, `InRange`,
`NonNegative`, `NonZero` of a value; `NonDecreasing` implies `Ordered` of any two values in list
order), and a weaker claim of the same kind. `StrictlyIncreasing` implies `NonDecreasing` and
`Distinct`; `Permutation(p, 0..n)` implies `ElementsWithin { p, 0, n - 1 }` and `Distinct(p)`.

## 3. How a claim is established

1. **By definition** (`array_facts::definitional`): the claim holds whatever the witness, because
   of what the statements are. Constants; the value lanes of one sort at increasing ranks (every
   sort construction yields NumPy's sort); the index lanes of one sort (a stable argsort: all `n`
   of them are a permutation of `0..n`, any of them are distinct); squares, absolute values and
   sums of such terms (non-negative, under a magnitude bound that keeps them from wrapping the
   field). A definitional fact is free, and it may discharge a program check (as a scalar gadget's
   postcondition does).
2. **By bounds**: the range engine's interval of each value (`optim::RangeResolver`) lies in the
   claimed range, or two values' intervals are ordered. Free.
3. **By the program's checks**: an array claim is the conjunction of scalar claims about its
   values (`array_facts::basis`): `NonDecreasing(L)` is `Ordered { L[i], L[i+1] }` for every `i`,
   `ElementsWithin` one `Within` per value and side. The layer adds a *composition node* that
   establishes the array claim from those scalar facts at no cost; each scalar fact is justified
   as usual, typically by the program check that states it (`Assert(a_i <= a_{i+1})` reads as
   `Ordered`, `Assert(x >= 0)` as `NonNegative`, `Assert(x <= 255)` as `Within`). A check that
   justifies a fact a selected construction relies on is kept.
4. **By a construction's postcondition**: `Permutation(p, 0..n)` by the `is_permutation` check
   (`lower::permutation`).
5. **By a fact the IR states** (`IR::ArrayFact`, section 4).

A construction that relies on an array claim calls `Context::established` and offers its
candidate only when the claim is established by one of these routes without a new check, then
lists the claim among its preconditions (so the selection keeps what it rests on). Such claims
are assumptions about the input, not obligations of NumPy's semantics: the layer never emits a
check of one (that would reject inputs the program accepts). Where a claim is NumPy's own
precondition (a sorted table for `np.searchsorted`), the construction lists it without asking,
and the layer checks it when nothing establishes it.

## 4. Stating a fact from the IR generator: `IR::ArrayFact`

```rust
use crate::lower::array_facts::ArrayFactKind;
// cells: the array's cells (row-major); basis: Booleans the claim rests on
b.ir_array_fact(ArrayFactKind::NonDecreasing, &cells, &basis);
```

`IRBuilder::ir_array_fact(kind, cells, basis)` emits one `IR::ArrayFact` statement whose operands
are the cells, then the basis. It has no value and constrains nothing.
`IRBuilder::ir_array_fact_scoped(kind, cells, basis, Some(p))` states a claim that holds where the
Boolean `p` holds (a fact derived under an execution predicate): `p` is the last operand, and the
layer reads the fact with the scope `When(p)`, so it justifies only what holds under `p` too.
`IRBuilder::state_definitional_facts` (`builder/array_facts.rs`) states a definitional segmentation
in the frontend's store and hands what these claims can say of it to the layer (the hull of its
ranges with ends the interval engine fixes from the statements, the order of each segment between
constant cuts, a permutation); the sources of `np.nonzero`, `np.flatnonzero`, `np.argwhere` and
`np.unique` call it.

| `ArrayFactKind` | the claim about `cells` |
|---|---|
| `ElementsWithin { lo, hi }` | every cell's code in `[lo, hi]` (constant ends; `None` open) |
| `NonDecreasing`, `StrictlyIncreasing`, `Distinct` | as in section 2 |
| `CountBound { max_nonzero }` | at most `max_nonzero` cells nonzero |
| `Permutation` | the cells are a permutation of `0..len(cells)` |

**The trust rule for stated facts.** The layer believes a stated fact without checking it, so:

- **With a basis** (non-empty `basis`): the claim must hold on every witness for which every
  Boolean of `basis` is true, and each of those Booleans must be asserted by the program
  (`Assert(c)` emitted unconditionally). The layer establishes the claim from `Holds(c)` for each
  `c` (`emit::add_stated_fact_nodes`), so the asserts it rests on are kept whenever a construction
  uses the claim. Example: `@requires(is_sorted(a))` asserts every `a_i <= a_{i+1}` and states
  `NonDecreasing(a)` on those comparisons (`optim/predicates/witness.rs`).
- **Without a basis** (empty `basis`): the claim must hold on every witness the circuit accepts,
  *by the constraints that compute the cells*, independently of any assertion of the program.
  Use it for operator postconditions: the output of a construction the circuit pins exactly (the
  values of `np.unique` strictly increasing, the positions of `np.flatnonzero` increasing, the
  cells of `np.arange` in order). A fact that in truth rests on an assertion must list the
  assertion's condition in its basis: a definitional fact may discharge the program's checks, and
  one that rested on a check it discharges would justify its own removal.

**What a fact is about.** A fact is about the SSA values it lists, not about an array name or a
memory version. A write to an array produces new cells (or a new version), which the fact does not
mention, so a fact never goes stale in the IR: state it on the cells as they are when the
operator produces them. A fact about cells the program later overwrites simply stops being
relevant. (The frontend's store, `optim::array_facts`, follows the same model: a fact is about
a list of cells or an IR array version, and every query reads the array's current value.) An array whose cells
the IR generator does not know (a memory version read at runtime indices) has no cells to state
a fact about; state the fact on the cells written into it, or on the values read out.

**Passes.** Dead-code elimination keeps an `IR::ArrayFact` exactly while all its operands live,
and it does not keep any operand alive; use counts (`optim::pattern_match_optim`) ignore it. The
lowering layer reads the facts of the graph it receives (`array_facts::stated`) and emits none, so
no backend sees one. Nothing else reads them.

**What the layer does with each kind** (consumers, section 5):

| fact | used by |
|---|---|
| `ElementsWithin` | bounds of the values for the fact-decided statements (`lower::simplify`), the narrow range of a sort's hinted values, `Within`/`InRange`/`NonNegative` preconditions (and index checks) of the values |
| `NonDecreasing` | sort of a sorted input (`sort::SortedInput`), argmax/argmin of a sorted line (`sort::arg_extreme`), ordered pairs for the fact-decided maxima and minima |
| `StrictlyIncreasing` | as `NonDecreasing`; the argmax of a strictly increasing line is `n - 1` |
| `Distinct` | implied by the two above and by `Permutation`; no construction keys on it alone yet |
| `CountBound` | none yet (section 7) |
| `Permutation` | the permutation scatter (`gather::ScatterByHits`), the bounds of the values |

## 5. Constructions and rules keyed on array facts

| rule | fact | operator | code |
|---|---|---|---|
| Range realization (merged range check) | the checks bounding a value on both sides | the value's checks | `checks::range_merges` |
| Narrow comparisons and selects | a value's established range | comparisons, selects, sort networks (through the backends' widths) | `checks::range_merges`, `prove/halo2_base/synthesizer/ranges.rs` (`prove_range` across offsets), `builder.rs` (provenance across offsets), `prove/noir/emit.rs` (`settle_range_widths`) |
| Narrow hinted sort values | `ElementsWithin(elements)` | every hinted sort plan | `sort::narrow_range`, `sort::emit::Elements::bound` |
| Fact-decided comparisons, maxima and minima (clip/where identities) | ranges and orders of the operands | `<`, `<=`, `>`, `>=`, `select(a > b, a, b)` | `lower::simplify` |
| Sort of a sorted input | `NonDecreasing(elements)` | `np.sort`, `np.argsort`, `np.max`, `np.min`, `np.median`, `np.percentile` | `sort::sorted_input` |
| Argmax/argmin of a sorted line | `NonDecreasing`, `StrictlyIncreasing` | `np.argmax`, `np.argmin` | `sort::arg_extreme` |
| Permutation implies its range | `Permutation(p, 0..n)` | index and window checks of `p`'s values | `array_facts::implies` |
| Non-negative by definition | squares, absolute values, sums of them | domain checks (`x >= 0` before `np.sqrt`, halo2-base) | `array_facts::non_negative_by_definition`, `checks::kept_for_backend` |

Each has its soundness argument in its doc comment; `eval-redo/layer/fact-rules.md` has the
measurements.

## 6. Measurement switch

`ZINNIA_ABLATE_ARRAY_FACTS=1` turns every array claim off: no claim is established, implied or
composed; program checks are read as before array claims existed (no `Ordered`, `Holds`, float
or strict-bound forms); no merge, no fact-decided statement, no narrow hint. The constructions
keyed on array claims do not apply. On the 179 perf entries the compiled circuits are then those
of the compiler without array claims (`docs/ablations.md`; the two programs with `np.argmax` of a
numeric line differ only in where constants are placed, at equal cost). The measurement scripts
call this configuration `abl_array_facts` (`benchmarking/ablations/common.py`).

## 7. Not supported yet

- Symbolic bounds (`0 <= inv[i] <= count - 1` for `np.unique`'s inverse): `ElementsWithin` has
  constant ends only. A bound by a statement would need `InRange { hi: Bound::Stmt }`-style ends
  in the array claim.
- Segments: a claim covers a whole list; a fact about a prefix (the live part of a compaction,
  "zero past position `k`") is stated on the sublist of its cells when they are known.
- `CountBound` and `Distinct` have no construction of their own (an output capacity sized by
  `nnz` would be the IR generator's decision, before the layer).

## 8. From the frontend's fact store (section by branch `array-facts-port`)

*Written by the array-facts-port agent; not part of the fact-rules commits.*

The frontend's store (branch `array-facts-port`, `docs/array-facts.md` there) has the same
model as `IR::ArrayFact`: a fact describes the cells it lists (SSA values) or an IR array
version, and every query reads the array's current cells, so nothing expires on a write.
(The sentence of section 4 that the store "keys them by array" describes fact-staleness's
value-id records, which array-facts-port replaced.) The interface fits, with three limits:

1. **Scope.** Each store fact carries a guard: `Always`, or the conjuncts of the execution
   predicate it was derived under (a range derived inside `if c:` from a check made there).
   `IR::ArrayFact` has no scope, so the store can state only its `Always` facts. An optional
   predicate operand (the layer's `Scope::When(p)`) would let it state the others.
2. **Basis.** The store does not record which program checks a derived fact rests on (a range
   transferred through `a + 1` from a `forall(a, in_range(...))`). It can state with an empty
   basis only the facts that hold by the constraints computing the cells, and with a basis the
   `@requires` element facts (the Booleans the witness emitter asserts).
3. **Vocabulary.** Segmentations with a runtime cut (`{0} live {count} padding {n}`) and
   symbolic ends (`<= count - 1`) have no claim; the store can state the constant hull of a
   whole list, or a prefix cut at a constant.

What the store would state through `ir_array_fact`, once both branches are in one tree (a call
next to each source's `state_defined_facts`, `builder/array_facts.rs`):

| source | `ArrayFactKind`, cells, basis |
|---|---|
| `np.nonzero` / `np.flatnonzero` lane, `np.argwhere` | `ElementsWithin { 0, extent - 1 }` on the lane's cells (padding 0 included), no basis |
| `np.unique` inverse | `ElementsWithin { 0, n - 1 }` (the constant hull of `[0, count - 1]`), no basis |
| `np.unique` values | none (the order holds on the live prefix only, cut at the runtime count) |
| sort / argsort lines | none (the layer's `definitional` has them; it replaced `FreeFacts::is_argsort_line` at the merge) |
| `@requires(forall(a, in_range(lo, hi)))` (constant ends), `lt_bound`, `is_nonneg`, `eq_const` | `ElementsWithin` on the input's cells, basis: the asserted comparisons |
| Boolean input arrays | `ElementsWithin { 0, 1 }`, basis: the domain checks |
| `np.linspace` with a runtime `num`, constant ascending endpoints | none (ordered on the live prefix only) |

Nothing measured needs these yet (the perf cases' facts are all established by the layer's own
routes), so array-facts-port does not emit any; the calls are a few lines each.

*Follow-up in fact-rules, after the merge of `array-facts-port` (64b56f03):* limit 1 is lifted
(`ir_array_fact_scoped` states a fact under a predicate); the sentence of section 4 now describes
the store as above; and `IRBuilder::state_definitional_facts` hands the definitional facts of the
table's `np.nonzero` / `np.flatnonzero` / `np.argwhere` and `np.unique` rows to the layer. The
`@requires` element facts reach the layer from the witness emitters (`optim/predicates/witness.rs`),
with their asserted comparisons as the basis; the sort and argsort lines from the layer's own
`definitional`, which replaced `FreeFacts::is_argsort_line`. One switch,
`ZINNIA_ABLATE_ARRAY_FACTS=1`, turns off both the store's queries and the layer's claims.
