# North Star Design: Dynamic NDArrays with Cross-Axis Bound Analysis

## 0. Framing

This document is the target design, not a migration plan. It describes what the compiler *should* know and *should* refuse, assuming the static analysis is as strong as it can reasonably be. Gaps between this target and the current implementation are bugs against the target, not features of the target.

The core commitment: **the compiler must know a finite element count for every dynamic ndarray at every point in the program, or it must refuse to compile.** No silent fallbacks to per-axis products. No materialisation of `max_len_a × max_len_b` broadcasts. Programs that cannot be bound-analysed are programs the programmer must annotate or rewrite.

This is strictly more restrictive than the current design, and that's the point. The current design's failure mode is silent circuit explosion; the target's failure mode is a compile error pointing at the line that needs an annotation. The second is strictly better: explosions become visible at the time and place they can be fixed.

## 1. Design Principles

**P1. Bounds are a type, not a hint.** Every dynamic ndarray carries a statically-known upper bound on its total element count. This bound is part of the type and participates in type checking. Operations that cannot produce a finite bound from their inputs do not type-check.

**P2. The compiler extracts everything it can.** Before asking the programmer to annotate, the compiler exhausts what can be derived from: symbolic dim unification, arithmetic on dim variables, linearity analysis of operations, and propagation through op-specific rules. Annotation is the escape hatch for what's genuinely beyond static analysis, not a crutch for a lazy analyser.

**P3. Refusal is a first-class outcome.** When the compiler cannot prove a finite bound and no annotation is supplied, it emits a diagnostic naming the specific operation, the specific axes, and the specific reason the bound is not derivable. The programmer's response is either an annotation, a rewrite, or accepting that the program is not expressible in this system.

**P4. Annotations are checked, not trusted.** A programmer-supplied `#[bound(...)]` becomes a runtime assertion at the point where the bound must hold. If the assertion fails, proof generation fails. The compiler trusts the annotation for static analysis but verifies it dynamically.

**P5. The per-axis product is never the answer.** If the analysis concludes the bound is `max_len_a × max_len_b` for independent dims, the analysis has failed. The compiler reports the failure; it does not emit a circuit of that size.

## 2. The Type System for Dynamic NDArrays

### 2.1 Static vs dynamic data: the split

Every dynamic ndarray value has two layers of data, with a strict rule about which is known when.

**Compile-time (static) data — part of the type:**

- **Rank** `r ∈ ℕ`. The number of axes. Always static. An array whose rank is not statically known is not a dynamic ndarray in this system; it's something else (probably an error).
- **Element type** `τ`. Always static.
- **Per-axis symbolic bounds** `[D₁, D₂, …, D_r]` where each `Dᵢ` is either a concrete integer or a symbolic dim variable. Dim variables are drawn from a program-wide pool and can be unified across arrays via union-find.
- **Per-axis maxima** `[M₁, M₂, …, M_r]` with `Mᵢ ∈ ℕ ∪ {⊤}`. `Mᵢ` is a static upper bound on the runtime value of `Dᵢ`. `⊤` means "no per-axis bound known." Concrete integer dims have `Mᵢ` equal to themselves.
- **Total bound** `T ∈ ℕ ∪ {⊤}`. A static upper bound on the product of runtime dim values — that is, on the total element count. This is the critical field. **If `T = ⊤`, the value cannot be materialised and most operations on it are compile errors.**
- **Bound provenance** `prov`. A record of how `T` was derived: inference from a specific rule, propagation from a source, programmer annotation, or circuit input declaration. Used for diagnostics when `T` is `⊤` and for explaining bounds to the programmer.
- **Storage kind** `Segment | View`. Whether the array is backed by a ZKRAM segment or a lazy view tree. This is type-level because it affects which operations are legal (e.g., random write requires `Segment`).

**Runtime (dynamic) data — part of the value, not the type:**

- **Per-axis runtime sizes** `[d₁, d₂, …, d_r]` with `dᵢ ≤ Mᵢ` enforced by circuit constraints at every point the value is constructed or modified.
- **The element data itself**, stored either in a ZKRAM segment of capacity `T` (if storage is `Segment`) or implicit in the view tree (if storage is `View`).
- **A runtime total** `d₁ · d₂ · … · d_r`, computable on demand via IR multiplies, constrained to be `≤ T`.

### 2.2 The invariant

For every dynamic ndarray value at every point in the program:

> `T ≠ ⊤`, and at runtime the product of the per-axis sizes is `≤ T`.

The first clause is a static type-checking obligation. The second is a dynamic circuit-constraint obligation. A program violates the type system if any value has `T = ⊤`; a proof fails if any runtime product exceeds `T`.

### 2.3 Why `T` and not just the `Mᵢ`

The per-axis maxima `Mᵢ` are a *weaker* form of the same information. Their product `∏ Mᵢ` is always a valid value for `T`, but often much larger than the true bound. The whole point of the type system is to carry `T` as an independent, tighter fact whenever the compiler can derive one, and to refuse the program when even the weaker `∏ Mᵢ` is the only option *and* that product would be the forbidden multiplicative explosion.

The rule for when `∏ Mᵢ` is acceptable as `T`: only when the per-axis maxima are small enough that the product is not an "explosion" in the programmer's sense. The threshold is a compiler parameter (e.g., "refuse if the product exceeds 4× the largest input's `T`"), not a hard number, and it's tunable per project.

### 2.4 Dim variables and unification

Dim variables are the mechanism for expressing that two dims are *the same runtime value*, not merely that they have the same upper bound. When the compiler can prove two dims unify, the bound analysis gains a cross-axis constraint that would otherwise be invisible.

Sources of unification:

- **Explicit**: a function signature naming the same dim variable in two positions.
- **Structural**: an operation that produces outputs whose shape is structurally tied to an input (e.g., `map` preserves shape, so output dims unify with input dims).
- **Constraint-derived**: an operation like `broadcast` or `matmul` whose validity requires certain dims to match, forcing unification at the call site.
- **Annotation**: the programmer supplies a `where D1 = D2` clause.

Unification is the primary lever for keeping `T` finite across operations. Two arrays with the *same* dim variable can be broadcast/added/multiplied without any product explosion, because the compiler knows the dims are equal at runtime, not independently ranging up to their maxima.

## 3. Bound Propagation Rules

The compiler derives `T` for operation outputs from inputs. The rules are exhaustive for the operations the system supports. Any operation not covered here produces `T = ⊤` and is therefore a compile error without annotation.

### 3.1 Leaves

- **Circuit input** with declared `max_length = N`: `T = N`, rank and per-axis bounds from the declaration. Provenance: `Declared`.
- **Constant array** with shape `[n₁, …, n_r]`: `T = ∏ nᵢ`. Provenance: `Constant`.

### 3.2 Shape-preserving operations

For operations that don't change the element count (`map`, element-wise unary, `transpose`, `reshape` to a compatible shape, `negate`, scalar ops):

- `T_out = T_in`, `prov = PreservedFrom(in)`.

Reshape requires that the new shape's per-axis maxima multiply to something `≤ T_in`; otherwise the reshape is rejected.

### 3.3 Same-shape binary operations (dim vars unified)

When `a` and `b` have the same rank and every corresponding dim unifies (`a.Dᵢ = b.Dᵢ` for all `i`):

- `T_out = min(T_a, T_b)`, `prov = SameShape(a, b)`.

The `min` is valid because at runtime both arrays have identical shapes, so the output has the same element count as either input, and the tighter bound wins.

### 3.4 Broadcast binary operations

This is the interesting case and where refusal lives. Broadcast combines two arrays by aligning their shapes with implicit size-1 dims and pairing elements. The output's rank is `max(r_a, r_b)`, and each output axis comes from `a`, from `b`, or from both.

Classify each output axis:

- **Shared axis**: both `a` and `b` have a non-trivial dim here, and those dims unify. Output dim unifies with both.
- **`a`-only axis**: `a` has a non-trivial dim here, `b` has size-1 (explicit or implicit). Output dim unifies with `a`'s.
- **`b`-only axis**: symmetric.
- **Independent axis**: both `a` and `b` have non-trivial dims here, but they do *not* unify.

The rule for `T_out`:

- **If there are no independent axes and no disjoint dynamic broadcast**:
  `T_out` is derivable. Let `A_only` be the product of per-axis maxima for
  `a`-only axes, and `B_only` symmetrically. Then
  `T_out = min(T_a · B_only_max, T_b · A_only_max)` where `A_only_max`
  and `B_only_max` are computed from the per-axis maxima on the respective
  sides. Provenance: `BroadcastNoIndep(a, b)`.

  Intuition: every element of `a` appears in the output `B_only_max` times
  (once per position along `b`-only axes), so the output element count is
  at most `T_a · B_only_max`. The symmetric bound via `b` also holds, and
  the min is tighter.

- **If there are any independent axes**: `T_out = ⊤`. **This is the first
  refusal case.** The compiler emits a diagnostic naming the axes, the dim
  variables involved, and the suggestion to either unify the dims (via
  annotation or rewrite) or supply a `#[bound(total ≤ N)]` annotation at
  this operation.

- **If the broadcast is disjoint** (both `a`-only and `b`-only axes exist,
  and any of the `a`-only or `b`-only dims are dynamic): `T_out = ⊤`.
  **This is the second refusal case.** The `[D1, 1] ⊕ [1, D2]` pattern
  produces `T = D1_max · D2_max` under the formula above — exactly the
  multiplicative explosion P5 forbids. The explosion comes from
  disjoint-axis broadcast (outer-product shape) rather than same-axis
  independent dims, but the effect is the same. The compiler refuses and
  demands annotation.

  Exception: if all `a`-only and `b`-only dims are statically known, the
  broadcast factor is a compile-time constant and the multiplication is
  predictable. In this case `T_out` is derivable from the formula above
  (e.g., `[D, 1] + [1, 3]` → `T_out = 3 · T_a`).

This rule is the heart of the system. It says: a broadcast between
genuinely-independent dynamic dims is not allowed to silently produce a
`max_a × max_b` result. The programmer must either prove the dims are
related, or declare the bound explicitly, or the program doesn't compile.

**Interaction with views (§11.6):** When the result of a refused broadcast
is stored as a view (not materialised), the refusal may be deferred to the
materialisation site. See §11.6 for the two-level bound refinement.

### 3.5 Reductions

For `reduce` over a set of axes `S`, with input `T_in` and per-axis maxima `[M₁, …, M_r]`:

- `T_out = T_in / ∏_{i ∈ S} min_axis(i)` where `min_axis(i)` is the statically-known *minimum* runtime size of axis `i` (often 1, sometimes higher if the dim is bounded below).
- In the common case with no lower bound, `T_out = T_in` (reduction can reduce as little as one element per output position, leaving the count unchanged in the worst case).
- Provenance: `ReductionFrom(in)`.

This is conservative but sound. Reductions that can be algebraically decomposed (§5) get tighter bounds via the decomposition rules, not via the generic reduction rule.

### 3.6 Matmul

For `matmul(a: [..., M, K], b: [..., K, N])`:

- The `K` dims must unify (structural unification at the call site; fail otherwise).
- Batch dims follow broadcast rules (§3.4), including the refusal case for independent batch dims.
- `T_out = T_batch · M_max · N_max` where `T_batch` comes from the broadcast of the batch dims.
- Provenance: `Matmul(a, b)`.

### 3.7 Concatenate

For `concat` along axis `i` of arrays `a₁, …, a_n`, all with the same rank and all non-`i` dims unifying:

- `T_out = ∑ T_aⱼ`.
- Provenance: `Concat(a₁, …, a_n)`.

### 3.8 Filter / boolean mask

For `filter(a, mask)` where `mask` is a boolean array unifying with `a`:

- `T_out = T_a` (worst case: every element passes).
- Storage is forced to `Segment` (random-write compaction).
- Provenance: `FilterFrom(a)`.

### 3.9 Gather / scatter

For `gather(source, indices)` where `indices` is a dynamic array of index tuples:

- `T_out = T_indices`.
- Provenance: `GatherBy(indices)`.

Scatter is symmetric but forces `Segment` storage on the output.

### 3.10 User-supplied annotation

A `#[bound(total ≤ N)]` annotation on an expression overrides whatever bound the rules would derive, setting `T_out = N` with `prov = Annotated(site)`. The compiler emits a runtime assertion at that site checking the actual element count against `N`, which causes proof failure if violated.

Annotations are only legal at sites where the rules would otherwise produce `T = ⊤` or where the derived bound is strictly larger than `N`. Annotating a site with a *looser* bound than what the compiler derived is either a warning (likely wrong) or a no-op (honoured but the compiler uses the tighter derived bound internally).

## 4. The Refusal Path: What a Good Diagnostic Looks Like

When the compiler hits `T = ⊤`, the diagnostic must contain enough information for the programmer to act. At minimum:

- **Location**: the specific operation site.
- **Operation**: the op that failed to produce a finite bound.
- **Reason**: which axes are independent and why they don't unify. Name the dim variables.
- **Inputs' bounds**: `T_a`, `T_b`, and the per-axis maxima involved.
- **The forbidden product**: what the per-axis-product fallback *would* have been, so the programmer sees the scale of the explosion they're being protected from.
- **Fix suggestions**, in priority order:
  1. If the dims are "morally" the same (e.g., both derived from the same input length), unify them via annotation: `where D1 = D2`.
  2. If the programmer knows a cross-axis bound, supply `#[bound(total ≤ N)]` at the operation.
  3. If neither applies, rewrite to avoid the independent broadcast.

The diagnostic is part of the design, not an implementation detail. The system's user-facing story is "the compiler tells you exactly where and why," and that story is only credible if the diagnostics are genuinely informative.

## 5. Algebraic Decomposition (Bound-Preserving Optimisation)

The rules in §3 produce *sound* bounds. Some operation patterns admit *tighter* bounds via algebraic rewrites that the compiler can apply before bound analysis runs.

The canonical example: `sum(a + b)` where `a` and `b` broadcast on disjoint axes. Under §3.4 with unified non-broadcast axes, the broadcast itself has a finite `T_out`, so this case doesn't need decomposition for the bound — decomposition is an *execution-cost* optimisation, not a bound-analysis optimisation.

But decomposition interacts with bounds in one important way: when decomposition eliminates a broadcast entirely, the intermediate bound calculation is replaced by the bounds of the source operands, which may be tighter than the broadcast's `T_out`. The compiler should apply decomposition before final bound assignment so the tightest bound wins.

Decomposition rules, applied under the disjoint-axis precondition (axis-maps of the two sides partition the output axes):

```
sum(a ⊕ b), ⊕ ∈ {+, -}   → bcast_b · sum(a) ± bcast_a · sum(b)
sum(a · b), disjoint      → sum(a) · sum(b)
max(a + b), disjoint      → max(a) + max(b)
min(a + b), disjoint      → min(a) + min(b)
```

Outside the disjoint-axis precondition, these rules are unsound and must not fire. The precondition is a structural check on the axis-map metadata and is either true or false; no heuristic.

Unsupported combinations (e.g., `prod(a + b)`, `argmax(a · b)`) get no decomposition and fall through to the general §3.4 rules. If those rules produce `T = ⊤`, the operation is refused.

## 6. Storage: Segments and Views

Storage kind is part of the type (§2.1) and is determined by the operation producing the value.

**Segment storage**: backed by a ZKRAM segment of capacity `T`. Used for:
- Leaves (circuit inputs, constants).
- Results of operations requiring random write (filter, scatter).
- Results that cross circuit boundaries.
- Any value used more than once (the multi-use hard invariant).
- Any value whose consumer requires eager iteration (e.g., operations not implemented for views).

**View storage**: lazy expression tree over source arrays. Used for:
- Single-use intermediate results of supported operations (broadcast, element-wise binary).
- Allocated whenever the compiler can prove single-use and the op is view-compatible.

The view ↔ segment transition is driven by type rules, not runtime decisions. The compiler decides storage kind during type checking based on use-count analysis and op compatibility. Materialisation (view → segment) is inserted as an explicit IR node wherever the type rules require it.

Crucially, **storage kind does not affect `T`.** A view and a segment with the same logical shape have the same bound. Storage is about *when* the element count is paid in circuit cost, not about *what* the bound is. This is what makes §5's "bound is tighter than per-axis product" story work: even a materialised segment gets the tight bound, because `T` is a type-level fact that survives the view → segment transition.

## 7. Use-Count Analysis

The multi-use hard invariant (§6) requires the compiler to know, for every view-typed value, how many times it is consumed. This is a standard use-count pass over the IR, with one subtlety: inlining, function-call boundaries, and loop-carried values can all duplicate uses in ways that aren't apparent from the surface syntax.

The rule: use-count is computed *after* all inlining and specialisation passes, on the final IR the backend sees. A value with use-count ≥ 2 is materialised to a segment before its first use. Use-count 1 values remain as views if their op supports view storage.

There is no "heuristic" version of this. Either the use-count is 1 and views are safe, or it's ≥ 2 and materialisation is mandatory. The cost of being wrong (exponential expression-tree duplication) is too high to leave to heuristics.

## 8. Putting It Together: The Programmer's Experience

A programmer writing against this system sees:

1. **Explicit ranks and dim variables in signatures.** Function signatures that take dynamic ndarrays name the dim variables, so callers know which dims unify and which are independent.

2. **Inferred bounds in most cases.** The compiler derives `T` for the common cases (same-shape ops, broadcasts with unified dims, reductions, concatenations) without any annotation. Most code never mentions `#[bound(...)]`.

3. **Loud refusals at the boundary of what's inferable.** When the code hits a genuinely-independent broadcast, the compiler stops and explains. The programmer's choices are explicit, documented, and local to the site of the problem.

4. **Annotations where the programmer knows more than the compiler.** When domain knowledge provides a cross-axis bound the compiler can't derive (sparsity, problem-specific invariants, input-shape relationships), the programmer supplies `#[bound(...)]` and accepts the runtime assertion.

5. **No silent cost explosions.** The circuit size is always a function of bounds that are either derived or annotated. There is no path from "I wrote a reasonable-looking expression" to "the circuit is 10,000× larger than I expected." Every large bound in the circuit corresponds to either a declared input size, a derived bound the programmer can inspect via `prov`, or an annotation the programmer wrote.

## 9. What This Design Does Not Solve

Honest limitations of the North Star itself, not of any particular implementation:

- **Genuinely unbounded programs.** If a program's element count truly depends on adversarial runtime data with no static bound, the programmer must annotate with a worst-case they're willing to pay for, or the program doesn't compile. There is no mechanism for "decide the bound at proving time" — the circuit is static and its size is a compile-time fact.

- **Bound inference is not complete.** The rules in §3 are exhaustive for the ops listed, but real systems will have ops whose bound rules are hard to state (custom user ops, FFI to external chips, complex indexing patterns). Those ops either get conservative rules (often `T = ⊤`, forcing annotation) or bespoke rules added as the system matures. The target is that bespoke rules cover the standard library; user code falls back to annotation.

- **Annotations are trusted for analysis, checked at runtime.** A wrong annotation doesn't corrupt the analysis — it causes proof failure at runtime. That's a safe failure mode but it means annotation bugs surface late. A secondary static check that tries to validate annotations against what the compiler *can* derive (even if not used) would catch some cases earlier; worth considering but not essential.

- **Cross-function bound propagation requires bound-polymorphic signatures.** For bounds to flow through function calls, function signatures need to express "this argument has some bound `T`, and the return has bound `f(T)`" for some compile-time function `f`. This is bound-polymorphism and it's a nontrivial type-system feature. The target assumes it exists; a simpler implementation might require annotations at function boundaries until it's built.

- **The per-axis product is still the fallback *for annotations*.** A programmer who annotates `#[bound(total ≤ max_a * max_b)]` is allowed to do so, and the compiler will obediently allocate that much. The system protects against silent explosions, not against programmers who explicitly request explosions. This is correct; the design goal is visibility, not prevention of all large circuits.

## 10. Summary of the Type System

| Field | Static? | Description |
|---|---|---|
| Rank `r` | Yes | Number of axes |
| Element type `τ` | Yes | Scalar type of elements |
| Per-axis symbolic dims `[D₁…D_r]` | Yes | Either concrete ints or unifiable dim variables |
| Per-axis maxima `[M₁…M_r]` | Yes | Static upper bound on each runtime dim, or `⊤` |
| Total bound `T` | Yes | Static upper bound on element count. `⊤` is a compile error. |
| Bound provenance `prov` | Yes | How `T` was derived; used in diagnostics |
| Storage kind | Yes | `Segment` or `View`; determined by use-count and op compatibility |
| Runtime dims `[d₁…d_r]` | **No** | Actual per-axis sizes at runtime, constrained `dᵢ ≤ Mᵢ` |
| Runtime total `∏ dᵢ` | **No** | Constrained `≤ T` at every construction site |
| Element data | **No** | In a segment of capacity `T`, or implicit in a view tree |

The invariant: `T` is always finite (compile-time), and the runtime element count is always `≤ T` (circuit-enforced). Any program that would violate the first clause is refused; any proof that would violate the second clause fails to generate.

---

The North Star in one sentence: **every dynamic ndarray has a statically-known, finite element count, derived by the strongest analysis the compiler can do, and backed by a loud refusal when analysis is insufficient.** Everything else — views, decomposition, ZKRAM layout, runtime strides — is implementation detail that serves this invariant.

---

## 11. Implementation Refinements

The following refinements were identified during review against the current
Zinnia codebase and are amendments to the design above.

### 11.1 Full reduction: T_out = 1

§3.5's formula `T_out = T_in / ∏ min_axis(i)` is sound but too
conservative for **full reductions** (reduce over all axes, no axis
argument). A full reduction always produces a scalar — one element — so
`T_out = 1` unconditionally. The generic formula gives `T_out = T_in` when
all `min_axis(i) = 1`, carrying the input bound forward onto a scalar
result.

Amended rule for §3.5:

- **Full reduction** (S = all axes): `T_out = 1`. Provenance: `FullReduction(in)`.
- **Axis-specific reduction**: `T_out = T_in / ∏_{i ∈ S} min_axis(i)` (unchanged).

### 11.2 Broadcast factor: static vs dynamic, not threshold

§2.3 proposes a tunable threshold ("refuse if product exceeds 4× the
largest input's T") to catch a-only/b-only explosions. This creates
discontinuities: 399 compiles, 401 doesn't. Replace with a structural
rule:

**A broadcast is allowed without annotation when all broadcast factors are
statically known.** A broadcast factor is the product of the other
operand's dims at positions where this operand has size 1. If any
broadcast factor involves a dynamic dim, the compiler refuses and
requires annotation.

| Case | Broadcast factor | Allowed? |
|---|---|---|
| `[D, 1] + [1, 3]` | Static(3) | Yes, `T_out = 3 · T_a` |
| `[D, 1] + [1, D₂]` | Dynamic(D₂) | Refuse — require `#[bound]` |
| `[D₁, D₂] + [D₁, 1]` | Static(1) | Yes, `T_out = T_a` |
| `[D₁, D₂] + scalar` | None | Yes, `T_out = T_a` |
| `[D₁] + [D₂]` (same axis) | N/A — same-axis unification | See §11.3 |

This is purely structural: the compiler checks whether the a-only and
b-only dims have `is_static() == Some(_)`. No tunable parameter.

### 11.3 Independent axes DO arise — silent unification is the problem

The original review incorrectly claimed that independent axes "may never
arise naturally" because `broadcast_envelopes` always unifies same-position
dims. This is wrong. Consider:

```python
@zk_circuit
def f(a: DynamicNDArray[int, max_length=100],
      b: DynamicNDArray[int, max_length=100]):
    c = a + b   # a.D₁ vs b.D₃ — genuinely independent dim vars
```

`a` and `b` come from independent circuit inputs. Their dims are drawn
from independent sources — there is no structural reason they should be
equal. The current `broadcast_envelopes` silently auto-unifies D₁ ≡ D₃,
inserting an implicit runtime assertion that the dims are equal. If the
programmer didn't intend this, the proof fails at runtime with no
compile-time warning about the hidden constraint.

When dims are data-dependent (e.g., both derived from filter operations on
different data), they are genuinely independent. Broadcasting them requires
the compiler to either:

1. **Detect prior unification**: if D₁ and D₃ are already in the same
   equivalence class (unified earlier in the program), proceed — no new
   constraint.
2. **Refuse and require explicit annotation**: if D₁ and D₃ are in
   different equivalence classes, the compiler does not silently unify.
   Instead it emits a diagnostic: "dims D₁ and D₃ are independent; to
   broadcast, assert they are equal with `where D₁ = D₃` or supply
   `#[bound(total ≤ N)]`."

This replaces the current behaviour (auto-unify, hide the constraint) with
a loud refusal, consistent with P3 and P5. The programmer's explicit
`where D₁ = D₃` annotation becomes a runtime assertion (circuit
constraint) at the broadcast site.

**Why this matters for bounds**: auto-unification makes `T_out =
min(T_a, T_b)`, which looks tight — but the tightness is an artefact of a
hidden assertion. If the assertion is wrong (dims aren't actually equal at
runtime), the proof fails. Making the unification explicit does not change
the bound arithmetic; it changes *who is responsible* for the equality
claim. The compiler refuses responsibility; the programmer accepts it via
annotation.

### 11.4 Bound analysis and cost optimisation are separate concerns

§6 correctly states that storage kind does not affect `T`. This has an
important implication that should be stated explicitly:

**Views do not mitigate bound explosion. They mitigate cost explosion.**

A `BinaryView` with logical `max_total = 10,000` still has `T = 10,000`.
The view avoids allocating a ZKRAM segment and defers element computation
to the consumer, but the *bound* is unchanged. If `T = 10,000` violates
the programmer's expectations, the answer is tighter bound analysis or
annotation — not views.

Views help because:
- They avoid intermediate ZKRAM segments (fewer memory slots).
- They allow consumers with tighter bounds to iterate fewer times (the
  consumer's `T`, not the view's, determines iteration count).
- They enable algebraic decomposition (§5) by preserving expression
  structure.

But none of these change `T`. The bound system (§2–§4) and the cost system
(§5–§7) are orthogonal. A program that passes bound analysis is guaranteed
not to explode in circuit size. A program that also uses views is
guaranteed to have lower circuit cost where applicable.

### 11.5 Phased implementation plan

The design in §1–§10 is the target. Implementation proceeds in phases,
each of which is self-contained and testable:

**Phase A — total_bound on Envelope (foundation):**
- Add `total_bound: usize` to `Envelope`.
- Propagate through all existing ops (shape-preserving, filter, concat,
  constructors, promotion).
- Circuit inputs set `total_bound = max_length`.
- `Envelope::max_total()` returns `min(∏ dim.max, total_bound)`.

**Phase B — element-wise binary ops with bound rules:**
- Same-shape binary ops: `T_out = min(T_a, T_b)` (§3.3).
- Broadcast binary ops with static-only broadcast factors: derive `T_out`
  per §3.4.
- Broadcast with dynamic factors or independent axes: compile error
  (annotations not yet implemented).

**Phase C — DynStorage and views:**
- `DynStorage` enum: `Segment(u32) | BinaryView { op, lhs, rhs }`.
- `read_element` for lazy resolution through view trees.
- `materialise()` escape hatch.
- Conservative materialisation: materialise at variable binding.

**Phase D — annotations and diagnostics:**
- `#[bound(total ≤ N)]` syntax in the Python frontend.
- Runtime assertion emission at annotation sites.
- Diagnostic infrastructure for refusal messages (§4).
- `where D₁ = D₃` syntax for explicit dim unification.

**Phase E — algebraic decomposition:**
- Detect decomposable patterns (`sum(BinaryView("add", ...))`, etc.).
- Emit decomposed computation using broadcast factors from runtime
  metadata.
- Provenance: `Decomposed(pattern, sources)`.

**Phase F — use-count analysis:**
- Use-count pass on final IR.
- Automatic view ↔ segment decisions based on use count.
- Remove conservative materialisation heuristic from Phase C.

### 11.6 Two-level bounds for views (path-sensitive bound checking)

§3.4 (as amended above) refuses disjoint dynamic broadcasts at the
construction site. This is correct for eagerly-materialised arrays. But
when the result is stored as a view, the refusal can be too conservative:
the view may never be materialised at the explosive size if the consumer
has a tighter bound.

**The refinement**: split `T` into two levels for view-typed values:

- `T_logical`: the logical element count of the view's shape. May be
  large or `⊤` for disjoint broadcasts. This is *not* an error by
  itself — a view with `T_logical = ⊤` is fine as long as it is never
  materialised without a tighter bound.
- `T_materialised`: the bound that applies when the view is forced to a
  segment. Must be finite and non-exploding at every materialisation
  point. Derived from the *consumer's* context, not the view's logical
  shape.

**Construction-site rule (amended)**: creating a view with `T_logical = ⊤`
is allowed. Creating a segment (materialisation) with `T = ⊤` is refused.

**Materialisation-site rule**: at every point the type system forces
materialisation (boundary crossing, multi-use, filter, consumer that
doesn't support views), the compiler checks that a finite `T_materialised`
is derivable. If not, it refuses with a diagnostic pointing at the
materialisation site, not the construction site.

**How `T_materialised` is derived at consumption**:

- **Same-shape binary op** (`view * z` where z has `T_z`): dims unify,
  `T_materialised = min(T_logical, T_z) = T_z`. The consumer's bound
  wins.
- **Decomposable reduction** (`sum(view)`): the decomposition rewrites
  the expression to use source arrays directly. `T_materialised` is never
  needed because no segment is allocated.
- **Boundary crossing** (`return view`): no consumer context available.
  `T_materialised = T_logical = ⊤` → refuse.
- **Multi-use**: materialisation forced before first use. `T_materialised`
  must be derived from the use sites. If any use site cannot provide a
  bound, refuse.

**Concrete examples**:

Example A — disjoint broadcast consumed by same-shape op:
```
v      : [D1, 1] + [1, D2]   # view, T_logical = D1_max · D2_max
z      : [D1, D2]            # segment, T_z = 100
result : v * z               # dims unify, T_out = 100. Accepted.
```
Under construction-site rules alone, `v` is refused. Under the two-level
rule, `v` is a view with large `T_logical`, and the consumption by `z`
provides `T_materialised = 100`. Program accepted.

Example B — disjoint broadcast escaping to boundary:
```
v      : [D1, 1] + [1, D2]   # view, T_logical = ⊤
return v                     # materialisation forced, T = ⊤ → refuse
```
Same outcome as construction-site refusal, but the error points at
`return v` instead of `v = ...`, which is more informative: it tells the
programmer *where* the explosive view needs a bound.

Example C — multi-use of a disjoint view:
```
v : [D1, 1] + [1, D2]   # view, T_logical = ⊤
a : sum_axis_0(v)        # decomposable → accepted
b : sum_axis_1(v)        # decomposable → accepted
```
Multi-use forces materialisation under §7's hard invariant. But both
consumers are independently decomposable. A more sophisticated rule:
allow multi-use views when *every* consumer's bound derivation succeeds
independently, and the total work is bounded. This is equivalent to
duplicating the view tree per consumer in the IR. Whether this
refinement is worth the implementation complexity depends on how common
the pattern is in real workloads.

**Phasing**: the two-level bound is a Phase C+ feature. Phase B uses the
simpler construction-site rule (§11.2). Phase C introduces views with
`T_logical` but initially materialises conservatively. The full
path-sensitive bound checking (deferred refusal) lands when the view
infrastructure is stable and real programs provide evidence for the
coverage gain.

**Tradeoff**: the two-level bound makes the type system harder to reason
about locally — you can no longer look at a single operation and say
"accepted or refused." You must trace the value to its materialisation
point. This is a real cost in cognitive load and diagnostic complexity.
The cost is justified if the "disjoint construction, tight same-shape
consumer" pattern is common. Survey real programs before committing.