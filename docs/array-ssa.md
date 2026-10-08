# array-ssa: arrays as SSA values, their representation a layer choice

Agent "array-ssa", worktree `wt/array-ssa`, branch `array-ssa` (from `eval-refactor` 85625961).
Base for every "no case costlier" comparison: `wt/array-ssa-base` (detached at 85625961, same
build), measured in `layer/array-ssa-data/costs_base.json`. Data and scripts:
`layer/array-ssa-data/` (`census.py`: memory ops in the final IR; `measure.py`: halo2-base
cells, Noir gates and IR digests of the 100 DS-1000 and 79 NPBench/Pythran perf cases).

The author's decision (2026-10-07): "Let's introduce IR versions and get rid of the
ReadMemory/WriteMemory. Old memory segments and ReadMemory/WriteMemory statements are tailored
for zkRAM. Since zkRAM is decided in the construction layer, that does not make sense anymore. I
prefer a clean design."

## Terms

- **Cell**: one scalar element of an array (an IR statement's value).
- **Storage**: the compile-time identity of the cells a set of Python names can reach: an array
  and every view of it share one storage. It exists only in the frontend (the IR builder); it is
  what `segment_id` names today.
- **Array value / version**: an IR statement whose value is a whole fixed-length vector of cells.
  A write does not change a version; it makes a new one (SSA: static single assignment, every
  value defined once).
- **Version tree**: an `ArrayInit` and every version made from it by writes, with the reads of
  all of them. The layer decides how to build one version tree as a whole (a **memory group**).
- **Access map**: how a read or write names its cell: `offset + Σ_k i_k · stride_k` for runtime
  positions `i_k`, each in `[0, len_k)`. Compile-time indices are folded into the offset.
- **Position**: an index after NumPy's wrap (`i + n` for a negative `i`), so in `[0, n)` when
  NumPy accepts the index.
- **Construction**, **requirement**, **program-wide requirement**, **free fact**: as in
  `LOWERING-LAYER.md` and `layer/core-api.md`.
- **Overlay**: the functional representation of writes, `cell' = select(hit ∧ p, written, cell)`
  for every cell a write may touch.
- **ROM / RAM**: a read-only table (halo2-base dynamic lookup, Noir native array) / a read-write
  memory with a consistency argument (halo2-base shuffle trace, Noir mutable array).

## Status

| step | what | state |
|---|---|---|
| 0 | inventory, census, base measurements, this document | done (1e8e0095) |
| 1 | IR ops `ArrayInit/Write/Read`, `Rom*/Ram*`: export/import, mock, folds, backends | done (c1a06b78) |
| 2 | layer, read side: element groups from `ArrayRead(ArrayInit)`, rounds, `Rom*` | done (8724bc75) |
| 3 | store behind the builder memory API; memory groups (overlay, RAM trace, native RAM) | done |
| 4 | frontend families onto the store API (writes at runtime positions are `ArrayWrite`s); B1-B9 | done |
| 5 | delete old ops, passes, backend memory code | done (rename deferred, see log) |
| 6 | validation and per-case cost table | todo |

(Updated as steps land; see "Status log" at the end.)

---

## 1. Inventory

Census of the final IR (`census_base.jsonl`, 178 perf cases × 2 targets, scattering excluded):
memory ops reach the backends in 33 of them, in four shapes:

| shape | cases | what the backend does |
|---|---|---|
| a segment written at constant addresses and never read (dead) | 0322, 0353, 0374, 0412, 0442-0444, 0462, 0481, 0507-0510, gemver, mvt, symm, trmm, clip, ising, lu ×2, periodic_dist, tax | nothing (halo2-base wiring; Noir rebinding) — the IR keeps them only because writes are `is_fixed` |
| runtime-address reads of a written-once segment | halo2: 0319, 0450, 0503-0505 (`ReadTable`); Noir: 0319, 0381, 0383, 0450, value_index (`ReadMemory`); both: insertion_sort, make_decision, projection_simplex | halo2-base lookup table, or its own mux / one-hot gather; Noir native array read or one-hot gather |
| runtime-address writes then reads | 0298 (5 writes into a 25-cell dynamic array) | halo2-base overlay (or RAM by its own cost rule); Noir one-hot scatter |
| none | the other 145 | — |

Every other runtime access is already lowered before the backends: reads of known cells are
`OpKind::Element` placeholders, writes to known cells are frontend overlays or
`OpKind::IndexWrite`/`MaskWrite` placeholders.

### 1.1 IR and builder

- `IR::AllocateMemory{segment_id, size, init_value}`, `WriteMemory`, `ReadMemory`, `ReadTable`,
  `MemoryTraceEmit`, `MemoryTraceSeal`, `DynamicNDArrayGetItem/SetItem` (`ir_defs.rs:263-316`).
  All are `is_fixed` (`ir_defs.rs:852-862`): dead-code elimination never removes them, and reads
  are linked to writes only by statement order (`get_topological_order` is the vector order,
  `ir.rs:328`). Export/import `ir_defs.rs:1058-1112, 1381-1427`; result typing `builder.rs:3497`.
- `DynamicNDArrayGetItem/SetItem` have **no producer** outside builder unit tests
  (`builder.rs:3958, 4028`); every proving backend ignores their `segment_id` and uses segment
  `10000 + array_id`. `MemoryTraceEmit/Seal` are no-ops on every backend.
- Builder chokepoints (`builder.rs:2648-2990`): `ir_allocate_memory` (capacity,
  `predicate_exempt_segments`), `ir_write_memory` (`guard_address`, `apply_write_predicate` =
  `ReadMemory` + `select(p, new, old)`, per-cell bound capture/invalidation, provenance
  `on_segment_write`), `ir_read_memory` (guard, bool cells, republished cell bounds),
  `ir_read_table`. Side tables keyed by segment: `segment_capacity`, `static_array_payload`
  (compile-time cells), `lazy_static_array_segments`, `static_array_product_provenance`,
  `segment_cell_bounds`, `segment_default_bounds`, `bool_segments`, `fixed_width.segments`,
  `predicate_exempt_segments`.
- Provenance keyed by segment (`provenance.rs`): `segment_hull` (value interval of a segment so
  far; read by halo2-base `interval_of` and Noir `apply_segment_iv`), `write_version`,
  `slice_views`, `mask_read_views`, `runtime_extents`, `compaction`, `layout` (unused).

### 1.2 Frontend producers and consumers (about 400 call sites in 40 files)

Two representations live side by side today: the segment (what the backends see) and the
**payload cache** `static_array_payload` (the cells the compiler knows). Static reads use the
cache; the segment is written lazily ("materialized") when something needs memory. Cache
coherence is the callers' job (`ir_write_memory` never touches it); the masked-write bug of
`fix/npb-gaps.md` was a cache that disagreed with the segment.

| family | files | constant address | runtime address |
|---|---|---|---|
| constructors, inputs | `static_array/base.rs`, `segment.rs`, `ir_gen/helpers.rs:369,494`, `ops/static_ndarray_ops/constructors.rs` | cache + lazy segment, or eager writes | — |
| generic consumers (elementwise, reductions, einsum, statistics, matmul, sorting, `select_value`, transpose/concat/stack) | `elementwise.rs:48 payload_cells`, `base.rs:406 to_value_list`, `shape.rs:86` | cache, else constant reads | — |
| static reads | `static_array/read.rs` | cache (`read_leaf_at_flat`) | `Element` placeholders over known cells (`read_cells`, `PayloadGather`, scan results, sorted outputs); otherwise `ReadMemory` (`read_leaf_at_checked_address`, guarded variants, runtime-bound slices, `materialise_axis0_dynamic_row`) |
| static writes | `static_array/write.rs`, `mask.rs` | `WriteMemory` + cache update | ≤ 1024 known cells: overlay selects in the cache (`try_overlay_*`) or `IndexWrite` placeholders; else `WriteMemory` + cache dropped |
| masked writes | `mask.rs:273` | constant writes of `select`/`MaskWrite` lanes | — |
| dynamic (runtime-shaped) arrays | `ops/dyn_ndarray/*`, `array_ops/{indexing,assignment}.rs`, `named_attr/dynamic_array_compat.rs` | `read_all_typed` + `alloc_and_write_values` into a fresh segment per op | `dyn_getitem_element`, in-place `dyn_setitem`, `dyn_setitem_slice`, `dyn_concatenate`, `dyn_transpose` scatter, aggregation `last_idx` |
| views | `read.rs:804 static_basic_view`, `shape.rs` reshape/transpose/flatten | same segment, adjusted offset/strides | — |
| outputs, in-place kernels | `ir_gen/outputs.rs:313-380` | cache / constant reads | in-place outputs found by scanning for `WriteMemory` to an argument's segment |
| lists | `ir_gen/list_sharing.rs`, `detached_copies.rs` | lists are values; a shared list's mutation is rejected | — |
| `out=` / `op=` | `named_attr/array_compat.rs:95-124` | constant writes, cache dropped | — |

### 1.3 Passes

| pass | role today | after |
|---|---|---|
| `UnreadMemoryEliminator` (`unread_memory_elimination.rs`) | drop segments nobody reads (writes are fixed) | deleted: array ops are pure, DCE does it |
| `ConstantMemoryForwarding` | forward constant-address reads to stored values; drop dead stores (order-sensitive walk) | deleted: constant accesses never reach the IR before a runtime write (the store), and the version folds (2.4) cover the rest |
| `SelectChainForwarding` | read-after-write forwarding on the unrolled overlay form | kept for select chains; the array form is the fold `read(write(v, i, x, true), i) = x` (2.4) |
| `DynamicNDArrayMemoryLowering` | legacy `halo2` + `enable_memory_consistency` only: Get/SetItem to mux or memory | deleted (its ops have no producer) |
| `MemoryTraceInjection` | no-op trace markers | deleted |
| `DynamicNDArrayMetaAssertInjection` | shape asserts of dynamic inputs | kept (not memory) |
| `DuplicateCodeElimination` | memory ops never deduplicated | array ops deduplicated like any pure op |
| `ConstantTableReads`, `ArgmaxQueries` (`lower/sort/{table,argmax}.rs`) | recognize constant segments read at runtime addresses (order-sensitive) | match `ArrayRead` of an `ArrayInit` of constants |
| range, SMT, select-merge, predicated simplification | memory opaque | `ArrayRead` bounded by the hull of its version (2.5) |

### 1.4 Lowering layer

- `OpKind::Element` (gather): `x[i]` over one line of known cells. Constructions: select chain,
  indicators (optionally one-hot asserted), recorded one-hot, `ElementOfWrite`, **`ElementByMemory`**
  (writes the cells to a segment and emits `ReadMemory`: the backend picks mux / one-hot on
  halo2-base, native array or one-hot on Noir), `ElementByIndexBits` + `IndexBitsNode`,
  **`ElementByTable`** (segment + `ReadTable`). `Wrap` (memory) for NumPy's index wrap.
- `OpKind::IndexWrite` (`scatter/write.rs`: one-hot write, zero-write chain) and `MaskWrite`.
- The sort family reads payloads and outputs through segments (`sort/mod.rs:427`,
  `sort/emit.rs:226-246, 456-474`, `sort/search.rs:345-378`), priced by `UnitCosts::read`.
- The table infrastructure is priced by an amortized share (`memory/cost.rs:50-81`), not as a
  program-wide requirement; the challenge is a `Requirement::Program` (`challenge.rs`,
  `emit::select_all`).

### 1.5 Backends

| op | mock | halo2-base | Noir | legacy halo2 |
|---|---|---|---|---|
| Allocate | vector of init | segment, init a constant | slots = init literals | same |
| Write, constant address | out of range → error | wiring before the first runtime write, overlay entry after | rebinding | rebinding |
| Write, runtime address | out of range → error | `assert_less` + overlay entry, or a RAM trace row (`plan.rs::ram_segments`) | one-hot scatter over feasible slots | one-hot + select per slot |
| Read, constant address | out of range → **init** | slot (+ overlay) | slot | slot |
| Read, runtime address | out of range → **init, satisfied** | mux / one-hot / two-root planes (+ overlay), or RAM; out of range unsatisfiable | one-hot or native ROM (`prefer_native_read`) | one-hot chain |
| ReadTable | out of range → unsatisfied | dynamic lookup per (segment, version) | native ROM | read |

halo2-base keys pin `table_layout` and `ram_traces` (`circuit.rs:184-195`). Backend memory
code: `halo2_base/synthesizer/memory.rs` (727 lines), `plan.rs::{ram_segments, reads_per_root,
address_only}`, `circuit/{ram,tables}.rs`, `noir/emit.rs` (memories, epochs, gathers,
`segment_array`), legacy `halo2/synthesizer.rs:3124-3287`, `mock/synthesizer.rs:502-685`,
`preprocess/mod.rs:464-556`.

### 1.6 Bugs found by the inventory (confirmed by running the mock; the circuits compile the same IR)

| # | program | Zinnia proves | NumPy |
|---|---|---|---|
| B1 | `def kernel(A, k): A[k] = 3` (in place, runtime `k`, ≤ 1024 cells) | **no output at all** (the overlay emits no `WriteMemory`, which is what `in_place_outputs` scans for) | `A` with `A[k] = 3` |
| B2 | `t = m.T; t[t < 0] = 0; return m` | wrong cells zeroed (mask write addresses `offset + i`, ignoring strides) | |
| B3 | `c = m[:, 1]; c[c < 0] = 0; return m` | wrong cells (same cause) | |
| B4 | `t = m.T; return t[i]` (runtime `i`) | a row of `m` instead of a column (`materialise_axis0_dynamic_row` ignores inner strides) | |
| B5 | `f = x.flatten(); f[0] = 99; return x` | `x` changed (`flatten` returns a view) | `flatten` copies |
| B6 | `y = x[2:6].reshape(2, 2); y[0, 0] = 99; return x` | `x` unchanged (reshape of an offset view copies) | reshape of a contiguous view is a view |

Each is a wrong output that verifies. The migration fixes them (section 3) with regression tests
on every backend.

---

## 2. IR design

### 2.1 Operations before lowering (frontend → layer)

```
v  = ArrayInit  { len, elem }               (c_0, ..., c_{len-1})
v' = ArrayWrite { len, elem, access }       (v, i_0, ..., i_{r-1}, x, p)
y  = ArrayRead  { len, elem, access }       (v, i_0, ..., i_{r-1})
access = { offset, axes: [(len_0, stride_0), ..., (len_{r-1}, stride_{r-1})] }
```

- `elem` is `Int` or `Float` (the number type of the cells and of a read; floats are Q32 field
  elements as everywhere). Complex arrays are two storages (real, imaginary), as today.
- A version is a statement like any other: its `Value` in the builder is an opaque handle the
  store keeps; no frontend code treats it as a number.
- Cell address `a = offset + Σ_k i_k · stride_k`; the producer guarantees `a < len` for every
  `i_k < len_k` (checked when the op is built).

**Semantics** (exact, on field elements):

- `ArrayInit(c)` is the vector `c`.
- `ArrayRead(v, i)`: if every `i_k ∈ [0, len_k)`, the cell `v[a]`. Otherwise **there is no
  witness** (no proof): a construction either proves the bound itself or relies on the
  precondition fact `0 <= i_k < len_k`, which the frontend's check or a free fact justifies.
- `ArrayWrite(v, i, x, p)`: same index rule; the vector `v` with `v[a] := x` when `p ≠ 0`, and
  `v` unchanged when `p = 0`. `p` is the write's execution predicate (the constant `true` when the
  write runs on every path).
- All three are **pure**: no side effect, removed by dead-code elimination when unused,
  deduplicated when equal.

**Indices.** Operands are positions: the frontend applies NumPy's wrap (the `Wrap` placeholder,
or the pinned sign, as today) and checks `-n <= i < n` where the access runs
(`checked_axis_index`, `discharge_index_in_range`), and passes the position guarded by the
execution predicate (`select(p, w, 0)`, which is in range where the code does not run). So every
index operand is in range on every path of an accepted witness, and the "no witness" rule only
bites on programs NumPy rejects. A negative or too-large position has no proof on every backend
and on the mock (today the mock returned the init value for an out-of-range runtime read: fixed).

**Multi-dimensional access.** Each runtime index keeps its own axis in `access.axes`
(`a[i, :, j]` with `i, j` runtime is two axes; constant indices go into `offset`). Constructions
use the per-axis structure (per-axis hits for an overlay, per-axis one-hots or planes for reads)
and compute a flat address only where they need one (RAM, ROM).

**Bounded-dynamic arrays** (runtime shape, `DynamicNDArray`): the version holds the capacity
cells (`len = envelope.total_bound`); the runtime extent stays in the array's metadata (scalar
statements), as today. Padding cells hold defined values (the constructor's fill, 0 by default,
never another array's data). Accesses use flat runtime addresses over the capacity (runtime
strides), so their access map is one axis of length `capacity`; the frontend's checks enforce the
runtime extent where NumPy would raise.

### 2.2 Operations after lowering (layer → backends)

```
t  = RomInit { len }   (c_0, ..., c_{len-1})        a read-only table
y  = RomRead { len }   (t, a)                       t[a]; no witness for a ∉ [0, len)
m  = RamInit { len }   (c_0, ..., c_{len-1})        a read-write memory, version 0
m' = RamWrite { len }  (m, a, x)                    m with m[a] := x
y  = RamRead { len }   (m, a)                       m[a]
```

- Flat addresses `a` (the layer computes them). ROM/RAM ops are pure, not fixed: a
  `RomRead`/`RamRead` that nothing uses is dead code, so it is never the only proof of a bound
  (the bound is a fact justified by a check or by the node that emits the read).
- RAM discipline (checked by every backend; a violation is a compiler error): each RAM version is
  consumed by at most one `RamWrite`, and every `RamRead(m, ·)` lies, in statement order, after
  `m`'s definition and before the `RamWrite` that consumes `m`. So statement order is a valid
  access order (the trace's time).
- RAM writes are unconditional; the layer emits a predicated write as
  `RamWrite(m, a, select(p, x, RamRead(m, a)))`.
- No other memory op reaches a backend; `ArrayInit/Write/Read` and `IR::ArrayOp` are rejected
  there ("not lowered").

### 2.3 Mock

The mock evaluates all eight ops: versions, tables and memories are vectors of field elements.
`ArrayRead`, `ArrayWrite`, `RomRead`, `RamRead`, `RamWrite` with an index outside its range mark
the proof **unsatisfied** (not a synthesis error). It checks the RAM discipline. Because the mock
interprets the pre-lowering ops too, the layer can be tested against it construction by
construction, and the IR can be checked before and after lowering against NumPy.

### 2.4 Folds (in the builder, so at IR generation and in `ConstantFold`)

With `w = ArrayWrite(v, j, x, p)` and positions compared as constants or as the same statement:

1. `ArrayRead(ArrayInit(c), k)` with a constant address `k` → `c[k]`.
2. `ArrayRead(w, k)` → `ArrayRead(v, k)` when no cell `k` can name is one `j` can name: two
   different constant addresses, or lines of cells that do not meet (`a[r, i]` and `a[s, j]`
   for rows `r != s`; an access with at most one runtime position names an arithmetic
   progression of cells, `ArrayAccess::disjoint_from`).
3. `ArrayRead(w, j)` (same constant address, or same index statements and access map) → `x`
   when `p` is the constant true. Under a runtime `p` the read stays: the rewrite
   `select(p, x, ArrayRead(v, j))` would read an older version after the write, which one
   memory (the RAM constructions) cannot serve, and it is not cheaper for the overlay.
4. `ArrayWrite(v, j, x, false)` → `v`.

Each is the definition of the ops; they replace `ConstantMemoryForwarding` and the memory part of
`SelectChainForwarding`. Two frontend rules make them apply: one index on axes of one length
under one execution predicate has one checked position (`IRBuilder::checked_position`, so a read
and a write through `a[i]` name the same statement), and `ConstantFold` runs again after
`DuplicateCodeElimination` (equal index expressions written twice, `a[c - 1] *= m` then
`a[c - 1]`, become one statement there), followed by dead-code elimination of the writes no read
reaches.

### 2.5 Facts and ranges

The interval of a read is the hull of the cells its version can hold: `hull(ArrayInit(c)) =
∪ interval(c_k)`, `hull(ArrayWrite(v, ., x, .)) = hull(v) ∪ interval(x)` (`range.rs`, and the
builder's provenance `set_interval` for a read, replacing `segment_hull`). A read of a version
whose cells are all bool is a bool cell (replacing `bool_segments`). Per-cell bounds the builder
learns at constant writes stay compile-time knowledge of the store (no IR).

### 2.6 What is *not* an op: no `ArraySelect`

A runtime `if` never merges two versions of one storage: writes in a branch are `ArrayWrite`s
predicated by the branch's path condition, on the one current version. A variable bound to
different arrays in the two arms is merged cell by cell (`select_value`, as today) into a new
storage. An `ArraySelect(c, v1, v2)` would only pay off when a merged array is read at a runtime
index and neither arm's cells are otherwise needed (two reads and one select instead of `n`
selects); no measured case does that. It can be added later with `read(select(c, v1, v2), i) =
select(c, read(v1, i), read(v2, i))` as its only rule.

---

## 3. The frontend: storages and the store

The builder owns one **store**: `storage id → Storage { len, elem, state, bool, fixed_width,
hull }` in a `BTreeMap`. A storage's state is either

- `Cells(Vec<Value>)`: the compiler knows every cell's value (a statement or a constant); no IR
  exists for the array, or
- `Version { v, reads }`: the cells are those of the IR version `v` (after a runtime write), with
  the constant-address reads of `v` built so far (memoized).

API (one chokepoint each; every current `ir_*_memory`, payload-cache and materialize call site
moves onto it):

| call | `Cells(c)` | `Version{v}` |
|---|---|---|
| `read(s, constant a)` | `c[a]` (no IR) | `ArrayRead(v, a)`, memoized |
| `read(s, access, runtime i)` | `ArrayRead(ArrayInit(c), access, i)` (the init memoized per state) | `ArrayRead(v, access, i)` |
| `write(s, constant a, x)` under predicate `p` | `c[a] := select(p, x, c[a])` (`x` when `p` is true) | `ArrayWrite(v, a, x, p)` |
| `write(s, access, runtime i, x)` under `p` | `ArrayWrite(ArrayInit(c), access, i, x, p)` → `Version` | `ArrayWrite(v, access, i, x, p)` |
| `cells(s)` | `c` | `read(s, a)` for every `a` |

Rules that follow from it:

- **One source of truth.** There is no second copy of the cells: the payload cache, lazy
  segments and materialization disappear, and with them the class of the masked-write bug (a
  cache updated differently from memory). The write predicate is applied in exactly one place.
- **No branch-local exemption.** Every write under a runtime predicate `p` keeps `p`. (An
  earlier version let a storage created under `p` skip it, assuming such a storage is invisible
  where `p` is false. It is not: rebinding a name under `p` merges `select(p, new, old)` cell by
  cell into a new storage created under `p`, which holds the old cells where `p` is false, so a
  later write under `p` changed values of the branch not taken: review finding F1. The base's
  `predicate_exempt_segments` had the same bug for constant-index writes.)
- **Constant accesses stay compile-time** while no runtime write has happened (`Cells`), exactly
  as the cache did; after one, they are IR reads of the version, which the layer resolves
  (overlay: the cell; RAM: a trace read) and the folds of 2.4 simplify.
- **Reads of known cells keep the frontend's strategies.** `read_cells`, `PayloadGather` planes,
  scan results and sorted outputs pick a line of cells and a (combined) index, as today; they now
  emit `ArrayRead(ArrayInit(line), index)` instead of an `Element` placeholder, so there is one
  read op. The layer groups such reads exactly as it grouped placeholders (same cells, same group).

### 3.1 Pitfalls and how the design handles each

- **Predicated writes** (a write in a branch not taken changes nothing). The predicate is an
  operand of `ArrayWrite` (runtime index) or folded into the cell's select by the store (constant
  index); there is no other path. Tests: every write family (element, slice, row, mask,
  augmented, complex, fancy, dynamic arrays) under a runtime `if` with the branch not taken, on
  the mock, halo2-base and Noir; nested branches; loops with a `break`.
- **NumPy views.** Today `x[a:b]`, `x[i]` on ≥ 2-D, `x[:, j]`, `.T`, `moveaxis`, contiguous
  `reshape`, `ravel` share the segment (writes through the view change the base), reversed or
  runtime-bound slices, fancy indexing and runtime-index rows are copies, and lists are values
  whose shared mutation is rejected (probe: `array-ssa-data/probe_views.py`, all equal to NumPy).
  Preserved: a view is `(storage, offset, strides, shape)` and every access goes through the
  storage's absolute address. Wrong today and fixed (B2-B6): mask writes and runtime-index rows
  through non-contiguous views use the view's strides; `flatten` copies (NumPy); `reshape` of a
  contiguous view with an offset is a view (NumPy). `ravel` stays a view (NumPy returns a view of
  a contiguous array).
- **Python lists.** Unchanged: lists are values; mutating a list reachable through two names is
  rejected (`list_sharing.rs`), arrays inside lists keep their storage.
- **Loops** are unrolled; a write inside a loop is one `ArrayWrite` per iteration on the current
  version, predicated by the iteration's activity. Reads after writes read the current version.
  Test: write-then-read chains in loops, with runtime and constant indices, under breaks.
- **In-place kernels / mutated outputs.** An argument is an output iff its storage is not in its
  initial state at the end (any write, runtime or constant): fixes B1. Its cells are `cells(s)`.
- **Function calls** are inlined with arguments bound to the same storage (unchanged).
- **The masked-write fix** (`fix/npb-gaps.md`) becomes structural: the mask write is a sequence
  of constant-address `write`s under the predicate, and the store keeps what it stored.
  `test_predicated_mask_write.py` stays.
- **Copies.** `x.copy()`, `np.array(x)` and every op building a new array create a storage in
  `Cells` state (sharing cell values, which is free); a copy of a `Version` storage reads its
  cells (`read` per cell) — the layer builds them only if used.

---

## 4. The layer

### 4.1 Groups

`ArrayLowering` lowers in **rounds** until no placeholder, `ArrayInit`, `ArrayWrite` or
`ArrayRead` is left (at most 4; a fifth is a compiler error). A round lowers every group present;
an emitter may emit `ArrayRead`s of an `ArrayInit` (a read whose construction it leaves to the
layer, e.g. an overlay's runtime reads or a sort's read at a hinted index), which the next round
lowers.

- **Memory group** (new): a version tree with at least one `ArrayWrite`. Requirements: its live
  `ArrayRead`s (runtime and constant: the constant ones are the live cells of versions, including
  the final versions behind outputs).
- **Element group** (`OpKind::Element`, unchanged constructions): the reads of one line of a
  write-free `ArrayInit`: `ArrayRead(ArrayInit(c), access, i)`; its shared operands are the cells
  the access can reach, its lanes the reads, and a lane's own operand its index (a combined
  index for a multi-axis access is built by the frontend, as today).
- Every other `IR::ArrayOp` group as today.

### 4.2 Constructions of a memory group (`lower/memory/versions.rs`)

Each is a whole-group candidate (the selection weighs the array's representation once, with all
its reads and writes):

| construction | applies | emits | halo2-base cost | Noir cost |
|---|---|---|---|---|
| **overlay** | always | per write, per cell the write may reach (pruned by the index's range), `select(hit ∧ p, x, cell)`, hits per axis by equality tests (shared per index), the `Wrap` hits, or the pinned position one-hot (`scatter::position`); per runtime read an `ArrayRead(ArrayInit(line cells of its version))` for the next round; per constant read the cell | selects + hits (calibrated units of `scatter/write.rs`, `gather/cost.rs`) + per read the cheapest element construction for its line | same, Noir units |
| **zero writes** (`ZeroWrites`) | every write stores the constant 0 through one runtime position, unconditionally | per cell `x · Π keep`, a keep bit `Π (1 - h)` per (axis, coordinate) shared by the cells there | 4 cells per product | 1 gate per product |
| **RAM trace** | halo2-base; the tree is a chain whose reads respect the RAM discipline (a read of an older version is allowed when no later write before it can change its cell: fold 2 left it there) | `RamInit`, `RamWrite` (predicated as in 2.2), `RamRead` | ≈ 8 cells per access + 8 per slot (measured, `layer-memory.md` 1.2); **pre**: `Program{RAM}` | — |
| **native RAM** | Noir; same chain condition | same ops (Noir: `let mut` array, `array_index`) | — | measured per access/slot (1.2) |

Preconditions: none beyond the overlay's hit bits (the `Wrap` encodings). Every position is in
its axis on every path of an accepted witness by the frontend's contract (its check or proof,
and the guard where the code does not run; `crate::ir_array`), so no construction lists the
per-axis facts. Postconditions: the lanes (and the cells of the versions read at runtime
positions, for the element constructions). The overlay's cost counts a select storing the
constant 0 as one product. The position one-hot wrap (`WrapByPositionHits`) also states
`-n <= i < n` for the raw index, as the scatter family's index writes did, so a check it implies
is dropped.

Soundness: overlay — each new cell is NumPy's after the write, a function of its inputs (no
hint), because exactly the cell at the write's address is selected when `p` holds and every other
cell is unchanged (the index facts make the address one cell); the pinned one-hot's argument is
in `scatter/position.rs`. RAM — `circuit/ram.rs`: the shuffle makes the sorted trace a
permutation of the program-order trace; the sorted-trace gates make each read equal the latest
earlier write at its address (or the initial row); a read's value is a hint pinned by that
argument (tamper test). Noir native RAM — ACIR memory semantics plus `array_index`'s bound.

### 4.3 Program-wide requirements: tables and RAM paid once

`Requirement::Program { name }` gains two names next to `CHALLENGE`:

- `TABLES` (halo2-base): the table groups' fixed columns and lookup argument, paid once;
  `ElementByTable` lists it in `pre` instead of the amortized share. Its producer cost is one
  table group's fixed columns (`TABLE_GROUP_CELLS`); slots beyond one group's rows are priced
  per slot inside the candidates.
- `RAM` (halo2-base): one RAM trace for the whole program. The backend lays every RAM array
  into one trace (addresses of array `t` offset by the sizes of the arrays before it). The trace
  proves only that every global address is in `[0, total)`, which would let an address outside
  array `t` reach another array's cells; so every access at a runtime address also asserts its
  own bound `a < n_t` in the circuit (free when the program's index check already proved it).
  Review finding F2: before, the backend relied on the frontend's check alone and recorded the
  bound as proven. Producer cost: the trace's fixed columns
  (`RAM_FIXED_CELLS`); rows priced in the candidates.

`emit::select_all` already enumerates subsets of producible program-wide requirements and pays
each once; Noir has none (its native ROM/RAM cost is per array).

### 4.4 Reads (`Element` groups): what changes

- `ElementByTable` emits `RomInit`/`RomRead` instead of a segment and `ReadTable`.
- `ElementByMemory` is deleted. Its halo2-base behaviour (the backend's mux for at most two reads
  per index, else shared indicators and inner products) is what `ElementByIndexBits` and
  `ElementByIndicators` already build in IR; its Noir behaviour (a native array read) is
  `ElementByTable` on Noir. Costs re-checked case by case (0319, 0381, 0383, 0450, value_index on
  Noir).
- The sort family's reads at hinted indices and runtime ranks emit `ArrayRead(ArrayInit(cells))`
  for the next round instead of segment reads; `UnitCosts::read` stays the estimate.
- `ConstantTableReads` and `ArgmaxQueries` match `ArrayRead` of an `ArrayInit` of constants.
  Lines of constants that are not affine are element groups with constant cells; a construction
  for them (indicators with constant coefficients, `Σ_k [i = k]·c_k`, the backend's former
  constant-slot gather) keeps their cost.

---

## 5. What gets deleted

- IR: `AllocateMemory`, `WriteMemory`, `ReadMemory`, `ReadTable`, `MemoryTraceEmit`,
  `MemoryTraceSeal`, `DynamicNDArrayGetItem`, `DynamicNDArraySetItem` (and their export/import,
  typing, interpreter dispatch, `Synthesizer` trait methods).
- Builder: `ir_allocate_memory`, `ir_write_memory`, `ir_read_memory`, `ir_read_table`,
  `ir_dynamic_ndarray_get/set_item`, `alloc_segment_id`, `reserve_segment_ids_below`,
  `apply_write_predicate`, `predicate_exempt_segments`, `segment_capacity`,
  `static_array_payload`, `lazy_static_array_segments`, `segment_cell_bounds` /
  `segment_default_bounds` (moved into the store), `bool_segments` (store); `helpers/segment.rs`;
  `materialize_static_array_*`.
- Provenance: `segment_hull`, `write_version`, `on_segment_write`, `layout`; slice/mask views,
  runtime extents and compaction re-keyed to storages.
- Passes: `UnreadMemoryEliminator`, `ConstantMemoryForwarding`, `MemoryTraceInjection`,
  `DynamicNDArrayMemoryLowering` (with the `enable_memory_consistency` flag), the memory branch
  of `SelectChainForwarding`.
- Layer: `ElementByMemory`, `next_free_segment_id`, the amortized table share, `ElementOfWrite`
  (replaced by fold 3), `IndexWrite` placeholders (the overlay construction).
- halo2-base: `synthesizer/memory.rs` (segments, overlay, gathers, `ram_chosen`), `plan.rs`
  `ram_segments` / `reads_per_root` / `address_only`; `circuit/ram.rs` and `tables.rs` stay,
  driven by `Ram*`/`Rom*` ops. Noir: segments, epochs, one-hot gathers and scatters of
  `emit.rs`; native arrays stay for `Rom*`/`Ram*`. Mock and preprocess: segment code. Legacy
  halo2 (compiled, retired from proving) implements neither `Rom*` nor `Ram*`: a program whose
  lowering keeps a table or a memory is rejected there ("unsupported"), which is acceptable only
  while it stays retired.

After lowering the backends receive scalar ops, `Hint`/`IdentityCheck`, and `Rom*`/`Ram*`.

---

## 6. Migration plan (each step builds, passes, and is committed)

1. **IR ops** `ArrayInit/Write/Read` and `Rom*/Ram*` with export/import, typing, purity, mock and
   preprocess semantics, the folds of 2.4, range hull; unit tests. Nothing produces them yet.
2. **Layer, read side**: element groups from `ArrayRead(ArrayInit)`; rounds; `ElementByTable` →
   `Rom*`; halo2-base and Noir implement `Rom*` (on the existing table and native-array code).
   Frontend: the element placeholder sites emit `ArrayRead(ArrayInit(line))`. Costs must be
   identical (same groups, same constructions).
3. **Store** behind the existing builder memory API: `ir_allocate_memory`, `ir_write_memory`,
   `ir_read_memory` operate on storages and emit `ArrayInit/Write/Read`; the payload cache
   becomes a view of the store. Memory groups with the overlay and RAM/native-RAM constructions;
   halo2-base and Noir implement `Ram*`. From here no `*Memory` op reaches a backend.
4. **Frontend families moved onto the store API** one by one (static writes and overlays,
   `IndexWrite` → `ArrayWrite`, mask writes, dynamic arrays, outputs and in-place detection,
   views), fixing B1-B6 on the way; sort family reads → `ArrayRead`.
5. **Deletions**: old IR ops, passes, backend memory code, segment helpers, provenance records;
   rename `segment_id` → `storage`.
6. **Validation**: costs on 179 cases × 2 targets against the base (no case costlier), output
   checks, real proofs, acceptance account, test suites, tamper and semantic tests.

Each step's cost table is compared with `costs_base.json`; any change is explained here.

## 7. Status log

- 2026-10-07: inventory, census (`census_base.jsonl`), base costs (`costs_base.json`, 178 cases ×
  2 targets, no errors), bugs B1-B6 confirmed on the mock, this design.
- Steps 1-2 (c1a06b78, 8724bc75): the ops, mock, backends' `Rom*`/`Ram*`; element reads as
  `ArrayRead(ArrayInit)`; costs identical to the base.
- Steps 3-4 (one commit): the store (`builder/store.rs`) is the only copy of every array's cells;
  the payload cache, lazy segments and materialization are gone; every write at a runtime
  position is an `ArrayWrite` (the frontend overlays, `OpKind::IndexWrite` and its one-hot and
  zero-chain constructions are deleted; the memory group's overlay, zero writes, RAM trace and
  native RAM decide the representation); `ElementByMemory`, `UnreadMemoryEliminator`,
  `ConstantMemoryForwarding`, `MemoryTraceInjection`, `DynamicNDArrayMemoryLowering` deleted.
  Fixed: B1 (in-place outputs by the store's write clock), B2/B3 (mask writes through view
  strides), B4 (runtime row of a view through every stride), B5 (`flatten` copies), B6 (reshape
  of a contiguous view at an offset is a view), B7/B8 (a typed store keeps the real fill cells),
  B9 (complex `out=` writes both parts), and a cycle in the layer's emission order (a node's
  start waited for every version its lanes read; now only for the versions whose cells it
  reads, so a RAM serving a read-modify-write chain lowers). Costs against `costs_base6ff.json`
  (`costs_step4d.json`): 334 of 356 unchanged, 19 cheaper (0381 halo2 -2,007; insertion_sort
  halo2 -10,075 at the base's k, Noir -1,031; periodic_dist -9,237 / -729; lu, make_decision,
  projection_simplex, clip, 0450, 0383, 0298, value_index), 3 costlier on halo2 by 15-17 cells
  (0345, 0346, 0347: reads of a column at a runtime index, fixed after the merge with
  eval-refactor 35c86a81, see "Open"). After it (`costs_calib.json` against `costs_base35c.json`):
  337 unchanged, 19 cheaper, none costlier.

- Step 5: `AllocateMemory`, `WriteMemory`, `ReadMemory`, `ReadTable`, `MemoryTraceEmit/Seal`,
  `DynamicNDArrayGetItem/SetItem` deleted with their builder API (`ir_*_memory`: callers use the
  store's `storage_read_at` / `storage_write_at`), the `Synthesizer` trait's memory methods and
  every backend's implementation (halo2-base `synthesizer/memory.rs` and the plan's
  `ram_segments`, `reads_per_root`, `address_only`; Noir's segments, epochs, gathers and native
  segment arrays; the mock's and preprocess's segments; legacy halo2's segments), the
  provenance table's `segment_hull`, `write_version` and `layout` (and the segment-keyed
  records' wire form: only the frontend reads them), the scatter family's `IndexWrite`
  constructions, and their tests. The select chains' one-hot inner products live in
  `halo2_base/synthesizer/one_hot.rs`. Tests of the deleted code are deleted; the tamper tests
  of tables were ported to `RomRead` (halo2-base and Noir). Not done: renaming `segment_id` to
  `storage` across the frontend (several hundred sites in files other branches edit; a
  mechanical change best made when they have merged).

- Step 6 (validation at 28c4b9c9 = eval-refactor 35c86a81 merged): costs on 178 cases x 2 targets
  against `costs_base35c.json` (the merged base, which equals `costs_base6ff.json`): 337 unchanged,
  19 cheaper, none costlier (`costs_calib.json`); IR and Noir digests identical across separate
  measurement processes. check_outputs 100/100 on halo2-base and Noir; noir_differential (fresh
  output) 100/100 match; costs_halo2 `--check` 100/100 (keygen, prove, verify, tamper rejected, mock
  equal, NumPy match); acceptance account 214 accepted correct on both trees, no wrong output, two
  rejections whose reason moved (nbody, convnet: see "Open"); Python suite 3,548 passed; Rust lib:
  the known 18 failures (two legacy-halo2 cache tests flaky in parallel, pass alone); nargo test
  105 passed; harness run of the 79 NPBench/Pythran entries on zinnia_halo2 and zinnia_noir: 158
  of 158 ok (scattering included; its cost is not in the tables, which skip it on both sides).

- Soundness review fixes (branch `array-ssa-fix`, `eval-redo/layer/array-ssa-review/REVIEW.md`;
  regression tests `testing/lang/test_array_ssa_review.py` on the mock, halo2-base and Noir):
  - F1 (wrong output provable): the branch-local exemption is gone (section 3.1): every write
    under a runtime predicate keeps it.
  - F2: halo2-base asserts each RAM access's own bound `a < n_t` (the shared trace bounds only
    the global address) and no longer records the bound as proven without a constraint; the
    trait documents the per-memory requirement; MockProver test of unchecked IR reading another
    memory's cells.
  - F4 (wrong output provable, pre-existing): `np.swapaxes`, `np.asarray`, `np.atleast_1d`,
    `np.atleast_2d` are views of an array argument; `np.array` of an array and `.copy()` of a
    runtime-shaped array are copies. Where Zinnia's result is a copy of what NumPy returns as a
    view (`np.flip`, `np.fliplr`, `np.flipud`, `np.rot90`, `.T` / `transpose` / `swapaxes` /
    `moveaxis` of a runtime-strided array, and the listed functions on any fallback path), the
    two arrays are linked (`ir_gen::detached_copies`), so a write through either is a compile
    error when the other is read afterwards.
  - F5: the RAM emitter asserts that it serves reads in statement order and that no write it
    is past changes a read's cell; dead code removed (`overlay_known_cells`,
    `RamDiscipline::defined`); the legacy-halo2 claim corrected (section 5); the NumPy
    reference helpers keep scalar arguments scalars; `SelectMerge` computed an interval's width
    in i64, which overflowed and enumerated 2^64 points (the fuzz program that hung the
    compiler). The 37 GB acceptance compile: the acceptance entries that hit resource limits
    (`pythran_zero`, `pythran_mandel`, `case0489`, `pythran_factorize_naive` and six timeouts)
    are the same on the base; `zero` and `mandel` build 1,024 x 1,024 lists from runtime-bounded
    loops, which the compiler unrolls to their bounds, so an uncapped compile grows without
    limit. Not a bug of this branch; reported.
  - Validation against b0edb4da (`array-ssa-data/costs_af1.json`, `costs_fix.json`, `fix/`):
    IR and `main.nr` digests identical on 355 of 356 case-targets; `insertion_sort`
    halo2-base 9,642 -> 9,704 cells (+62: the per-access RAM bound on reads and writes whose
    positions the loop's structure proves, with no in-circuit check to make the bound free);
    acceptance 327 outcomes identical; check_outputs 100/100 on both backends;
    noir_differential 100/100; costs_halo2 `--check` 100/100; Python suite 3,686 passed; Rust
    lib the known 18 failures; determinism 200/200 and 158/158 stable over 3 processes.

- Second soundness review fixes (branch `array-ssa-fix2`,
  `eval-redo/layer/array-ssa-review/REVIEW-2.md`; regression tests
  `testing/lang/test_array_ssa_review2.py`, every `r2/q_*.py` reproducer on the mock,
  halo2-base and Noir):
  - N1-N4 (stale copies read through a method call, written through `out=` without a
    record, nested more than one container deep, or read by a merge). The detached-copy
    bookkeeping moved into the store (`builder/stale.rs`), so one place sees every read and
    one place every write. A link `(copy, base)` is kept between storages NumPy would share.
    A store write (`note_storage_written`) marks every storage linked to its target stale,
    and so does an update that rebinds a name to a new storage (`note_storage_replaced`).
    Every read of a storage's cells (`known_cells`, `storage_read*`, ...) rejects a stale
    storage with a compile error, whatever syntax reached it. Staleness is
    path-qualified: a storage is stale on the conjunctions of literals of the writes'
    execution predicates. A read is rejected when one of them can hold with the read's own
    predicate, which keeps the "elif rebind" pattern of F1 compiling. A merge
    (`select_value`) reads its operands without the check and carries their staleness to
    its result, conjoined with the branch condition or its negation, so a merged stale
    value is rejected only when it is read. Links are path-qualified too: a merge
    `select(c, a, b)` of the arrays a name was bound to is linked to `a` on the paths
    where `c` holds and to `b` on the others, so a write through the merge makes `b`
    stale only where the merge is `b`. (A negated conjunction is kept as such: a path
    that implies the conjunction cannot be in its negation.) A storage that a write reaches
    along several chains of links is marked stale on the literals the chains' paths have
    in common (a superset of their union), so the walk stays polynomial; following every
    chain path by path took exponential time on `In, Out = Out, In` in a loop that may
    stop early (Pythran `lapl2d`).
  - N5: `np.asarray(x, dtype=d)`, `np.array(x, copy=False | None)` and
    `x.astype(d, copy=False)` return `x` when `d` is `x`'s own exact dtype (bool, int64, a
    fixed-width integer). Zinnia has one float dtype for every float width (and one complex
    dtype), so for a float or complex `d` it cannot tell whether NumPy copies. It converts
    and links the result as a view, so a write through either array makes reading the other
    a compile error. `copy=False` where a copy may be needed is a compile error (NumPy
    raises `ValueError`).
  - N6: `IRContext::set` wrote an assignment in a loop into the nearest scope holding the
    name. When that scope lay outside an enclosing `else`/`elif` branch, the branch merge
    then overwrote the value with the other branch's. The write now goes into the innermost
    conditional scope between the loop and the owner. Tested with `while`, nested loops,
    `break`, `continue`, `return` and loops in every branch. The branch's scope then holds the name
    while the enclosing scope keeps its value before the branch; the list-sharing check
    (`IRContext::find_other_binding`) treats every binding of the name inside the current
    function as the variable itself, not as a second reference to the list.
  - N7: a merge of a sequence (an array's elements, as a list or tuple) with a number is a
    compile error. The two values have different shapes, and the merge used to take the
    true side on every path; a loop that may break before rebinding a name reaches it, the
    `if` check for different ranks does not.
  - N8: a reshape that copies a non-contiguous array is linked to its source when NumPy's
    no-copy reshape (`_attempt_nocopy_reshape`, ported as `reshapes_without_copy`) would
    succeed. Writes through either array are then compile errors.
  - N9: `np.real` / `np.imag` of a list or tuple (what a merged complex array is) take the
    part of each element.
  - N10: the results of `np.broadcast_to` and of the imaginary part of a real array are
    read-only storages. A write to one is a compile error, as NumPy raises.
  - N11: `np.expand_dims` of a runtime-shaped array exposed the old rank statement when the
    rank was not a constant; the rank is static and is stated as such.
  - Found by re-running the reviewer's fuzzers on the first commit of the branch (all
    pre-existing on bc584e01): an `op=` update of a runtime-shaped array or of a list did
    not note the write, and an update that rebinds a name to a new storage left the links
    on the old storage (the new one now takes them over, `move_links`). A write into an
    array held by a list was dropped silently (`set_nested_value`); it is written now,
    and the remaining silent drops there are compile errors. Two multi-dimensional arrays
    of different shapes merged into a list of rows, on which `m[i, j]` read row `i` and
    `m[i, j] = v` wrote nothing; such a merge is a compile error. A name bound in one arm
    of an `if` (the join drops it) and bound again in a loop that may stop before the
    assignment kept the loop's value on every path; that is a compile error
    (`IRContext::is_branch_only_binding`).
  - Validation against bc584e01 (`array-ssa-data/fix2/`; the code at ea0dff81 and the
    merge of eval-refactor fe282b23): IR and `main.nr` digests identical on all 179 perf
    cases on both targets (scattering included); acceptance 327 outcomes identical;
    check_outputs 100/100 on halo2-base and Noir; noir_differential 100/100;
    costs_halo2 `--check` 100/100; Python suite 3,968 passed (342 new tests, 96 of which
    fail on bc584e01); Rust lib the known 18 failures; determinism 200/200 and 158/158
    stable over 3 processes; the reviewer's fuzzers (44,300 programs, same seeds) 0
    mismatches; every `r2/q_*.py` reproducer matches NumPy or is a compile error.

- Third soundness review fixes (branch `array-ssa-fix3`,
  `eval-redo/layer/array-ssa-review/REVIEW-3.md`; regression tests
  `testing/lang/test_array_ssa_review3.py`, every `r3/q3_*.py` reproducer on the mock,
  halo2-base and Noir):
  - B1: unpacking a multi-dimensional array (`a, b, c = m`, `a, *rest = m`) gives the
    views of its rows that iterating it gives; they used to be lists of cells, so writes
    were lost in both directions. A chained write through a list element
    (`rest[0][0] = v`) writes into the array the list holds.
  - B2, B3: `np.broadcast_to` and `np.diag` of a 2-D array are linked to their argument
    as views and are read-only (`np.diag` of a 1-D array builds a new, writable array).
  - B4: Python binds a local name exactly where an assignment to it ran. Zinnia's scopes
    keep a name bound after the scope that first bound it, which is exact except for a
    first binding in a loop body the loop may leave before it (a `break` or `continue`
    earlier in it or in an enclosing loop). For such a name `IRContext` records the execution
    predicates of its assignments (an unconditional assignment ends the record), and a
    read of it (a bare name, a method or attribute of it, a write to an element of it)
    asserts their disjunction under the read's predicate (`ir_gen/unbound_names.rs`). The
    assertion is left out where the read's predicate implies an assignment's (a read in
    the iteration that bound it), so it costs nothing where Python cannot raise and no
    perf program changes.
  - FR1: the compile error for a name bound in one branch and assigned again in a loop
    (fix2) is gone; B4's record covers the case. Where the loop may skip its assignment,
    Python keeps the branch's value, which Zinnia dropped at the branch join, so those
    paths have no proof (sound, not complete).
  - FR2: a name bound to arrays of different shapes on different paths (multi-dimensional
    arrays of different shapes, or an array and a number) is bound to a stand-in array
    whose shape is marked unknown (`Staleness` of kind `Shape` in `builder/stale.rs`):
    reading its cells or metadata, or writing to it, is a compile error, so a merge that
    is never used costs nothing. A read whose path decides the merge's condition (a later
    iteration of the loop that rebound the name, under the condition that the loop went
    on) reads the merged array that path selects. A function's result merged over its
    `return`s stays a compile error at the merge, since it reaches code that reads its
    shape without a name lookup. An empty array has no storage of its own to identify a
    stand-in, so a merge whose stand-in would be empty is a compile error.
  - FR3: `.shape`, `.ndim` and `.size` of a static array come from its layout, so the
    metadata of a stale array is available; its cells are still rejected. (`len()` of a
    stale name stays rejected: call arguments are converted to cells before dispatch.)
  - A merge under a constant condition carries the selected operand's staleness on every
    path (`carry_merge_staleness`), not only through the operand being returned itself.
  - Found by fuzzing the first commit of the branch (the reviewer's `fuzz_rebind` and a new
    `fuzz_unbound.py` in `array-ssa-data/fix3/`): `name op= value` and element writes through
    a name rebound in a branch to an array of another shape went to the stand-in; they now
    resolve the name to the array the path selects (`usable_name_value`) and write it in
    place, leaving the name bound to the stand-in (rebinding it to the resolved array would
    drop the other paths' arrays: that first version was wrong output, caught by the same
    fuzzers), and are rejected where the path does not decide it. A name bound in both arms of an `if` in a loop the
    loop may leave before the `if` is recorded at the join (`note_join_binding`), and the
    record of a name the join drops (bound in one arm only) is forgotten, so a later
    binding does not inherit it. Zinnia still drops such a name at the join: where Python
    keeps the arm's value and a later loop may skip its own assignment, there is no proof.
  - Validation against bc584e01 (`array-ssa-data/fix3/`, code at 7b59f95c): IR and
    `main.nr` digests identical on all 179 perf cases on both targets (scattering
    included); acceptance 327 outcomes identical; check_outputs 100/100 on halo2-base and
    Noir; noir_differential 100/100; costs_halo2 `--check` 100/100; Python suite 4,503
    passed (538 tests in `test_array_ssa_review3.py`, 165 of which fail on bcc20473 and
    178 on bc584e01); Rust lib the known 18 failures; determinism 200/200 and 158/158 over
    3 processes. The reviewers' fuzzers on their round-2 and round-3 seeds and on new seeds
    (141,600 programs, in the automatic, forced-RAM, Noir native RAM, `@selects`,
    `@delegate`, joint-selection and layer-facts ablation configurations, 5,100 of them
    on all three backends) and `fuzz_unbound.py` (2,689 programs x 12 inputs, 214 of them
    on all three backends):
    no wrong output, no hang, no crash; 6 `fuzz_rebind` programs have no proof for an
    input Python runs, all of the branch-join drop described above. Every `r3/q3_*.py`
    and `r2/q_*.py` reproducer matches NumPy, is a compile error, or has no proof where
    Python raises.

### Open
- Dynamic-array lanes beyond the runtime length: `1.0 / x[x > 0]` and `x[x > 0] ** -1.5` used to
  be compile errors (the division's or power's precondition could not be stated over the
  padded cells); they now compile, and the precondition is asserted on every lane, padding
  included, so the honest witness has no proof when a padding cell is 0. No wrong output can
  verify (the circuit rejects), and no acceptance program changes outcome (npbench nbody and
  pythran convnet now stop at a later unsupported construct, a runtime `np.ndarray` length). The
  fix is to compute a runtime-shaped array's lanes under their validity (`lane < length`) as the
  execution predicate; it changes the IR of every runtime-shaped elementwise operation, so it is
  left for a decision. (Review finding F3, sound but incomplete; the affected set also includes
  `np.log(x[x > 0])`, `np.sqrt(x[x > 0] - 1.0)`, and, already on the base, integer `100 // y`
  and `100 % y` with `y = x[x > 0]`.) When fixing it, keep each padding lane's *result* pinned,
  e.g. `select(valid, r, 0)`: padding results are public values (a length-4 array exposes all
  its capacity cells), and a guarded division's quotient hint must not become free on them.

- Not implemented from section 4: the program-wide `TABLES` requirement (each `ElementByTable`
  candidate still prices its table group's fixed columns itself). The RAM constructions state no
  per-axis facts as postconditions, and must not without more: the backends' per-access bound
  is on the flat address `a < n_t`, which does not imply each position's own axis bound (on a
  2-D access `a[1, 5]` of a 4x4 array the flat address 9 is in range), so such a fact would let
  the selection drop the program's per-axis check (review finding F2); `ElementOfWrite` stays (it applies to
  any select pattern a program builds, and costs nothing when it does not).
- (resolved) 0345-0347 were 15-17 halo2-base cells over the base: the writes fold away, the
  selection priced the column reads alone, and its estimate of a select chain over a bounded index
  (fused by halo2-base into an inner product) counted one term per link where the backend emits
  one more (the value no link tests). Priced as emitted (`gather::cost::element_by_select_chain`),
  the position one-hot wins and all three equal the base.
