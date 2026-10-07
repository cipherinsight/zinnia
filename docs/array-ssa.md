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
| 0 | inventory, census, base measurements, this document | done |
| 1 | IR: `ArrayInit`/`ArrayWrite`/`ArrayRead` with mock semantics, export/import, DCE, folding | todo |
| 2 | frontend store: segments become storages; every access through the store | todo |
| 3 | layer: memory groups (overlay, RAM), reads of `ArrayInit` as element groups, rounds | todo |
| 4 | backends: `RomInit/RomRead`, `RamInit/RamWrite/RamRead`; old memory code deleted | todo |
| 5 | delete segments, passes, old ops; rename; tests | todo |
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

### 2.4 Folds (in `ConstantFold`, i.e. before lowering)

With `w = ArrayWrite(v, j, x, p)` and positions compared as constants or as the same statement:

1. `ArrayRead(ArrayInit(c), k)` with a constant address `k` → `c[k]`.
2. `ArrayRead(w, k)` with the address of `k` a constant different from `j`'s constant address →
   `ArrayRead(v, k)`.
3. `ArrayRead(w, j)` (same index statements, same access map) → `x` when `p` is the constant
   true, else `select(p, x, ArrayRead(v, j))`.
4. `ArrayWrite(v, j, x, false)` → `v`.

Each is the definition of the ops; they replace `ConstantMemoryForwarding` and the memory part of
`SelectChainForwarding`, and keep the costs those passes saved.

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
- **Branch-local storages.** A storage created under a runtime execution predicate `p` is
  invisible where `p` is false (it escapes only through a merge under `p`), so constant writes
  to it under the same `p` store `x` without the select (today's `predicate_exempt_segments`).
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
| **zero-write chain** | every write writes the constant 0 through one runtime axis (`scatter::write::ZeroWrites`) | products of `1 - h` | as today | as today |
| **RAM trace** | halo2-base; the tree is a chain whose reads respect the RAM discipline | `RamInit`, `RamWrite` (predicated as in 2.2), `RamRead` | ≈ 8 cells per access + 8 per slot (measured, `layer-memory.md` 1.2); **pre**: `Program{RAM}` | — |
| **native RAM** | Noir; same chain condition | same ops (Noir: `let mut` array, `array_index`) | — | measured per access/slot (1.2) |

Preconditions: every index's fact `0 <= i_k < len_k` (justified by the program's check, as for
element reads). Postconditions: the lanes; RAM also proves `0 <= a < len` for the flat address
(not the per-axis facts, which the program's checks keep).

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
  into one trace (addresses of array `t` offset by the sizes of the arrays before it); each
  array's accesses are in range by the memory group's precondition facts, and the trace proves
  every global address is in `[0, total)`. Producer cost: the trace's fixed columns
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
  halo2 (compiled, retired from proving): `Rom*`/`Ram*` mapped onto its select-based memory.

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
