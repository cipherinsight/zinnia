# Arrays as SSA values

Zinnia's IR represents an array as a value in static single assignment (SSA) form: a write does
not modify an array, it defines a new array value. How an array value is finally built in the
circuit (scalar selects, a lookup table, a RAM argument, a Noir native array) is decided later,
by the array-lowering layer (`src/lower/`), for each array as a whole. This document describes
the IR operations, how the IR generator produces them, and how the layer lowers them.

Code: `src/ir_array.rs` (operations and access maps), `src/ir_defs.rs` (the `IR` variants),
`src/builder/arrays.rs` (building the operations, with their folds), `src/builder/store.rs` (the
array store), `src/builder/stale.rs` (arrays NumPy would share and Zinnia copies),
`src/lower/memory/versions.rs` (lowering of written arrays). Tests:
`testing/lang/test_array_versions.py`, `testing/lang/test_array_ssa_review*.py`,
`src/prove/mock/array_tests.rs`, `src/builder/arrays.rs` (unit tests).

## 1. Terms

- **Cell**: one scalar element of an array. In the IR it is the statement whose value it is, or a
  constant.
- **Array version**: an IR statement whose value is a whole vector of `len` cells (`ArrayInit` or
  `ArrayWrite`).
- **Version tree**: an `ArrayInit`, every version made from it by writes, and the reads of all of
  them.
- **Access map** (`ArrayAccess`): how a read or write names its cell. It names
  `offset + Σ_k i_k · stride_k` for runtime positions `i_k`, each in `[0, len_k)` (one
  `AccessAxis { len, stride }` per runtime index). Compile-time indices are folded into
  `offset`. `ArrayAccess::new` checks that every in-range position names a cell of the array.
- **Position**: an index after NumPy's negative-index wrap (`i + n` for a negative `i`), so in
  `[0, n)` whenever NumPy accepts the index.
- **Storage**: the compile-time identity of the cells that a set of Python names can reach. An
  array and every view of it share one storage. Storages exist only in the IR generator; the
  frontend code still calls a storage id `segment_id` in many places.
- **Overlay**: the representation of writes as scalar selects,
  `cell' = select(hit ∧ p, written, cell)` for every cell a write may reach.
- **ROM / RAM**: a read-only table (halo2-base dynamic lookup, Noir native array) and a
  read-write memory with a consistency argument (halo2-base RAM trace, Noir mutable array).

## 2. IR operations

### 2.1 Before lowering (IR generator to layer)

```
v  = ArrayInit  { len, elem }   (c_0, ..., c_{len-1})
v' = ArrayWrite (access)        (v, i_0, ..., i_{r-1}, x, p)
y  = ArrayRead  (access)        (v, i_0, ..., i_{r-1})
```

`elem` (`ArrayElem`) is `Int` or `Float`, the number type of the cells and of a read (floats are
Q32 fixed-point field elements, as elsewhere). Semantics, exact on field elements:

- `ArrayInit(c)` is the vector `c`.
- `ArrayRead(v, i)`: the cell `v[a]` at the address the access map names, if every `i_k` is in
  `[0, len_k)`. Otherwise there is no witness (no proof) on every backend and on the mock.
- `ArrayWrite(v, i, x, p)`: the same index rule. The result is `v` with `v[a] := x` when the
  predicate `p` is nonzero, and `v` itself when `p` is zero. `p` is the write's execution
  predicate (the constant `true` for a write that runs on every path).
- All three are pure: dead-code elimination removes them when unused, and duplicate-code
  elimination merges equal ones.

**Indices.** The IR generator passes positions: it applies NumPy's wrap, checks `-n <= i < n`
where the access runs, and guards the position where the code does not run
(`IRBuilder::guard_address`, `select(p, i, 0)`). So every index operand is in range on every path
of an accepted witness, and the "no witness" rule only rejects programs NumPy rejects. No array
operation is ever the only proof of an index bound.

**Multi-dimensional access.** Each runtime index keeps its own axis in the access map
(`a[i, :, j]` with runtime `i`, `j` has two axes). Constructions use the per-axis structure
(per-axis hit bits for an overlay) and compute a flat address only where they need one (RAM,
ROM). The layer accepts reads of a *written* version with at most one runtime position
(`versions::trees` panics otherwise); the store reads cells at several runtime positions as one
element read of the line of cells they can name, at a combined row-major index
(`IRBuilder::storage_read_positions`).

**Runtime-shaped arrays** (`DynamicNDArray`): the storage holds the capacity cells (the shape
envelope's bound). The runtime extent stays in the array's metadata as scalar statements, and
accesses use flat runtime addresses over the capacity (`storage_read_at`, `storage_write_at`).

### 2.2 After lowering (layer to backends)

```
t  = RomInit  { len, elem }   (c_0, ..., c_{len-1})   a read-only table
y  = RomRead  { len, elem }   (t, a)                  t[a]
m  = RamInit  { len, elem }   (c_0, ..., c_{len-1})   a read-write memory, first version
m' = RamWrite { len, elem }   (m, a, x)               m with m[a] := x
y  = RamRead  { len, elem }   (m, a)                  m[a]
```

- Addresses `a` are flat; an address outside `[0, len)` has no witness. These operations are
  pure as well.
- RAM discipline: every RAM version is consumed by at most one `RamWrite`, and every
  `RamRead(m, ·)` lies, in statement order, after `m`'s definition and before the `RamWrite` that
  consumes `m`. So statement order is a valid access order. The interpreter
  (`prove/interpreter.rs`) checks it for every backend; a violation is a compiler error.
- RAM writes are unconditional. The layer emits a predicated write as
  `RamWrite(m, a, select(p, x, RamRead(m, a)))`.
- No other array operation reaches a proving backend: `ArrayInit`, `ArrayWrite`, `ArrayRead` and
  `IR::ArrayOp` are rejected there as not lowered (`prove/traits.rs`).

### 2.3 Mock

The mock backend evaluates all eight operations: versions, tables and memories are vectors of
field elements. An `ArrayRead`, `ArrayWrite`, `RomRead`, `RamRead` or `RamWrite` with an index
outside its range marks the proof unsatisfied (no witness), not a synthesis error. Because the mock
interprets the operations before lowering too, the layer can be tested construction by
construction, and the IR can be compared with NumPy before and after lowering.

### 2.4 Folds (in the builder)

`IRBuilder::ir_array_read` and `ir_array_write` apply the operations' definitions where the
operands decide them, at IR generation and again whenever `ConstantFold` rebuilds the graph. With
`w = ArrayWrite(v, j, x, p)`:

1. `ArrayRead(ArrayInit(c), k)` with a constant address `k` is `c[k]`.
2. `ArrayRead(w, k)` is `ArrayRead(v, k)` when no cell `k` can name is one that `j` can name:
   two different constant addresses, or lines of cells that do not meet (`a[r, i]` and
   `a[s, j]` for rows `r != s`). This is decided for accesses with at most one runtime position
   each, which name arithmetic progressions of cells (`ArrayAccess::disjoint_from`).
3. `ArrayRead(w, j)` at the same cell (the same constant address, or the same access map at the
   same index statements) is `x` when `p` is the constant true. Under a runtime `p` the read
   stays: rewriting it to `select(p, x, ArrayRead(v, j))` would read an older version after the
   write, which a single memory (the RAM constructions) cannot serve.
4. `ArrayWrite(v, j, x, false)` is `v`.

A read of a constant address outside the array is never folded, so both sides of every fold have
no witness for the same out-of-range positions. Two rules make the folds apply often. A read and
a write through `a[i]` under one execution predicate use one checked position statement
(`IRBuilder::checked_position`). And the optimizer runs `ConstantFold` again after
`DuplicateCodeElimination` (`lib.rs`), which makes equal index expressions written twice one
statement, followed by dead-code elimination of the writes no read reaches.

### 2.5 Ranges

The range engine (`optim/range.rs`) bounds a version by the hull of the cells it can hold:
`hull(ArrayInit(c))` is the union of the cells' intervals, `hull(ArrayWrite(v, ., x, .))` is
`hull(v) ∪ interval(x)`, and a read (`ArrayRead`, `RomRead`, `RamRead`) gets the interval of the
version it reads. A read the store builds from a storage of booleans is recorded as a boolean
cell. Array facts (`docs/array-facts.md`) give finer ranges per segment of an array.

### 2.6 Branch merges

There is no array-valued select. A runtime `if` never merges two versions of one storage: writes
inside a branch are `ArrayWrite`s predicated by the branch's path condition, applied to the one
current version. A variable bound to different arrays in the two arms is merged cell by cell
(`helpers::value_ops::select_value`) into a new storage.

## 3. The IR generator: storages and the store

The builder owns one array store (`builder/store.rs`): a map from storage id to `Storage`. A
storage is typed (integer or float cells; a value is converted to the storage's type when it is
stored) and is in one of two states:

- `Cells(c)`: the compiler knows every cell's value (a statement or a constant). No IR exists for
  the array.
- `Version { version, reads }`: after a write at a runtime position, the cells are those of the
  IR version `version`. The reads of constant addresses built so far are memoized.

Every access goes through the store, which is also the only place the write predicate is applied:

| access | `Cells(c)` | `Version(v)` |
|---|---|---|
| read a constant address `a` (`storage_read`) | `c[a]` (no IR) | `ArrayRead(v, a)`, memoized |
| read at a runtime address `i` (`storage_read_at`, `storage_read_positions`) | `ArrayRead(ArrayInit(c), i)` | `ArrayRead(v, i)` |
| write `x` at a constant address under `p` (`storage_write`) | `c[a] := select(p, x, c[a])` (`x` when `p` is true) | `ArrayWrite(v, a, x, p)` |
| write `x` at a runtime address under `p` (`storage_write_at`, `storage_write_positions`) | `ArrayWrite(ArrayInit(c), i, x, p)`; the storage becomes `Version` | `ArrayWrite(v, i, x, p)` |

Consequences:

- **One copy of the cells.** The store is the only record of an array's contents, so no cached
  copy can disagree with the IR.
- **Predicated writes.** Every write under a runtime predicate keeps the predicate, whatever
  storage it writes (`storage_write_predicate`). A storage created under `p` is not invisible where
  `p` is false: rebinding a name under `p` merges `select(p, new, old)` into a new storage that
  holds the old cells where `p` is false, so a later write under `p` must keep `p`.
- **Constant accesses stay compile-time** while no runtime write has happened (`Cells`). After
  one, they are IR reads of the version, which the folds of 2.4 and the layer resolve.
- **Reads of known cells.** Gathers of known cells at a checked index (`read_cells`, scan
  results, sorted outputs) emit `ArrayRead(ArrayInit(line), index)`, with one `ArrayInit` per
  distinct line of cells (`IRBuilder::array_of_cells`), so the reads of one line form one element
  group in the layer.
- **Loops** are unrolled. A write in a loop body is one `ArrayWrite` per iteration on the current
  version, predicated by the iteration's activity.
- **In-place outputs.** An array argument of an in-place kernel is an output exactly when its
  storage was written after circuit entry, by the store's write clock (`storage_write_clock`,
  `storage_written_since`, used in `ir_gen/outputs.rs`). Its output is the storage's final cells.
- **Complex arrays.** A static complex array is a pair of float storages, real and imaginary
  parts at the same addresses (`pair_complex_storages`). A runtime-shaped complex array is one
  float storage of interleaved (real, imaginary) pairs (`helpers/segment.rs`).

### 3.1 Views, copies and stale arrays

A static array value (`Value::StaticArray`) is a storage id with a shape, strides and an offset.
NumPy views whose layout is a compile-time constant (basic slices, `x[i]` of a 2-D or higher
array, transposes, a `reshape` or `ravel` that needs no copy, aliases) share the storage, and every access
goes through the storage's absolute address, so writes through a view change the base as in
NumPy. Copies (`x.copy()`, `flatten`, `np.array(x)`, every operation that builds a new array)
create a new storage in `Cells` state, sharing the cell values, which costs nothing.

Some results that NumPy returns as views are copies in Zinnia: views whose layout depends on a
private index, reversed or runtime-bounded slices, `np.flip`, `np.rot90`, a reshape that copies a
non-contiguous array, and a name bound to one of two arrays depending on a runtime condition. The
IR generator links each such copy to the array it came from (`ir_gen/detached_copies.rs`,
`IRBuilder::link_storages`). A write to either storage marks the other *stale* on the paths where
the write runs (`builder/stale.rs`), and every read of a stale storage's cells, through any syntax,
is a compile error. A merge (`select_value`) reads its operands without the check and carries
their staleness to its result, so a merged stale value is rejected only when it is read. Results
NumPy makes read-only (`np.broadcast_to`, the imaginary part of a real array) are read-only
storages, and a write to one is a compile error.

Python lists are values. Mutating a list reachable through two names is rejected
(`ir_gen/list_sharing.rs`); an array held in a list keeps its storage.

## 4. The layer

### 4.1 Groups

`ArrayLowering` lowers in rounds until no placeholder (`IR::ArrayOp`), `ArrayInit`, `ArrayWrite` or
`ArrayRead` is left, at most `MAX_ROUNDS` (4) rounds; more is a compiler error
(`lower/emit.rs`). An emitter may emit `ArrayRead`s of an `ArrayInit` that it leaves to the next
round (an overlay's runtime reads, a sort's reads at hinted ranks). Array operations form these
groups (`lower/graph.rs`):

- **Memory group** (`OpKind::Versions`): a version tree with at least one write and one read of a
  written version. Its lanes are the reads of written versions at constant addresses.
- **Element group** (`OpKind::Element`): the reads at a runtime position of one line of cells.
  Either the cells of an `ArrayInit` (the group's shared operands) or cells of a written version
  (`Group::version`), which the version's memory group must produce (the shared value
  `CELLS`, "memory: cells of a version"). Each lane is one read; its own operand is its index.
- Every `IR::ArrayOp` placeholder kind (sort, scan, wrap, and the others in `lower/op.rs`), as
  before.

### 4.2 Constructions of a memory group (`lower/memory/versions.rs`)

Each construction decides the representation of the whole version tree, with all its reads and
writes, as one candidate of the layer's selection:

| construction (`Construction::name`) | applies | emits | cost estimate |
|---|---|---|---|
| `Overlay` ("memory: overlay of selects") | always | per write, per cell its access can reach (the axis narrowed by the position's proven range), `select(hit ∧ p, x, cell)`, with hits per axis: equality tests shared per (position, coordinate), or the one-hot encoding of a `Wrap` group when the position is a wrap lane | one select per reached cell (one product when `x` is the constant 0) plus the conjunctions, and one equality test per tested (position, coordinate) |
| `ZeroWrites` ("memory: zero writes by keep products") | every write stores the constant 0 through one runtime position under a constant true predicate (`a[rows, :] = 0`) | per cell `x · Π keep`, one keep bit `Π (1 - hit)` per (axis, coordinate), shared by the cells there | halo2-base 4 cells, Noir 1 gate per product |
| `RamTrace` ("memory: RAM trace") | halo2-base; the versions form one chain read in program order (`Tree::is_chain_in_order`) | `RamInit`, `RamWrite` (predicated as in 2.2), `RamRead` | about 8 cells per initial row and per access, plus a select per predicated write; precondition: the program-wide RAM trace |
| `NativeRam` ("memory: native RAM") | Noir; the same chain condition | the same operations, emitted by the Noir backend as a mutable native array (bb's RAM block) | `3n` gates plus about `3.5n` per access over `n` cells |

The chain condition allows a read of an older version when no later write before it can change
the read's cell (fold 2 may leave such a read). The RAM emitter asserts that it serves reads in
statement order and that no write it is past changes a read's cell.

Preconditions are only the overlay's hit bits (the `Wrap` encodings it uses). Every position is
in its axis on every path of an accepted witness by the IR generator's contract (2.1), so no
construction lists per-axis facts. The overlay and zero-writes constructions produce the cells of
every version read at a runtime position (`CELLS`), which the element groups then read with any
element construction, and the reads at constant addresses. The RAM constructions produce every
read of the tree directly, including the lanes of the element groups over its versions.

Soundness. The overlay uses no hint: with every position in its axis, exactly the cell the access
names has all its hit bits set, so each new cell is NumPy's result of the write when `p` holds and
the old cell when it does not. Zero writes: a cell's keep bits are all 1 exactly when no write
hits it. RAM trace (`prove/halo2_base/circuit/ram.rs`): the shuffle makes the sorted trace a
permutation of the program-order trace, and the sorted-trace gates make each read equal to the
latest earlier write at its address (or the initial row); a read's value is a hint that argument
pins. Noir native RAM: ACIR memory semantics plus the bound check of `array_index`.

### 4.3 The program-wide RAM trace

On halo2-base, all memories of a program share one RAM trace
(`prove/halo2_base/synthesizer/ram.rs`): memory `t` occupies the addresses
`[base_t, base_t + n_t)`. The trace's fixed columns and arguments are a program-wide requirement
(`lower::challenge::RAM`, `Requirement::Program { name: "RAM trace" }`), paid once by the
selection, like the verifier challenge (`CHALLENGE`). The trace proves only that every global
address is in `[0, N)`, which would let an address outside memory `t` reach another memory's
cells, so each access at a runtime address also asserts its own bound `a < n_t` (free when the
program's index check already proved it). Noir has no program-wide memory requirement: its native
ROM and RAM are priced per array.

### 4.4 Element groups

Element groups have the gather constructions of `lower/gather/` and `lower/memory/`: select
chains, one-hot and indicator inner products, index bits (`ElementByIndexBits`),
`ElementOfWrite`, and `ElementByTable`, which emits
`RomInit`/`RomRead` (a dynamic-lookup table on halo2-base, a native array read on Noir). A table
read's cost includes a share of the table groups' fixed columns, amortized over the groups
offered a table read (`memory/constructions.rs`). `ConstantTableReads` and `ArgmaxQueries`
(`lower/sort/{table,argmax}.rs`) recognize `ArrayRead`s of an `ArrayInit` of constants before
the layer runs.

## 5. Known limitations

- **Operations on padding cells.** An element-wise operation on a runtime-shaped array is
  computed on every capacity cell, padding included, and an operator's precondition is asserted
  on every one. So `1.0 / x[x > 0]`, `np.log(x[x > 0])` or `100 // y` with `y = x[x > 0]` compile,
  but the honest witness has no proof when a padding cell violates the precondition (a padding
  cell is 0). This is sound (no wrong output verifies) but incomplete. The fix is to compute the
  lanes under their validity (`lane < length`) as the execution predicate, while keeping each
  padding lane's result pinned (for example `select(valid, r, 0)`), because padding results are
  public values.
- **Per-axis facts from RAM.** The RAM constructions state no per-axis bounds as
  postconditions. The backends' per-access bound is on the flat address, which does not imply
  each position's own bound (in a 4x4 array `a[1, 5]` has the in-range flat address 9), so such a
  fact would let the selection drop the program's per-axis check.
- **Lookup tables** are not a program-wide requirement: each `ElementByTable` candidate prices a
  share of the table columns itself (4.4).
