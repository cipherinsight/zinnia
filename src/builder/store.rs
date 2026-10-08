//! The array store: what the frontend knows about every array's cells
//! (`docs/array-ssa.md`, section 3).
//!
//! A **storage** is the compile-time identity of the cells some Python names
//! reach: an array and every view of it share one (views address it through
//! their offset and strides). It never reaches the IR. Its state is either
//!
//! * `Cells`: the compiler knows every cell's value (a statement or a
//!   constant); no IR exists for the array; or
//! * `Version`: after a write at a runtime position, the cells are those of
//!   an IR array version (`IR::ArrayWrite`, `crate::ir_array`), read with
//!   `ArrayRead`s.
//!
//! Every access goes through the functions below, which are the only place
//! the write predicate (a write in a branch not taken changes nothing) is
//! applied:
//!
//! | access | `Cells(c)` | `Version(v)` |
//! |---|---|---|
//! | read a constant address `a` | `c[a]` (no IR) | `ArrayRead(v, a)`, memoized |
//! | read at a runtime address `i` | `ArrayRead(ArrayInit(c), i)` | `ArrayRead(v, i)` |
//! | write `x` at a constant address under `p` | `c[a] := select(p, x, c[a])` | `ArrayWrite(v, a, x, p)` |
//! | write `x` at a runtime address under `p` | `ArrayWrite(ArrayInit(c), i, x, p)` → `Version` | `ArrayWrite(v, i, x, p)` |
//!
//! A storage is typed (integer or float cells; a complex array is a float
//! storage of (real, imaginary) pairs): a value is converted to the
//! storage's type when it is stored, so a read returns a value of the
//! array's type.

use std::collections::{BTreeMap, HashMap};

use crate::ir_array::{AccessAxis, ArrayAccess, ArrayElem};
use crate::types::{NumberType, Value};

use super::IRBuilder;

/// One array's cells.
pub(crate) struct Storage {
    elem: ArrayElem,
    state: State,
    /// The store's write clock at its last write (0: never written).
    last_write: u64,
    /// For the real-part storage of a static complex array, the storage of
    /// its imaginary parts (same addresses).
    imag: Option<u32>,
    /// For the imaginary-part storage of a static complex array, its
    /// real-part storage (the array's identity).
    pub(super) real: Option<u32>,
    /// Where the cells no longer hold what NumPy would show
    /// (`builder/stale.rs`).
    pub(super) stale: super::stale::Staleness,
    /// Where the array's shape itself is not the one NumPy would have (a
    /// merge of arrays of different shapes): neither its cells nor its
    /// metadata can be read there.
    pub(super) unshaped: super::stale::Staleness,
    /// Why the array is read-only, when NumPy makes it so (`np.broadcast_to`,
    /// the `imag` of a real array): a write to it is a compile error.
    pub(super) read_only: Option<&'static str>,
}

enum State {
    Cells(Vec<Value>),
    Version {
        version: Value,
        len: usize,
        /// The reads at constant addresses built for `version`.
        reads: HashMap<usize, Value>,
    },
}

/// The storages of one compilation, by id (`BTreeMap`: ids are allocated in
/// order, and nothing iterates it into the IR).
#[derive(Default)]
pub(crate) struct ArrayStore {
    storages: BTreeMap<u32, Storage>,
    next_id: u32,
    /// Writes so far (every write to any storage ticks it).
    write_clock: u64,
    /// Detached pairs: storages Python would share and Zinnia keeps apart
    /// (`builder/stale.rs`).
    pub(super) links: Vec<super::stale::Link>,
    /// Reads of stale storages allowed (inside `IRBuilder::with_stale_reads`).
    pub(super) stale_reads_allowed: u32,
}

/// The element type of a storage of `dtype` cells.
fn elem_of(dtype: NumberType) -> ArrayElem {
    match dtype {
        NumberType::Float | NumberType::Complex => ArrayElem::Float,
        NumberType::Integer => ArrayElem::Int,
    }
}

/// The complex value of a real and an imaginary cell.
fn complex_cell(re: &Value, im: &Value) -> Value {
    match (re, im) {
        (Value::Float(real), Value::Float(imag)) => Value::Complex { real: real.clone(), imag: imag.clone() },
        _ => panic!("the cells of a complex array are floats"),
    }
}

fn constant_truth(p: &Value) -> Option<bool> {
    p.bool_val().or_else(|| p.int_val().map(|v| v != 0))
}

impl IRBuilder {
    /// A new storage holding `cells`, converted to `dtype` (no IR unless a
    /// conversion needs one). Returns its id.
    pub fn alloc_storage(&mut self, cells: Vec<Value>, dtype: NumberType) -> u32 {
        self.new_storage(cells, dtype, false)
    }

    /// A new storage of `len` cells holding `fill` (converted to `dtype`).
    pub fn alloc_filled(&mut self, len: usize, fill: &Value, dtype: NumberType) -> u32 {
        let fill = self.ensure_ptr(fill);
        let zero = fill.int_val() == Some(0) || fill.float_val() == Some(0.0);
        self.new_storage(vec![fill; len], dtype, zero)
    }

    fn new_storage(&mut self, cells: Vec<Value>, dtype: NumberType, zero_filled: bool) -> u32 {
        let elem = elem_of(dtype);
        let cells: Vec<Value> = cells.into_iter().map(|c| self.stored_as(elem, &c)).collect();
        let id = self.store.next_id;
        self.store.next_id += 1;
        self.track_storage_alloc(id, &cells, zero_filled);
        self.store.storages.insert(id, Storage { elem, state: State::Cells(cells), last_write: 0, imag: None, real: None, stale: Default::default(), unshaped: Default::default(), read_only: None });
        id
    }

    /// Pair the real-part and imaginary-part storages of a static complex
    /// array: the known cells of `real` are then complex values.
    pub fn pair_complex_storages(&mut self, real: u32, imag: u32) {
        assert_eq!(self.storage_len(real), self.storage_len(imag), "complex parts of one length");
        self.storage_mut(real).imag = Some(imag);
        self.storage_mut(imag).real = Some(real);
    }

    /// The imaginary-part storage paired with `storage` (a static complex
    /// array's real-part storage), if any.
    pub fn imag_storage(&self, storage: u32) -> Option<u32> {
        self.storage(storage).imag
    }

    /// Every cell of `storage` when the compiler knows them all (no write at
    /// a runtime position happened), with no IR; `None` otherwise. The cells
    /// of a complex pair's real storage are complex values.
    pub fn known_cells(&mut self, storage: u32) -> Option<Vec<Value>> {
        self.check_readable(storage);
        self.force_pending_cells(storage);
        let State::Cells(cells) = &self.storage(storage).state else { return None };
        let cells = cells.clone();
        let Some(imag) = self.storage(storage).imag else { return Some(cells) };
        let State::Cells(imags) = &self.storage(imag).state else { return None };
        Some(cells.iter().zip(imags).map(|(re, im)| complex_cell(re, im)).collect())
    }

    /// The cell at `address` when the compiler knows the storage's cells
    /// (see [`Self::known_cells`]); `None` otherwise or past the end.
    pub fn known_cell(&mut self, storage: u32, address: usize) -> Option<Value> {
        self.check_readable(storage);
        self.force_pending_cells(storage);
        let State::Cells(cells) = &self.storage(storage).state else { return None };
        let cell = cells.get(address)?.clone();
        let Some(imag) = self.storage(storage).imag else { return Some(cell) };
        let State::Cells(imags) = &self.storage(imag).state else { return None };
        Some(complex_cell(&cell, imags.get(address)?))
    }

    /// [`Self::known_cell`] without computing deferred cells (`None` for a
    /// storage whose cells are a deferred product), for a reader that cannot
    /// build IR. Not complex-aware: the real part's cell.
    pub fn known_cell_now(&self, storage: u32, address: usize) -> Option<&Value> {
        // A stale storage's cells are read through the checked readers.
        let owner = self.storage(self.storage_owner(storage));
        if !owner.stale.is_empty() || !owner.unshaped.is_empty() {
            return None;
        }
        if self.static_array_product_provenance.contains_key(&storage) {
            return None;
        }
        let State::Cells(cells) = &self.storage(storage).state else { return None };
        cells.get(address)
    }

    /// Whether the compiler knows every cell of `storage`.
    pub fn knows_cells(&mut self, storage: u32) -> bool {
        self.force_pending_cells(storage);
        let known = |s: &Storage| matches!(s.state, State::Cells(_));
        known(self.storage(storage)) && self.storage(storage).imag.is_none_or(|imag| known(self.storage(imag)))
    }

    /// Replace each known cell `c` of `storage` by `f(c)`, another reading
    /// of the same value (e.g. the same truth value as a NumPy bool): not a
    /// write. Does nothing when the cells are not known.
    pub(crate) fn reread_known_cells(&mut self, storage: u32, mut f: impl FnMut(&mut IRBuilder, &Value) -> Value) {
        let State::Cells(cells) = &self.storage(storage).state else { return };
        let cells = cells.clone();
        let cells: Vec<Value> = cells.iter().map(|c| f(self, c)).collect();
        self.storage_mut(storage).state = State::Cells(cells);
    }

    /// The number of cells of a storage.
    pub fn storage_len(&self, storage: u32) -> usize {
        match &self.storage(storage).state {
            State::Cells(cells) => cells.len(),
            State::Version { len, .. } => *len,
        }
    }

    /// The cell at the constant `address`.
    ///
    /// # Panics
    /// When the address is outside the storage (a compiler error: the
    /// frontend reads constant addresses it computed from the shape).
    pub fn storage_read(&mut self, storage: u32, address: usize) -> Value {
        self.check_readable(storage);
        self.read_cell(storage, address)
    }

    /// [`Self::storage_read`] without the stale check: for a write's own
    /// read of the cell it changes.
    fn read_cell(&mut self, storage: u32, address: usize) -> Value {
        self.force_pending_cells(storage);
        let len = self.storage_len(storage);
        assert!(address < len, "a read of cell {address} of an array of {len} cells");
        let elem = self.storage(storage).elem;
        let (version, read) = match &self.storage(storage).state {
            State::Cells(cells) => {
                let cell = cells[address].clone();
                return self.track_storage_read(storage, cell);
            }
            State::Version { version, reads, .. } => (version.clone(), reads.get(&address).cloned()),
        };
        if let Some(read) = read {
            return read;
        }
        let access = ArrayAccess::constant(len as u32, elem, address as u32);
        let read = self.ir_array_read(&version, access, &[]);
        let read = self.note_storage_read(storage, read);
        if let State::Version { reads, .. } = &mut self.storage_mut(storage).state {
            reads.insert(address, read.clone());
        }
        read
    }

    /// The cell at the runtime flat `address` (a position over the whole
    /// storage, guarded into range where the code does not run). The caller
    /// has checked it where the read runs.
    pub fn storage_read_at(&mut self, storage: u32, address: &Value) -> Value {
        self.check_readable(storage);
        if let Some(a) = address.int_val() {
            return self.storage_read(storage, usize::try_from(a).expect("a non-negative address"));
        }
        let len = self.storage_len(storage);
        let address = self.guard_address(address, Some(len as u32));
        let elem = self.storage(storage).elem;
        let version = self.storage_version(storage);
        let read = self.ir_array_read(&version, ArrayAccess::flat(len as u32, elem), &[address]);
        self.note_storage_read(storage, read)
    }

    /// The cell `offset + Σ_k positions[k] · stride_k` for `axes[k] = (len_k,
    /// stride_k)`, each position in `[0, len_k)` wherever the code runs (the
    /// caller checked it) and everywhere else (guarded). Constant positions
    /// fold into the offset; the runtime ones keep their axes:
    ///
    /// * one runtime position: a read along its axis (`ArrayRead` of the
    ///   current version, or of the `ArrayInit` of that line of known cells);
    /// * several: a read of the line of cells they can name at their
    ///   row-major combined index (the known cells, or the version's cells at
    ///   constant addresses).
    pub fn storage_read_positions(&mut self, storage: u32, offset: usize, axes: &[(usize, usize)], positions: &[Value]) -> Value {
        self.check_readable(storage);
        self.force_pending_cells(storage);
        assert_eq!(axes.len(), positions.len(), "one position per axis");
        let mut base = offset;
        let mut runtime: Vec<((usize, usize), Value)> = Vec::new();
        for (&axis, p) in axes.iter().zip(positions) {
            match p.int_val() {
                Some(c) => base += usize::try_from(c).expect("a checked position") * axis.1,
                None => runtime.push((axis, p.clone())),
            }
        }
        if runtime.is_empty() {
            return self.storage_read(storage, base);
        }
        // A position whose proven range is narrower than its axis reads only
        // the cells that range reaches.
        for ((n, stride), position) in runtime.iter_mut() {
            let (lo, hi) = self.position_range(position, *n);
            if lo > 0 {
                let shift = self.ir_constant_int(lo as i64);
                *position = self.ir_sub_i(position, &shift);
                base += lo * *stride;
            }
            *n = hi - lo + 1;
        }
        let len = self.storage_len(storage);
        let elem = self.storage(storage).elem;
        let access = ArrayAccess::new(
            len as u32,
            elem,
            base as u32,
            runtime.iter().map(|&((n, stride), _)| AccessAxis { len: n as u32, stride: stride as u32 }).collect(),
        );
        if let (State::Version { version, .. }, [(_, position)]) = (&self.storage(storage).state, runtime.as_slice()) {
            let version = version.clone();
            let read = self.ir_array_read(&version, access, std::slice::from_ref(position));
            return self.note_storage_read(storage, read);
        }
        let line: Vec<Value> = access.line().into_iter().map(|a| self.storage_read(storage, a)).collect();
        let mut index = runtime[0].1.clone();
        for ((n, _), position) in &runtime[1..] {
            let extent = self.ir_constant_int(*n as i64);
            let scaled = self.ir_mul_i(&index, &extent);
            index = self.ir_add_i(&scaled, position);
        }
        let array = self.array_of_cells(&line, elem);
        let read = self.ir_array_read(&array, ArrayAccess::flat(line.len() as u32, elem), &[index]);
        self.note_storage_read(storage, read)
    }

    /// Store `value` at `offset + Σ_k positions[k] · stride_k` (positions as
    /// for [`Self::storage_read_positions`]), under the write predicate.
    pub fn storage_write_positions(&mut self, storage: u32, offset: usize, axes: &[(usize, usize)], positions: &[Value], value: &Value) {
        self.storage_write_positions_if(storage, offset, axes, positions, value, None);
    }

    /// [`Self::storage_write_positions`] that happens only when `condition`
    /// (a bool) also holds.
    pub fn storage_write_positions_if(
        &mut self,
        storage: u32,
        offset: usize,
        axes: &[(usize, usize)],
        positions: &[Value],
        value: &Value,
        condition: Option<&Value>,
    ) {
        assert_eq!(axes.len(), positions.len(), "one position per axis");
        let mut base = offset;
        let mut runtime_axes: Vec<AccessAxis> = Vec::new();
        let mut runtime: Vec<Value> = Vec::new();
        for (&(n, stride), p) in axes.iter().zip(positions) {
            match p.int_val() {
                Some(c) => base += usize::try_from(c).expect("a checked position") * stride,
                None => {
                    runtime_axes.push(AccessAxis { len: n as u32, stride: stride as u32 });
                    runtime.push(p.clone());
                }
            }
        }
        if runtime.is_empty() {
            self.write_cell(storage, base, value, condition);
            return;
        }
        let Some(predicate) = self.write_condition(condition) else { return };
        let len = self.storage_len(storage);
        let elem = self.storage(storage).elem;
        let value = self.stored_as(elem, value);
        self.note_storage_write(storage, &value);
        let access = ArrayAccess::new(len as u32, elem, base as u32, runtime_axes);
        self.write_version(storage, access, &runtime, &value, predicate.as_ref());
    }

    /// Every cell, in address order.
    pub fn storage_cells(&mut self, storage: u32) -> Vec<Value> {
        (0..self.storage_len(storage)).map(|a| self.storage_read(storage, a)).collect()
    }

    /// Store `value` at the constant `address`, under the write predicate.
    /// Returns the value the cell holds afterwards (`select(p, value, old)`
    /// under a runtime predicate).
    ///
    /// # Panics
    /// When the address is outside the storage (a compiler error).
    pub fn storage_write(&mut self, storage: u32, address: usize, value: &Value) -> Value {
        self.write_cell(storage, address, value, None)
    }

    /// Store `value` at the constant `address` under the write predicate and
    /// `condition`; the cell's value afterwards.
    fn write_cell(&mut self, storage: u32, address: usize, value: &Value, condition: Option<&Value>) -> Value {
        self.force_pending_cells(storage);
        let len = self.storage_len(storage);
        assert!(address < len, "a write of cell {address} of an array of {len} cells");
        let Some(predicate) = self.write_condition(condition) else {
            return self.read_cell(storage, address);
        };
        let elem = self.storage(storage).elem;
        let value = self.stored_as(elem, value);
        self.note_storage_write(storage, &value);
        match &self.storage(storage).state {
            State::Cells(_) => {
                let stored = match &predicate {
                    None => value,
                    Some(p) => {
                        let old = self.read_cell(storage, address);
                        let stored = self.select_cell_value(elem, p, &value, &old);
                        self.derive_select_cell_facts(&stored, &value, &old);
                        stored
                    }
                };
                if let State::Cells(cells) = &mut self.storage_mut(storage).state {
                    cells[address] = stored.clone();
                }
                stored
            }
            State::Version { .. } => {
                let access = ArrayAccess::constant(len as u32, elem, address as u32);
                self.write_version(storage, access, &[], &value, predicate.as_ref());
                self.read_cell(storage, address)
            }
        }
    }

    /// Store `value` at the runtime flat `address` (guarded into range where
    /// the code does not run), under the write predicate. The caller has
    /// checked the address where the write runs.
    pub fn storage_write_at(&mut self, storage: u32, address: &Value, value: &Value) {
        if let Some(a) = address.int_val() {
            self.storage_write(storage, usize::try_from(a).expect("a non-negative address"), value);
            return;
        }
        let Some(predicate) = self.storage_write_predicate() else { return };
        let len = self.storage_len(storage);
        let address = self.guard_address(address, Some(len as u32));
        let elem = self.storage(storage).elem;
        let value = self.stored_as(elem, value);
        self.note_storage_write(storage, &value);
        self.write_version(storage, ArrayAccess::flat(len as u32, elem), &[address], &value, predicate.as_ref());
    }

    /// The store's write clock: how many writes happened so far.
    pub fn storage_write_clock(&self) -> u64 {
        self.store.write_clock
    }

    /// Whether `storage` was written after the clock read `clock`.
    pub fn storage_written_since(&self, storage: u32, clock: u64) -> bool {
        self.storage(storage).last_write > clock
    }

    /// Record a write to `storage` (every store write does; the static
    /// arrays' compile-time overlays, which update their cell cache first,
    /// call it directly).
    pub fn note_storage_written(&mut self, storage: u32) {
        self.store.write_clock += 1;
        let clock = self.store.write_clock;
        self.storage_mut(storage).last_write = clock;
        self.propagate_write_to_links(storage);
    }

    // ── Internals ──────────────────────────────────────────────────────

    /// The positions `[lo, hi]` a runtime position on an axis of `len` can
    /// take on every path: the axis, narrowed by the interval the range
    /// engine proves over the position's definition (no fact that holds
    /// only where some code runs: the position is the guarded one, in range
    /// everywhere).
    fn position_range(&self, position: &Value, len: usize) -> (usize, usize) {
        use crate::optim::Resolver;
        let mut range = crate::optim::RangeResolver::new();
        let last = len as i64 - 1;
        let lo = range.resolve_min_with_stmts(position, &self.stmts).unwrap_or(0).clamp(0, last);
        let hi = range.resolve_max_with_stmts(position, &self.stmts).unwrap_or(last).clamp(lo, last);
        (lo as usize, hi as usize)
    }

    /// Compute the cells of a storage whose contents are a deferred product
    /// (`IRBuilder::static_array_product_provenance`): the storage holds
    /// them from then on. Not a write (the array was always those cells).
    fn force_pending_cells(&mut self, storage: u32) {
        let Some(provenance) = self.static_array_product_provenance.remove(&storage) else { return };
        let cells = crate::helpers::static_array::elementwise::product_provenance_payload_cells(self, &provenance);
        self.note_static_array_cells(storage, &cells);
        let elem = self.storage(storage).elem;
        let cells: Vec<Value> = cells.iter().map(|c| self.stored_as(elem, c)).collect();
        assert!(matches!(self.storage(storage).state, State::Cells(_)), "a deferred product is read before any write");
        self.storage_mut(storage).state = State::Cells(cells);
    }

    pub(super) fn storage(&self, storage: u32) -> &Storage {
        self.store.storages.get(&storage).unwrap_or_else(|| panic!("array storage {storage} does not exist"))
    }

    pub(super) fn storage_mut(&mut self, storage: u32) -> &mut Storage {
        self.store.storages.get_mut(&storage).unwrap_or_else(|| panic!("array storage {storage} does not exist"))
    }

    /// The predicate a write happens under: `None` (outer) when the write
    /// cannot happen, `Some(None)` unconditionally (no predicate, or a
    /// constant true one), `Some(Some(p))` under the runtime predicate `p`.
    ///
    /// Every write under a runtime predicate keeps it, whatever storage it
    /// writes. (A storage created under `p` is not invisible where `p` is
    /// false: a rebinding under `p` merges `select(p, new, old)` into a new
    /// storage, which holds the old cells there; skipping the predicate on
    /// it proved writes of a branch not taken, review finding F1.)
    fn storage_write_predicate(&self) -> Option<Option<Value>> {
        let Some(p) = self.write_predicate().cloned() else { return Some(None) };
        match constant_truth(&p) {
            Some(true) => Some(None),
            Some(false) => None,
            None => Some(Some(p)),
        }
    }

    /// [`Self::storage_write_predicate`] conjoined with `condition`.
    fn write_condition(&mut self, condition: Option<&Value>) -> Option<Option<Value>> {
        let predicate = self.storage_write_predicate()?;
        let Some(condition) = condition else { return Some(predicate) };
        match (constant_truth(condition), predicate) {
            (Some(false), _) => None,
            (Some(true), predicate) => Some(predicate),
            (None, None) => Some(Some(condition.clone())),
            (None, Some(p)) => Some(Some(self.ir_logical_and(&p, condition))),
        }
    }

    /// The storage's current version: its `ArrayInit` (memoized per cells)
    /// in `Cells` state.
    fn storage_version(&mut self, storage: u32) -> Value {
        self.force_pending_cells(storage);
        let elem = self.storage(storage).elem;
        match &self.storage(storage).state {
            State::Version { version, .. } => version.clone(),
            State::Cells(cells) => {
                let cells = cells.clone();
                self.array_of_cells(&cells, elem)
            }
        }
    }

    /// `ArrayWrite` of the storage's current version; the storage is then
    /// in `Version` state.
    fn write_version(&mut self, storage: u32, access: ArrayAccess, positions: &[Value], value: &Value, predicate: Option<&Value>) {
        let len = self.storage_len(storage);
        let version = self.storage_version(storage);
        let predicate = match predicate {
            Some(p) => p.clone(),
            None => self.ir_constant_bool(true),
        };
        let unconditional = constant_truth(&predicate) == Some(true);
        let written = self.ir_array_write(&version, access.clone(), positions, value, &predicate);
        self.derive_write_facts(&version, &written, &access, positions, value, unconditional);
        self.storage_mut(storage).state = State::Version { version: written, len, reads: HashMap::new() };
    }

    /// `value` as a cell of a storage of `elem`: an integer stored into a
    /// float array becomes its float (a constant, or a cast).
    fn stored_as(&mut self, elem: ArrayElem, value: &Value) -> Value {
        match (elem, value) {
            (ArrayElem::Float, Value::Integer(_) | Value::Boolean(_)) => self.ir_float_cast(value),
            _ => self.ensure_ptr(value),
        }
    }

    /// `select(p, x, old)` of two cells, a NumPy bool when both are.
    fn select_cell_value(&mut self, elem: ArrayElem, p: &Value, x: &Value, old: &Value) -> Value {
        match elem {
            ArrayElem::Float => self.ir_select_f(p, x, old),
            ArrayElem::Int => {
                let out = self.ir_select_i(p, x, old);
                if self.is_bool_cell(x) && self.is_bool_cell(old) {
                    self.record_bool_cell(&out);
                }
                out
            }
        }
    }

    /// What a write changes besides the cells: views and records derived
    /// from the old contents are dropped (provenance invariant #4), and a
    /// fixed-width array accepts only values of its type.
    fn note_storage_write(&mut self, storage: u32, value: &Value) {
        let owner = self.storage_owner(storage);
        if let Some(why) = self.storage(owner).read_only {
            panic!("assignment destination is read-only: {why}");
        }
        self.check_shape_known(owner);
        self.provenance.on_segment_write(storage);
        self.track_storage_write(storage, value);
        self.note_storage_written(storage);
    }

    /// A read the store built (`ArrayRead`): a NumPy bool when the array is
    /// a bool array, and of the array's fixed-width type. (What the array's
    /// facts say about it, the bound queries read from the read itself:
    /// `builder/array_facts.rs`.)
    fn note_storage_read(&mut self, storage: u32, read: Value) -> Value {
        if self.is_bool_segment(storage) {
            self.record_bool_cell(&read);
        }
        self.track_storage_read(storage, read)
    }

    // ── For the array facts (`builder/array_facts.rs`) ─────────────────

    /// What `storage` holds now, without building IR: its cells, or its
    /// version. `None` for a deferred product not computed yet and for a
    /// complex array (whose cells are pairs of storages).
    pub(super) fn storage_value(&self, storage: u32) -> Option<super::array_facts::StorageValue> {
        use super::array_facts::StorageValue;
        let s = self.storage(storage);
        if s.imag.is_some() || s.real.is_some() || self.static_array_product_provenance.contains_key(&storage) {
            return None;
        }
        match &s.state {
            State::Cells(cells) => Some(StorageValue::Cells(cells.clone())),
            State::Version { version, .. } => version.stmt_id().map(StorageValue::Version),
        }
    }

    /// Whether the cells of `storage` are what NumPy would show on every
    /// path: it is not stale and its shape is known (`builder/stale.rs`).
    pub(super) fn storage_is_current(&self, storage: u32) -> bool {
        let owner = self.storage(self.storage_owner(storage));
        owner.stale.is_empty() && owner.unshaped.is_empty()
    }

    /// Whether `storage` holds float cells.
    pub(super) fn storage_is_float(&self, storage: u32) -> bool {
        self.storage(storage).elem == ArrayElem::Float
    }

    /// Whether the array version `version` holds float cells.
    pub(super) fn version_is_float(&self, version: &Value) -> bool {
        use crate::ir_defs::IR;
        let Some(s) = version.stmt_id().and_then(|v| self.stmts.get(v as usize)) else { return false };
        match &s.ir {
            IR::ArrayInit { elem, .. } => *elem == ArrayElem::Float,
            IR::ArrayWrite(access) | IR::ArrayRead(access) => access.elem == ArrayElem::Float,
            _ => false,
        }
    }
}
