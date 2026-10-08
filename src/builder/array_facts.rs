//! The IR generator's side of array facts (`docs/array-facts.md`): stating
//! facts about the arrays it builds, deriving the facts of written and
//! merged arrays, and asking what holds for an array's elements.
//!
//! An array fact is about an **array value** (`optim::array_facts`), and
//! the store (`builder/store.rs`) holds every array in one of two states:
//!
//! * `Cells`: the frontend knows the cells as SSA values. The facts of the
//!   array are those of its cell list: the lists stated over runs of these
//!   cells, and each cell's own facts.
//! * `Version`: after a write at a runtime position the cells are those of
//!   an IR version. An `ArrayInit`'s facts are those of its cells; an
//!   `ArrayWrite`'s are derived from the version it writes by the write rule
//!   when the write is built ([`IRBuilder::derive_write_facts`]).
//!
//! Every query reads the array's *current* value through its storage, so a
//! write through any name of the array (an alias, a view: they share the
//! storage) gives the next query a new value, about which the old facts say
//! nothing. A storage linked to another that was written (a copy NumPy would
//! share, `builder/stale.rs`) is stale, and has no facts.
//!
//! **Where facts hold** ([`Where`]). A fact derived from what is known where
//! the current code runs (its execution predicate) holds there, and is
//! stated with the fact stack's current guard; one derived from what holds
//! on every path is stated for every path. [`IRBuilder::state_derived_facts`]
//! derives both. A query sees only the facts that hold in its context.

use std::collections::HashMap;

use crate::ir_array::{ArrayAccess, ReadOperands};
use crate::ir_defs::IR;
use crate::optim::array_facts::{Bound, Cell, ElemFacts, FactGuard, Order, Range, Scalars, Segmentation};
use crate::types::{NumberType, ScalarValue, StmtId, Value};

use super::IRBuilder;

/// Where a derived fact must hold, which decides what may be used to derive
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Where {
    /// On every path: constants, intervals over the IR, the facts that hold
    /// on every path, and the array facts stated for every path.
    Everywhere = 0,
    /// Where the code being lowered runs: everything the current execution
    /// guard makes visible.
    Here = 1,
}

/// The builder's array-fact state.
pub(crate) struct ArrayFactState {
    /// Array facts are off (`ZINNIA_ABLATE_ARRAY_FACTS=1`, read when the
    /// builder is made): every query sees none.
    off: bool,
    /// The stated facts (`optim::array_facts`).
    pub(crate) store: crate::optim::array_facts::ArrayFacts,
    /// Interval propagation over the IR for the solver-free tiers; separate
    /// from the layered resolver so asking it never reaches SMT. Behind a
    /// mutex (never contended) so queries can take `&self`.
    pub(crate) intervals: std::sync::Mutex<crate::optim::RangeResolver>,
    /// The facts of each `ArrayInit` read so far, per context: computed from
    /// its cells on first use (the cells of a statement never change, and
    /// facts are only added, so a remembered answer stays true).
    init_facts: std::sync::Mutex<HashMap<(StmtId, FactGuard), Vec<Segmentation>>>,
}

impl Default for ArrayFactState {
    fn default() -> Self {
        ArrayFactState {
            off: crate::optim::array_facts::ablated(),
            store: Default::default(),
            intervals: Default::default(),
            init_facts: Default::default(),
        }
    }
}

/// The builder as the oracle for the symbolic bounds of array facts (a
/// runtime length, a count): the solver-free scalar tiers, in one context.
pub(crate) struct Oracle<'a> {
    b: &'a IRBuilder,
    at: Where,
}

impl Scalars for Oracle<'_> {
    fn interval(&self, stmt: StmtId) -> Option<(i64, i64)> {
        let v = self.b.int_value_of_stmt(stmt)?;
        match self.b.scalar_bounds(&v, self.at) {
            (Some(min), Some(max)) if min <= max => Some((min, max)),
            _ => None,
        }
    }
}

/// The deepest chain of reads of arrays whose cells are reads that a range
/// query follows (each step looks at one array's cells).
const MAX_READ_DEPTH: u32 = 8;

thread_local! {
    static READ_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

impl IRBuilder {
    // ── Contexts ──────────────────────────────────────────────────────

    /// The guard the facts of a context must hold under.
    pub(crate) fn fact_context(&self, at: Where) -> FactGuard {
        match at {
            Where::Everywhere => FactGuard::Always,
            Where::Here => self.facts.guard().clone(),
        }
    }

    /// The oracle for symbolic bounds in context `at`.
    pub(crate) fn oracle(&self, at: Where) -> Oracle<'_> {
        Oracle { b: self, at }
    }

    /// An integer `Value` handle for the statement `stmt` (its constant when
    /// it is one).
    pub(crate) fn int_value_of_stmt(&self, stmt: StmtId) -> Option<Value> {
        let s = self.stmts.get(stmt as usize)?;
        let static_val = match s.ir {
            IR::ConstantInt { value } => Some(value),
            _ => None,
        };
        Some(Value::Integer(ScalarValue { static_val, stmt_id: Some(stmt), value_id: s.value_id }))
    }

    // ── Stating facts ─────────────────────────────────────────────────

    /// State `facts` about the cells of one array (row-major), where `guard`
    /// holds. Only scalar cells carry facts; a float cell carries order
    /// only (its range would be about its fixed-point encoding), so ranges
    /// and formulas are dropped from a list with one. The caller is the
    /// source and documents why the facts hold (the trust rule).
    pub fn state_array_facts(&mut self, cells: &[Value], facts: Segmentation, guard: FactGuard) {
        let Some(list) = Cell::list(cells) else { return };
        if list.len() != facts.len() {
            return;
        }
        let facts = if cells.iter().any(|c| matches!(c, Value::Float(_))) { facts.order_only() } else { facts };
        self.array_facts.store.state(list, facts, guard);
    }

    /// State facts that hold on every path by the definition of the cells
    /// (an operator's semantics, a check made at entry on every path).
    pub fn state_defined_facts(&mut self, cells: &[Value], facts: Segmentation) {
        self.state_array_facts(cells, facts, FactGuard::Always);
    }

    /// State facts derived from what is known about the operands:
    /// `derive(b, at)` computes them from what holds in context `at`. They
    /// are derived for every path and, where the code runs under a runtime
    /// guard, again for the paths where it runs (stated with that guard)
    /// when that gives more.
    pub fn state_derived_facts(
        &mut self,
        cells: &[Value],
        derive: impl Fn(&mut IRBuilder, Where) -> Option<Segmentation>,
    ) {
        let guard = self.facts.guard().clone();
        if guard == FactGuard::Never {
            return;
        }
        let everywhere = derive(self, Where::Everywhere);
        if let Some(facts) = everywhere.clone() {
            self.state_array_facts(cells, facts, FactGuard::Always);
        }
        if matches!(guard, FactGuard::When(_)) {
            if let Some(facts) = derive(self, Where::Here).filter(|f| Some(f) != everywhere.as_ref()) {
                self.state_array_facts(cells, facts, guard);
            }
        }
    }

    /// State facts about the current contents of `storage` (all its cells,
    /// in address order), as [`Self::state_array_facts`] does for its cell
    /// list, or for its version after a write at a runtime position.
    pub fn state_storage_facts(&mut self, storage: u32, facts: Segmentation, guard: FactGuard) {
        if self.storage_len(storage) != facts.len() {
            return;
        }
        match self.storage_value(storage) {
            Some(StorageValue::Cells(cells)) => self.state_array_facts(&cells, facts, guard),
            Some(StorageValue::Version(version)) => {
                let facts = if self.storage_is_float(storage) { facts.order_only() } else { facts };
                self.array_facts.store.state_version(version, facts, guard)
            }
            None => {}
        }
    }

    /// [`Self::state_derived_facts`] for the current contents of `storage`.
    pub fn state_derived_storage_facts(
        &mut self,
        storage: u32,
        derive: impl Fn(&mut IRBuilder, Where) -> Option<Segmentation>,
    ) {
        let guard = self.facts.guard().clone();
        if guard == FactGuard::Never {
            return;
        }
        let everywhere = derive(self, Where::Everywhere);
        if let Some(facts) = everywhere.clone() {
            self.state_storage_facts(storage, facts, FactGuard::Always);
        }
        if matches!(guard, FactGuard::When(_)) {
            if let Some(facts) = derive(self, Where::Here).filter(|f| Some(f) != everywhere.as_ref()) {
                self.state_storage_facts(storage, facts, guard);
            }
        }
    }

    /// State that every cell of `out` is one of the cells of `sources` (a
    /// selection: `np.where`, a gather at checked indices), hence in their
    /// joined range. Integer cells only; nothing when a range is unknown.
    pub fn state_selected_from(&mut self, out: &Value, sources: &[&Value]) {
        let Some(cells) = self.leaf_cells(out) else { return };
        let sources: Vec<Value> = sources.iter().map(|s| (*s).clone()).collect();
        self.state_derived_facts(&cells, |b, at| {
            let mut range: Option<Range> = None;
            for source in &sources {
                let r = b.range_of(source, at);
                if r.is_top() {
                    return None;
                }
                range = Some(match range {
                    None => r,
                    Some(acc) => acc.join(&r, &b.oracle(at)),
                });
            }
            let range = range.filter(|r| !r.is_top())?;
            Some(Segmentation::uniform(cells.len(), ElemFacts::range(range)))
        });
    }

    // ── The write rule and merges ─────────────────────────────────────

    /// The facts of `written`, the version `ArrayWrite(old, access,
    /// positions, value, predicate)`, from `old`'s by the write rule: a
    /// write on every path at a constant address replaces that cell
    /// ([`Segmentation::strong_write`]); any other write may change any cell
    /// it can reach, and does so to `value` ([`Segmentation::weak_write`]).
    ///
    /// Why the result holds wherever `old`'s facts do: where the write does
    /// not happen the cells are `old`'s; where it does, every cell but the
    /// written one is `old`'s and the written one is `value`, whose range is
    /// taken in the write's own context (the write happens only where its
    /// predicate, hence the current guard, holds).
    pub(crate) fn derive_write_facts(
        &mut self,
        old: &Value,
        written: &Value,
        access: &ArrayAccess,
        positions: &[Value],
        value: &Value,
        unconditional: bool,
    ) {
        let (Some(old), Some(written_stmt)) = (old.stmt_id(), written.stmt_id()) else { return };
        let guard = self.facts.guard().clone();
        if guard == FactGuard::Never {
            return;
        }
        let contexts: &[Where] = if matches!(guard, FactGuard::When(_)) { &[Where::Everywhere, Where::Here] } else { &[Where::Everywhere] };
        let constant = access.constant_address().filter(|_| unconditional);
        let reach = self.access_reach(access, positions);
        let float = self.version_is_float(written);
        let mut stated: Vec<(Segmentation, FactGuard)> = Vec::new();
        for &at in contexts {
            let before = self.version_facts(old, at);
            if before.is_empty() {
                continue;
            }
            let range = if float { Range::TOP } else { self.value_range(value, at) };
            let oracle = self.oracle(at);
            for facts in &before {
                let after = match constant {
                    Some(a) => facts.strong_write(a, &range, &oracle),
                    None => facts.weak_write(&range, Some(reach), &oracle),
                };
                let after = if float { after.order_only() } else { after };
                if after.is_top() || stated.iter().any(|(f, _)| *f == after) {
                    continue;
                }
                let holds = if at == Where::Everywhere { FactGuard::Always } else { guard.clone() };
                stated.push((after, holds));
            }
        }
        for (facts, holds) in stated {
            self.array_facts.store.state_version(written_stmt, facts, holds);
        }
    }

    /// The facts of the cell `out = select(p, x, old)` that a write under a
    /// runtime predicate stores into a known cell: it is `x` or `old`, so it
    /// lies in the join of their ranges (stated when either has array facts;
    /// the intervals of the IR already join what they know).
    pub(crate) fn derive_select_cell_facts(&mut self, out: &Value, x: &Value, old: &Value) {
        let has_facts = |b: &IRBuilder, v: &Value| {
            Cell::of(v).is_some_and(|c| !b.array_fact_range(&c, v, Where::Here).is_top())
        };
        if !has_facts(self, x) && !has_facts(self, old) {
            return;
        }
        let (x, old) = (x.clone(), old.clone());
        self.state_derived_facts(std::slice::from_ref(out), |b, at| {
            let range = b.value_range(&x, at).join(&b.value_range(&old, at), &b.oracle(at));
            (!range.is_top()).then(|| Segmentation::uniform(1, ElemFacts::range(range)))
        });
    }

    /// The facts of `out`, a merge `select(cond, tv, fv)` of two arrays of
    /// one shape (a branch join): the merged array is `tv` or it is `fv`,
    /// as a whole, so what holds for both holds for it
    /// ([`Segmentation::either`]: their ranges' join, an order both have).
    pub fn state_merged_facts(&mut self, out: &Value, tv: &Value, fv: &Value) {
        let Some(cells) = self.leaf_cells(out) else { return };
        if self.stale_reason_in(tv).is_some() || self.stale_reason_in(fv).is_some() {
            return;
        }
        let (tv, fv) = (tv.clone(), fv.clone());
        self.state_derived_facts(&cells, |b, at| {
            let a = b.primary_facts(&tv, at)?;
            let c = b.primary_facts(&fv, at)?;
            (a.len() == c.len() && a.len() == cells.len()).then(|| a.either(&c, &b.oracle(at)))
        });
    }

    /// One segmentation summarizing what is known about `arr`'s elements in
    /// context `at`: the meet of all its facts (each holds of the same
    /// cells).
    fn primary_facts(&self, arr: &Value, at: Where) -> Option<Segmentation> {
        let facts = self.view_facts(arr, at);
        let oracle = self.oracle(at);
        let mut all = facts.iter();
        let first = all.next()?.clone();
        Some(all.fold(first, |acc, f| acc.meet(f, &oracle)))
    }

    // ── Querying ──────────────────────────────────────────────────────

    /// What the array facts say about the scalar `v` (a cell `cell`) in
    /// context `at`: the facts of the lists it is in, and, for an
    /// `ArrayRead`, the facts of the array it reads over the cells it can
    /// read. Symbolic ends are kept.
    pub(crate) fn array_fact_range(&self, cell: &Cell, v: &Value, at: Where) -> Range {
        if self.array_facts.off {
            return Range::TOP;
        }
        let context = self.fact_context(at);
        let oracle = self.oracle(at);
        let mut range = self.array_facts.store.cell_range(*cell, &oracle, &context);
        if let (Cell::Stmt(stmt), false) = (cell, matches!(v, Value::Float(_))) {
            for derived in [self.read_range(*stmt, at), self.shifted_range(*stmt, at)].into_iter().flatten() {
                range = range.meet(&derived, &oracle);
            }
        }
        range
    }

    /// One step of arithmetic over array facts: the range of `x + c`,
    /// `x - c`, `c - x` or `x * c` (`c` a constant) from the array facts of
    /// `x` (the interval engine sees `x`'s interval over the IR, not its
    /// array facts). Symbolic ends survive a shift. `None` for any other
    /// statement.
    fn shifted_range(&self, stmt: StmtId, at: Where) -> Option<Range> {
        use crate::optim::array_facts::transfer;
        let s = self.stmts.get(stmt as usize)?;
        if !matches!(s.ir, IR::AddI | IR::SubI | IR::MulI) || s.arguments.len() != 2 {
            return None;
        }
        let depth = READ_DEPTH.with(|d| d.get());
        if depth >= MAX_READ_DEPTH {
            return None;
        }
        let (a, b) = (self.int_value_of_stmt(s.arguments[0])?, self.int_value_of_stmt(s.arguments[1])?);
        let fact_range = |v: &Value| -> Range {
            match v.int_val() {
                Some(c) => Range::point(c),
                None => {
                    READ_DEPTH.with(|d| d.set(depth + 1));
                    let r = Cell::of(v).map_or(Range::TOP, |c| self.array_fact_range(&c, v, at));
                    READ_DEPTH.with(|d| d.set(depth));
                    r
                }
            }
        };
        if a.int_val().is_none() && b.int_val().is_none() {
            return None;
        }
        let (x, y) = (fact_range(&a), fact_range(&b));
        let oracle = self.oracle(at);
        let range = match s.ir {
            IR::AddI => transfer::add(&x, &y, &oracle),
            IR::SubI => transfer::sub(&x, &y, &oracle),
            _ => transfer::mul(&x, &y, &oracle),
        };
        (!range.is_top()).then_some(range)
    }

    /// The range of the `ArrayRead` statement `stmt` from the facts of the
    /// version it reads, over the addresses its positions can name (every
    /// position is in its axis on every accepted witness: an `ArrayRead`
    /// outside has no witness). `None` for any other statement.
    fn read_range(&self, stmt: StmtId, at: Where) -> Option<Range> {
        let s = self.stmts.get(stmt as usize)?;
        let IR::ArrayRead(access) = &s.ir else { return None };
        let depth = READ_DEPTH.with(|d| d.get());
        if depth >= MAX_READ_DEPTH {
            return None;
        }
        READ_DEPTH.with(|d| d.set(depth + 1));
        let operands = ReadOperands::split(&s.arguments, access.rank());
        let positions: Vec<Value> = operands.positions.iter().filter_map(|&p| self.int_value_of_stmt(p)).collect();
        let (from, to) = self.access_reach(access, &positions);
        let oracle = self.oracle(at);
        let range = self
            .version_facts(*operands.version, at)
            .iter()
            .fold(Range::TOP, |acc, facts| acc.meet(&facts.over(from, to, &oracle).range, &oracle));
        READ_DEPTH.with(|d| d.set(depth));
        (!range.is_top()).then_some(range)
    }

    /// The facts of the array version `version` in context `at`.
    fn version_facts(&self, version: StmtId, at: Where) -> Vec<Segmentation> {
        let Some(s) = self.stmts.get(version as usize) else { return Vec::new() };
        match &s.ir {
            IR::ArrayInit { .. } => {
                let key = (version, self.fact_context(at));
                if let Some(known) = self.array_facts.init_facts.lock().expect("array facts").get(&key) {
                    return known.clone();
                }
                let cells = self.array_defs.get(&version).cloned().unwrap_or_default();
                let facts = self.cell_list_facts(&cells, at);
                self.array_facts.init_facts.lock().expect("array facts").insert(key, facts.clone());
                facts
            }
            IR::ArrayWrite(_) => self
                .array_facts
                .store
                .version_facts(version, &self.fact_context(at))
                .into_iter()
                .cloned()
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The facts of a list of cells as an array of its own, in context `at`:
    /// those of the stated lists it is a run of (and of its constants), and
    /// the join of its cells' ranges. Empty when nothing is known beyond
    /// what the intervals of the IR give each cell.
    fn cell_list_facts(&self, cells: &[Value], at: Where) -> Vec<Segmentation> {
        let Some(list) = Cell::list(cells) else { return Vec::new() };
        let context = self.fact_context(at);
        let oracle = self.oracle(at);
        let mut out = self.array_facts.store.list_facts(&list, &oracle, &context);
        let floats = cells.iter().any(|c| matches!(c, Value::Float(_)));
        // The join of the cells' ranges, when some cell has a fact the
        // intervals of the IR do not see (otherwise the interval tier
        // already gives that join).
        let informed = list.iter().zip(cells).any(|(c, v)| match c {
            Cell::Stmt(_) => !self.array_fact_range(c, v, at).is_top() || self.has_scalar_facts(v, at),
            _ => false,
        });
        if informed && !floats && !cells.is_empty() {
            let mut hull: Option<Range> = None;
            for v in cells {
                let r = self.value_range(v, at);
                if r.is_top() {
                    hull = None;
                    break;
                }
                hull = Some(match hull {
                    None => r,
                    Some(h) => h.join(&r, &oracle),
                });
            }
            if let Some(h) = hull.filter(|h| !h.is_top()) {
                out.push(Segmentation::uniform(cells.len(), ElemFacts::range(h)));
            }
        }
        out
    }

    /// Whether the fact stack bounds `v` in context `at` (a cheap gate).
    fn has_scalar_facts(&self, v: &Value, at: Where) -> bool {
        let Some(vid) = v.value_id() else { return false };
        match at {
            Where::Everywhere => self.facts.unconditional_facts_for(vid).next().is_some(),
            Where::Here => self.facts.has_facts_for(vid),
        }
    }

    /// The addresses `[first, last]` an access can name: every position in
    /// its axis, narrowed by the interval the IR gives it (positions are
    /// guarded into range where the code does not run, so the interval
    /// holds on every path).
    fn access_reach(&self, access: &ArrayAccess, positions: &[Value]) -> (usize, usize) {
        let mut first = access.offset as usize;
        let mut last = access.offset as usize;
        let mut range = self.array_facts.intervals.lock().expect("array facts");
        for (axis, p) in access.axes.iter().zip(positions) {
            use crate::optim::Resolver;
            let top = axis.len as i64 - 1;
            let (lo, hi) = match p.int_val() {
                Some(c) => (c.clamp(0, top), c.clamp(0, top)),
                None => {
                    let lo = range.resolve_min_with_stmts(p, &self.stmts).unwrap_or(0).clamp(0, top);
                    let hi = range.resolve_max_with_stmts(p, &self.stmts).unwrap_or(top).clamp(lo, top);
                    (lo, hi)
                }
            };
            first += lo as usize * axis.stride as usize;
            last += hi as usize * axis.stride as usize;
        }
        (first, last)
    }

    /// The facts of the elements of the array `arr` (a static array or a
    /// view of one, or a runtime-shaped array), as an array of their own in
    /// row-major order, in context `at`. Empty when nothing is known, the
    /// array is complex, or its storage is stale.
    pub(crate) fn view_facts(&self, arr: &Value, at: Where) -> Vec<Segmentation> {
        if self.array_facts.off {
            return Vec::new();
        }
        match arr {
            Value::StaticArray { dtype, shape, segment_id, strides, offset, .. } => {
                if *dtype == NumberType::Complex || !self.storage_is_current(*segment_id) {
                    return Vec::new();
                }
                let n: usize = shape.iter().product();
                let addresses = view_addresses(shape, strides, *offset);
                match self.storage_value(*segment_id) {
                    Some(StorageValue::Cells(cells)) => {
                        let view: Option<Vec<Value>> = addresses.iter().map(|&a| cells.get(a).cloned()).collect();
                        view.map(|v| self.cell_list_facts(&v, at)).unwrap_or_default()
                    }
                    Some(StorageValue::Version(version)) => {
                        let len = self.storage_len(*segment_id);
                        let oracle = self.oracle(at);
                        self.version_facts(version, at)
                            .iter()
                            .map(|facts| restrict(facts, &addresses, len, n, &oracle))
                            .collect()
                    }
                    None => Vec::new(),
                }
            }
            Value::DynamicNDArray(d) => {
                if d.dtype == NumberType::Complex || !self.storage_is_current(d.segment_id) {
                    return Vec::new();
                }
                let len = self.storage_len(d.segment_id);
                let whole = d.meta.logical_offset == 0
                    && d.meta.runtime_offset.static_val == Some(0)
                    && d.meta.logical_shape.len() == 1
                    && d.meta.logical_strides == [1]
                    && d.max_length() == len;
                let facts = match self.storage_value(d.segment_id) {
                    Some(StorageValue::Cells(cells)) => self.cell_list_facts(&cells, at),
                    Some(StorageValue::Version(version)) => self.version_facts(version, at),
                    None => Vec::new(),
                };
                if whole {
                    return facts;
                }
                // A view into the storage: its cells are some of the
                // storage's, so what holds for all of them holds for it.
                let oracle = self.oracle(at);
                let n = d.max_length();
                facts.iter().map(|f| Segmentation::uniform(n, ElemFacts::range(f.hull(&oracle).range))).collect()
            }
            _ => Vec::new(),
        }
    }

    /// A range holding every element of the integer array `arr` in context
    /// `at` (`TOP` when nothing is known).
    pub fn array_range(&self, arr: &Value, at: Where) -> Range {
        let oracle = self.oracle(at);
        self.view_facts(arr, at).iter().fold(Range::TOP, |acc, f| acc.meet(&f.hull(&oracle).range, &oracle))
    }

    /// [`Self::array_range`] as constant ends, both known.
    pub fn array_interval(&self, arr: &Value, at: Where) -> Option<(i64, i64)> {
        match self.array_range(arr, at).interval(&self.oracle(at)) {
            (Some(lo), Some(hi)) if lo <= hi => Some((lo, hi)),
            _ => None,
        }
    }

    /// The range of a scalar in context `at`: the array facts' (symbolic
    /// ends kept), met with the scalar tiers' constant ends.
    pub fn value_range(&self, v: &Value, at: Where) -> Range {
        let (lo, hi) = self.scalar_bounds(v, at);
        let scalar = Range { lo: lo.map(Bound::Const), hi: hi.map(Bound::Const) };
        match Cell::of(v) {
            Some(cell) if !matches!(v, Value::Float(_)) => {
                self.array_fact_range(&cell, v, at).meet(&scalar, &self.oracle(at))
            }
            _ => scalar,
        }
    }

    /// A range holding every one of the integer scalars `values` (the join of
    /// their ranges; `TOP` when one is unknown).
    pub fn join_of(&self, values: &[Value], at: Where) -> Range {
        let oracle = self.oracle(at);
        let mut acc: Option<Range> = None;
        for v in values {
            let r = self.value_range(v, at);
            if r.is_top() {
                return Range::TOP;
            }
            acc = Some(match acc {
                None => r,
                Some(a) => a.join(&r, &oracle),
            });
        }
        acc.unwrap_or(Range::TOP)
    }

    /// A range holding every integer element of a value: a scalar's, an
    /// array's, the join over a list's or tuple's elements.
    pub fn range_of(&self, v: &Value, at: Where) -> Range {
        match v {
            Value::StaticArray { .. } | Value::DynamicNDArray(_) => self.array_range(v, at),
            Value::List(d) | Value::Tuple(d) => {
                let oracle = self.oracle(at);
                let mut acc: Option<Range> = None;
                for e in &d.values {
                    let r = self.range_of(e, at);
                    if r.is_top() {
                        return Range::TOP;
                    }
                    acc = Some(match acc {
                        None => r,
                        Some(a) => a.join(&r, &oracle),
                    });
                }
                acc.unwrap_or(Range::TOP)
            }
            Value::Integer(_) | Value::Boolean(_) => self.value_range(v, at),
            _ => Range::TOP,
        }
    }

    /// Whether the array `arr`'s elements, in row-major order, are in order
    /// `order` (or a stronger one) in context `at`.
    pub fn array_has_order(&self, arr: &Value, order: Order, at: Where) -> bool {
        let oracle = self.oracle(at);
        self.view_facts(arr, at).iter().any(|f| f.hull(&oracle).order.is_some_and(|o| o.implies(order)))
    }

    /// Whether the first `len` elements of the flat runtime-shaped array
    /// `arr` (its live elements) are in order `order` or a stronger one:
    /// they lie in the first segment of a segmentation of its storage, which
    /// is so ordered. A segment over the whole capacity holds them all: the
    /// live length of a runtime-shaped array never exceeds its capacity (the
    /// constructors check it, `require_extent_within`, and a read past the
    /// capacity has no witness).
    pub fn live_prefix_has_order(&self, arr: &Value, len: &Value, order: Order, at: Where) -> bool {
        let oracle = self.oracle(at);
        let live_end = match (len.int_val(), len.stmt_id()) {
            (Some(c), _) => Bound::Const(c),
            (None, Some(s)) => Bound::stmt(s),
            _ => return false,
        };
        self.view_facts(arr, at).iter().any(|f| {
            let Some(first) = f.segments().first() else { return false };
            let covers = first.end == Bound::Const(f.len() as i64) || live_end.le(&first.end, &oracle);
            first.facts.order.is_some_and(|o| o.implies(order)) && covers
        })
    }

    /// Whether `elements` (in order) are known to be a permutation of
    /// `0..len`: the array facts of a list stated as one (an asserted
    /// `is_permutation`, the argsort of a line), read through a run that
    /// covers all of it, in context `Here`.
    pub fn is_known_permutation(&self, elements: &[Value]) -> bool {
        if self.array_facts.off {
            return false;
        }
        let Some(cells) = Cell::list(elements) else { return false };
        let context = self.fact_context(Where::Here);
        self.array_facts.store.list_facts(&cells, &self.oracle(Where::Here), &context).iter().any(Segmentation::is_permutation)
    }

    /// The scalar leaves of a value (a scalar, a list or tuple of them, or a
    /// static array whose cells are known), without emitting IR.
    pub fn leaf_cells(&self, v: &Value) -> Option<Vec<Value>> {
        match v {
            Value::Integer(_) | Value::Boolean(_) | Value::Float(_) => Some(vec![v.clone()]),
            Value::List(d) | Value::Tuple(d) => {
                let mut out = Vec::new();
                for e in &d.values {
                    out.extend(self.leaf_cells(e)?);
                }
                Some(out)
            }
            Value::StaticArray { dtype, shape, segment_id, strides, offset, .. } => {
                if *dtype == NumberType::Complex || !self.storage_is_current(*segment_id) {
                    return None;
                }
                let Some(StorageValue::Cells(cells)) = self.storage_value(*segment_id) else { return None };
                view_addresses(shape, strides, *offset).iter().map(|&a| cells.get(a).cloned()).collect()
            }
            _ => None,
        }
    }
}

/// A view's addresses in its storage, in row-major order.
fn view_addresses(shape: &[usize], strides: &[usize], offset: usize) -> Vec<usize> {
    let n: usize = shape.iter().product();
    let logical = crate::helpers::shape_arith::row_major_strides(shape);
    (0..n)
        .map(|flat| {
            let coords = crate::helpers::shape_arith::decode_coords(flat, shape, &logical);
            offset + coords.iter().zip(strides).map(|(c, s)| c * s).sum::<usize>()
        })
        .collect()
}

/// The facts of a storage of `len` cells restricted to a view of `n` cells
/// at `addresses` (row-major): the facts themselves for the whole storage
/// in order, a slice for a run of consecutive addresses, else the range of
/// the cells the view's addresses span.
fn restrict(facts: &Segmentation, addresses: &[usize], len: usize, n: usize, s: &dyn Scalars) -> Segmentation {
    let run = addresses.windows(2).all(|w| w[1] == w[0] + 1);
    match (addresses.first(), addresses.last()) {
        (Some(&first), Some(&last)) if run && first == 0 && last + 1 == len => facts.clone(),
        (Some(&first), Some(&last)) if run => facts.slice(first, last + 1, s),
        (Some(&first), Some(&last)) => {
            let (lo, hi) = (first.min(last), first.max(last));
            let lo = addresses.iter().copied().min().unwrap_or(lo);
            let hi = addresses.iter().copied().max().unwrap_or(hi);
            Segmentation::uniform(n, ElemFacts::range(facts.over(lo, hi, s).range))
        }
        _ => Segmentation::uniform(n, ElemFacts::TOP),
    }
}

/// What a storage holds now, for the fact queries.
pub(crate) enum StorageValue {
    /// The cells, in address order.
    Cells(Vec<Value>),
    /// The IR version statement.
    Version(StmtId),
}

#[cfg(test)]
mod tests {
    use crate::builder::{IRBuilder, Where};
    use crate::circuit_input::InputPath;
    use crate::optim::array_facts::{ElemFacts, Order, Range, Segmentation};
    use crate::types::{NumberType, Value};

    fn input(b: &mut IRBuilder, name: &str) -> Value {
        b.ir_read_integer(InputPath::new(name, vec![]), false)
    }

    /// Three runtime cells stated in `[0, 4]` and increasing, on every path.
    fn bounded_cells(b: &mut IRBuilder) -> Vec<Value> {
        let cells: Vec<Value> = (0..3).map(|i| input(b, &format!("a{i}"))).collect();
        let facts = ElemFacts { range: Range::new(0, 4), order: Some(Order::Increasing), affine: None };
        b.state_defined_facts(&cells, Segmentation::uniform(3, facts));
        cells
    }

    fn array_of(b: &mut IRBuilder, cells: &[Value]) -> Value {
        crate::helpers::static_array::base::build_static_array_from_flat(b, cells.to_vec(), vec![cells.len()], NumberType::Integer)
    }

    fn storage_of(arr: &Value) -> u32 {
        match arr {
            Value::StaticArray { segment_id, .. } => *segment_id,
            _ => unreachable!(),
        }
    }

    #[test]
    fn cell_facts_reach_the_solver_free_bounds() {
        let mut b = IRBuilder::new();
        let cells = bounded_cells(&mut b);
        assert_eq!(b.cheap_bounds(&cells[1], Where::Here), (Some(0), Some(4)));
        assert_eq!(b.cheap_bounds(&cells[1], Where::Everywhere), (Some(0), Some(4)));
        let other = input(&mut b, "x");
        assert_eq!(b.cheap_bounds(&other, Where::Here), (None, None));
    }

    #[test]
    fn an_array_and_its_views_see_the_facts_of_their_cells() {
        let mut b = IRBuilder::new();
        let cells = bounded_cells(&mut b);
        let arr = array_of(&mut b, &cells);
        assert_eq!(b.array_interval(&arr, Where::Here), Some((0, 4)));
        assert!(b.array_has_order(&arr, Order::Increasing, Where::Here));
        // A view of the last two cells: a run of the stated list.
        let Value::StaticArray { segment_id, dtype, value_id, .. } = arr.clone() else { unreachable!() };
        let view = Value::StaticArray { dtype, shape: vec![2], segment_id, strides: vec![1], offset: 1, imag_segment_id: None, value_id };
        assert!(b.array_has_order(&view, Order::Increasing, Where::Here));
        assert_eq!(b.array_interval(&view, Where::Here), Some((0, 4)));
    }

    #[test]
    fn a_write_through_any_name_gives_the_array_new_contents() {
        // AF-1 at the builder level: every name of the array reads its
        // storage's current cells, which a write replaces.
        let mut b = IRBuilder::new();
        let cells = bounded_cells(&mut b);
        let arr = array_of(&mut b, &cells);
        let alias = arr.clone();
        let x = input(&mut b, "x");
        b.storage_write(storage_of(&alias), 0, &x);
        assert_eq!(b.array_interval(&arr, Where::Here), None, "the written cell has no fact");
        assert!(!b.array_has_order(&arr, Order::NonDecreasing, Where::Here));
        // A write of a value in range keeps the range, not the order.
        let mut b = IRBuilder::new();
        let cells = bounded_cells(&mut b);
        let arr = array_of(&mut b, &cells);
        let four = b.ir_constant_int(4);
        b.storage_write(storage_of(&arr), 0, &four);
        assert_eq!(b.array_interval(&arr, Where::Here), Some((0, 4)));
        assert!(!b.array_has_order(&arr, Order::NonDecreasing, Where::Here));
    }

    #[test]
    fn a_runtime_write_joins_the_value_into_the_version() {
        let mut b = IRBuilder::new();
        let cells = bounded_cells(&mut b);
        let arr = array_of(&mut b, &cells);
        let storage = storage_of(&arr);
        let i = input(&mut b, "i");
        let nine = b.ir_constant_int(9);
        b.storage_write_at(storage, &i, &nine);
        // The array is now a version: its cells are its old ones or 9.
        assert_eq!(b.array_interval(&arr, Where::Here), Some((0, 9)));
        assert!(!b.array_has_order(&arr, Order::NonDecreasing, Where::Here));
        let j = input(&mut b, "j");
        let read = b.storage_read_at(storage, &j);
        assert_eq!(b.cheap_bounds(&read, Where::Here), (Some(0), Some(9)));
        // A read made before a later write keeps what its version held.
        let minus = b.ir_constant_int(-5);
        b.storage_write_at(storage, &i, &minus);
        assert_eq!(b.cheap_bounds(&read, Where::Here), (Some(0), Some(9)));
        assert_eq!(b.array_interval(&arr, Where::Here), Some((-5, 9)));
        // A value with no bound leaves the cells unbounded.
        let x = input(&mut b, "x");
        b.storage_write_at(storage, &i, &x);
        assert_eq!(b.array_interval(&arr, Where::Here), None);
    }

    #[test]
    fn a_constant_write_to_a_version_replaces_one_cell() {
        // After a runtime write the array is a version; a later write on
        // every path at a constant address splits its segment: the other
        // cells keep their facts, the written one has the value's.
        let mut b = IRBuilder::new();
        let cells = bounded_cells(&mut b);
        let arr = array_of(&mut b, &cells);
        let storage = storage_of(&arr);
        let i = input(&mut b, "i");
        let two = b.ir_constant_int(2);
        b.storage_write_at(storage, &i, &two);
        let hundred = b.ir_constant_int(100);
        b.storage_write(storage, 1, &hundred);
        assert_eq!(b.array_interval(&arr, Where::Here), Some((0, 100)));
        // Reading cell 0 or 2 at a runtime position that can only name them
        // (positions 0 and 2 of a 2-element strided view would; here a
        // constant read): the version's facts at that position.
        let zero = b.ir_constant_int(0);
        let first = b.storage_read_at(storage, &zero);
        assert_eq!(b.cheap_bounds(&first, Where::Here), (Some(0), Some(4)));
    }

    #[test]
    fn a_write_under_a_branch_joins_the_old_and_new_cells() {
        let mut b = IRBuilder::new();
        let cells = bounded_cells(&mut b);
        let arr = array_of(&mut b, &cells);
        let c = input(&mut b, "c");
        let zero = b.ir_constant_int(0);
        let cond = b.ir_not_equal_i(&c, &zero);
        let previous = b.swap_write_predicate(Some(cond));
        let seven = b.ir_constant_int(7);
        b.storage_write(storage_of(&arr), 1, &seven);
        b.swap_write_predicate(previous);
        // Cell 1 is `select(c, 7, a1)`: on every path in [0, 7].
        assert_eq!(b.array_interval(&arr, Where::Everywhere), Some((0, 7)));
        assert_eq!(b.array_interval(&arr, Where::Here), Some((0, 7)));
        assert!(!b.array_has_order(&arr, Order::NonDecreasing, Where::Here));
    }

    #[test]
    fn a_fact_derived_under_a_branch_is_not_used_outside_it() {
        // `y` has a scalar fact only where `c` holds (an assert in the
        // branch); a range derived from it there must not be used on every
        // path.
        let mut b = IRBuilder::new();
        let y = input(&mut b, "y");
        let c = input(&mut b, "c");
        let zero = b.ir_constant_int(0);
        let cond = b.ir_not_equal_i(&c, &zero);
        let previous = b.swap_write_predicate(Some(cond));
        {
            use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
            let vid = y.value_id().unwrap();
            for (op, n) in [(CmpOp::Ge, 0), (CmpOp::Le, 3)] {
                b.facts.insert_for(vid, ContractTerm::Cmp { op, lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))), rhs: Box::new(ContractTerm::LitInt(n)) });
            }
        }
        let out = input(&mut b, "out");
        let y2 = y.clone();
        b.state_derived_facts(std::slice::from_ref(&out), |b, at| {
            let r = b.value_range(&y2, at);
            (!r.is_top()).then(|| Segmentation::uniform(1, ElemFacts::range(r)))
        });
        assert_eq!(b.cheap_bounds(&out, Where::Here), (Some(0), Some(3)), "seen where `c` holds");
        b.swap_write_predicate(previous);
        assert_eq!(b.cheap_bounds(&out, Where::Here), (None, None), "not seen on every path");
        assert_eq!(b.cheap_bounds(&out, Where::Everywhere), (None, None));
    }

    #[test]
    fn a_merge_of_two_ordered_arrays_is_ordered() {
        let mut b = IRBuilder::new();
        let a = bounded_cells(&mut b);
        let a = array_of(&mut b, &a);
        let cells: Vec<Value> = (0..3).map(|i| input(&mut b, &format!("z{i}"))).collect();
        let facts = ElemFacts { range: Range::new(10, 20), order: Some(Order::NonDecreasing), affine: None };
        b.state_defined_facts(&cells, Segmentation::uniform(3, facts));
        let z = array_of(&mut b, &cells);
        let c = input(&mut b, "c");
        let zero = b.ir_constant_int(0);
        let cond = b.ir_not_equal_i(&c, &zero);
        let merged = crate::helpers::value_ops::select_value(&mut b, &cond, &a, &z);
        assert_eq!(b.array_interval(&merged, Where::Here), Some((0, 20)));
        assert!(b.array_has_order(&merged, Order::NonDecreasing, Where::Here));
        assert!(!b.array_has_order(&merged, Order::Increasing, Where::Here));
    }

    #[test]
    fn a_stale_array_has_no_facts() {
        // A copy NumPy would share with `arr` (linked): a write to `arr`
        // makes it stale, and its facts are not used.
        let mut b = IRBuilder::new();
        let cells = bounded_cells(&mut b);
        let arr = array_of(&mut b, &cells);
        let copy = array_of(&mut b, &cells);
        b.link_storages(storage_of(&copy), storage_of(&arr), crate::builder::COPIED_VIEW);
        assert_eq!(b.array_interval(&copy, Where::Here), Some((0, 4)));
        let x = input(&mut b, "x");
        b.storage_write(storage_of(&arr), 0, &x);
        assert_eq!(b.array_interval(&copy, Where::Here), None);
        assert!(b.view_facts(&copy, Where::Here).is_empty());
    }

    #[test]
    fn a_permutation_is_known_through_its_cells() {
        let mut b = IRBuilder::new();
        let cells: Vec<Value> = (0..4).map(|i| input(&mut b, &format!("p{i}"))).collect();
        b.record_permutation(&cells);
        assert!(b.is_known_permutation(&cells));
        assert!(!b.is_known_permutation(&cells[..3]), "a part of a permutation is not one");
        let mut swapped = cells.clone();
        swapped.swap(0, 1);
        assert!(!b.is_known_permutation(&swapped), "another order of the cells is another list");
    }
}
