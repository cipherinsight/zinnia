//! Checks under the execution predicate.
//!
//! Python raises `IndexError`, `ZeroDivisionError` or a math-domain error
//! only on the paths it executes. The IR generator sets the builder's write
//! predicate to the execution condition of every statement it lowers (its
//! enclosing branch conditions, loop-item activity, not broken / not
//! continued, not yet returned, and the chip-entry condition), so the same
//! predicate `p` says where a check has to hold. Every check the core emits
//! goes through the methods of this module, which emit the check `c` as
//! `p → c` when `p` is a runtime value. When `p` is statically true they
//! emit exactly what an unconditional check emits, so code that runs on
//! every path pays nothing.
//!
//! A conditional check is an unconditional constraint on a *guarded
//! operand*: `select(p, x, s)` with a safe constant `s` that passes the
//! check. This keeps two things sound:
//!
//! * Facts later derived from the constraint (the interval an
//!   `AssertU128Lt` deposits, a backend's range-check width) are true
//!   statements about the guarded operand on every path, never about `x`.
//! * Operations whose lowering has an implicit domain (a dynamic memory
//!   address must lie inside its segment; a range decomposition needs its
//!   operand in range) receive the guarded operand too, so the circuit stays
//!   satisfiable where the code does not run. Their results are observed only
//!   under `p`: writes are predicated, and values reach later code through
//!   the selects that merge branches and loop items.
//!
//! Facts learned while `p` is active are visible only to code that runs
//! under `p`'s conjuncts (see `FactStack::set_guard`).

use std::collections::HashMap;
use std::rc::Rc;

use crate::ir_defs::IR;
use crate::optim::predicates::FactGuard;
use crate::types::{StmtId, Value};

use super::IRBuilder;

/// Where the code being emitted runs.
pub(crate) enum Execution {
    /// On every path (no predicate, or a statically true one).
    Always,
    /// On no path (a statically false predicate).
    Never,
    /// Exactly where the runtime Boolean holds.
    When(Value),
}

/// An integer operand of a comparison: a constant, or the statement that
/// computes it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Operand {
    Const(i64),
    Stmt(StmtId),
}

/// The integer comparison `lhs + gap <= rhs`: the normal form of the
/// comparison atoms of an execution predicate (`a < b` is `a + 1 <= b`).
#[derive(Clone, Copy)]
struct IntOrder {
    lhs: Operand,
    rhs: Operand,
    gap: i64,
}

/// Memo tables of the execution-predicate machinery.
#[derive(Default)]
pub(crate) struct GuardCaches {
    /// Predicate statement → its conjuncts (see [`IRBuilder::predicate_atoms`]).
    atoms: HashMap<StmtId, Rc<[StmtId]>>,
    /// (predicate, operand, safe value) → the guarded operand, so repeated
    /// accesses through one index in one statement share a select.
    operands: HashMap<(StmtId, StmtId, i64), Value>,
    /// (execution predicate, index, axis length) → NumPy's checked position
    /// of the index on such an axis (`helpers::static_array::base::
    /// checked_axis_index`): every access through one index on one axis
    /// length under one predicate uses one position, so the array folds
    /// see the same cell (`builder/arrays.rs`).
    positions: HashMap<(Option<StmtId>, StmtId, usize), Value>,
}

/// The compile-time truth of a Boolean-like value, if known.
fn static_truth(v: &Value) -> Option<bool> {
    v.bool_val().or_else(|| v.int_val().map(|n| n != 0))
}

impl IRBuilder {
    /// Where the code being emitted runs, from the write predicate.
    pub(crate) fn execution(&self) -> Execution {
        match &self.write_predicate {
            None => Execution::Always,
            Some(p) => match static_truth(p) {
                Some(true) => Execution::Always,
                Some(false) => Execution::Never,
                // A runtime predicate always has a statement; without one
                // nothing can be guarded, and an unconditional check is the
                // sound fallback.
                None if p.stmt_id().is_none() => Execution::Always,
                None => Execution::When(p.clone()),
            },
        }
    }

    /// Keep the fact stack's guard equal to the execution predicate. Called
    /// whenever the write predicate changes.
    pub(super) fn sync_fact_guard(&mut self) {
        let guard = match self.execution() {
            Execution::Always => FactGuard::Always,
            Execution::Never => FactGuard::Never,
            Execution::When(p) => {
                let atoms = self.predicate_atoms(&p);
                let mut ids: Vec<_> = atoms.iter().map(|sid| self.stmts[*sid as usize].value_id).collect();
                ids.sort_unstable();
                ids.dedup();
                FactGuard::When(ids)
            }
        };
        self.facts.set_guard(guard);
    }

    /// The conjuncts of a runtime predicate: its `LogicalAnd` tree flattened
    /// (through double negations, which the loop scope builds for item
    /// activity), statically true leaves dropped, as sorted statement ids.
    /// `p` implies `q` whenever `atoms(q) ⊆ atoms(p)`.
    fn predicate_atoms(&mut self, p: &Value) -> Rc<[StmtId]> {
        let Some(root) = p.stmt_id() else {
            return Rc::from(Vec::new());
        };
        if let Some(atoms) = self.guard_caches.atoms.get(&root) {
            return atoms.clone();
        }
        // The predicate is a DAG (loop and branch conditions share
        // sub-conjunctions), so visit each statement once.
        let mut atoms = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut pending = vec![root];
        while let Some(sid) = pending.pop() {
            if !visited.insert(sid) {
                continue;
            }
            let stmt = &self.stmts[sid as usize];
            match &stmt.ir {
                IR::LogicalAnd => pending.extend(stmt.arguments.iter().copied()),
                IR::LogicalNot if self.is_negation(stmt.arguments[0]) => {
                    pending.push(self.stmts[stmt.arguments[0] as usize].arguments[0]);
                }
                IR::ConstantBool { value: true } => {}
                IR::ConstantInt { value } if *value != 0 => {}
                _ => atoms.push(sid),
            }
        }
        atoms.sort_unstable();
        atoms.dedup();
        let atoms: Rc<[StmtId]> = Rc::from(atoms);
        self.guard_caches.atoms.insert(root, atoms.clone());
        atoms
    }

    /// `true` when `q` holds wherever `p` holds, by inclusion of conjuncts.
    fn predicate_implies(&mut self, p: &Value, q: &Value) -> bool {
        if p.stmt_id() == q.stmt_id() {
            return true;
        }
        let (p_atoms, q_atoms) = (self.predicate_atoms(p), self.predicate_atoms(q));
        q_atoms.iter().all(|a| p_atoms.binary_search(a).is_ok())
    }

    /// The constant `v` takes on every path where `p` is false, when the IR
    /// determines it: a constant; `k` for `select(q, x, k)` with `q`
    /// implying `p` (a guarded operand); and sums, differences and products
    /// of such values (an address built from guarded indices).
    fn value_where_not_run(&mut self, v: &Value, p: &Value) -> Option<i64> {
        if let Some(c) = v.int_val() {
            return Some(c);
        }
        self.stmt_value_where_not_run(v.stmt_id()?, p, 0)
    }

    fn stmt_value_where_not_run(&mut self, sid: StmtId, p: &Value, depth: u32) -> Option<i64> {
        const MAX_DEPTH: u32 = 16;
        if let Some(c) = self.constant_of(sid) {
            return Some(c);
        }
        if depth > MAX_DEPTH {
            return None;
        }
        let stmt = self.stmts.get(sid as usize)?;
        let args = stmt.arguments.clone();
        match (&stmt.ir, &args[..]) {
            (IR::SelectI | IR::SelectB, &[cond, _, otherwise]) => {
                let k = self.constant_of(otherwise)?;
                let q = self.value_of_stmt(cond);
                self.predicate_implies(&q, p).then_some(k)
            }
            (IR::AddI, &[a, b]) => self
                .stmt_value_where_not_run(a, p, depth + 1)?
                .checked_add(self.stmt_value_where_not_run(b, p, depth + 1)?),
            (IR::SubI, &[a, b]) => self
                .stmt_value_where_not_run(a, p, depth + 1)?
                .checked_sub(self.stmt_value_where_not_run(b, p, depth + 1)?),
            (IR::MulI, &[a, b]) => {
                // A guarded index times a stride is 0 whatever the stride.
                let x = self.stmt_value_where_not_run(a, p, depth + 1);
                if x == Some(0) || self.constant_of(b) == Some(0) {
                    return Some(0);
                }
                x?.checked_mul(self.stmt_value_where_not_run(b, p, depth + 1)?)
            }
            _ => None,
        }
    }

    /// A `Value` handle on statement `sid`, enough for the predicate helpers
    /// (which read only its statement id).
    fn value_of_stmt(&self, sid: StmtId) -> Value {
        Value::Boolean(crate::types::ScalarValue {
            static_val: None,
            stmt_id: Some(sid),
            value_id: self.stmts[sid as usize].value_id,
        })
    }

    /// `true` for a `LogicalNot` statement.
    fn is_negation(&self, sid: StmtId) -> bool {
        matches!(self.stmts.get(sid as usize).map(|s| &s.ir), Some(IR::LogicalNot))
    }

    /// The integer value of a statement that computes a compile-time
    /// constant: a constant, or integer `+ - *` of such statements (the IR
    /// generator keeps loop counters as `start + k` additions).
    fn constant_of(&self, sid: StmtId) -> Option<i64> {
        self.constant_of_at_depth(sid, 0)
    }

    fn constant_of_at_depth(&self, sid: StmtId, depth: u32) -> Option<i64> {
        const MAX_DEPTH: u32 = 16;
        let stmt = self.stmts.get(sid as usize)?;
        let operands = |me: &Self| -> Option<(i64, i64)> {
            if depth >= MAX_DEPTH || stmt.arguments.len() != 2 {
                return None;
            }
            let a = me.constant_of_at_depth(stmt.arguments[0], depth + 1)?;
            let b = me.constant_of_at_depth(stmt.arguments[1], depth + 1)?;
            Some((a, b))
        };
        match stmt.ir {
            IR::ConstantInt { value } => Some(value),
            IR::ConstantBool { value } => Some(value as i64),
            IR::AddI => operands(self).and_then(|(a, b)| a.checked_add(b)),
            IR::SubI => operands(self).and_then(|(a, b)| a.checked_sub(b)),
            IR::MulI => operands(self).and_then(|(a, b)| a.checked_mul(b)),
            _ => None,
        }
    }

    /// A statement as a comparison operand (constants by value).
    fn operand_of_stmt(&self, sid: StmtId) -> Operand {
        self.constant_of(sid).map_or(Operand::Stmt(sid), Operand::Const)
    }

    /// A value as a comparison operand, if it is a constant or has a
    /// statement.
    fn operand_of_value(&self, v: &Value) -> Option<Operand> {
        match v.int_val() {
            Some(c) => Some(Operand::Const(c)),
            None => v.stmt_id().map(|sid| self.operand_of_stmt(sid)),
        }
    }

    /// An integer comparison statement, or the negation of one, in the form
    /// `lhs + gap <= rhs` (`¬(l + g <= r)` is `r + 1 - g <= l`).
    fn int_order(&self, sid: StmtId) -> Option<IntOrder> {
        let stmt = self.stmts.get(sid as usize)?;
        if matches!(stmt.ir, IR::LogicalNot) {
            let inner = self.int_order(*stmt.arguments.first()?)?;
            return Some(IntOrder { lhs: inner.rhs, rhs: inner.lhs, gap: 1 - inner.gap });
        }
        let [a, b] = stmt.arguments[..] else {
            return None;
        };
        let (a, b) = (self.operand_of_stmt(a), self.operand_of_stmt(b));
        // `LtIBounded` / `GtIBounded` are left out: a backend may compute
        // them wrongly for operands outside their contract, so their value
        // need not be the comparison's.
        let (lhs, rhs, gap) = match stmt.ir {
            IR::LtI => (a, b, 1),
            IR::GtI => (b, a, 1),
            IR::LteI => (a, b, 0),
            IR::GteI => (b, a, 0),
            _ => return None,
        };
        Some(IntOrder { lhs, rhs, gap })
    }

    /// `true` when the order atom `known` implies `wanted` over the
    /// integers: same sides with at least the wanted gap, or one side shared
    /// and the other two constants ordered accordingly.
    fn order_implies(known: IntOrder, wanted: IntOrder) -> bool {
        use Operand::Const;
        let same_lhs = known.lhs == wanted.lhs;
        let same_rhs = known.rhs == wanted.rhs;
        if same_lhs && same_rhs {
            return known.gap >= wanted.gap;
        }
        if same_rhs {
            // wanted.lhs + wanted.gap <= known.lhs + known.gap <= rhs
            if let (Const(kl), Const(wl)) = (known.lhs, wanted.lhs) {
                return wl.checked_add(wanted.gap) <= kl.checked_add(known.gap);
            }
        }
        if same_lhs {
            // lhs + wanted.gap <= known.rhs - known.gap + wanted.gap <= wanted.rhs
            if let (Const(kr), Const(wr)) = (known.rhs, wanted.rhs) {
                let reach = kr.checked_sub(known.gap).and_then(|v| v.checked_add(wanted.gap));
                return reach.is_some_and(|r| r <= wr);
            }
        }
        false
    }

    /// `true` when the atom `known` (a conjunct of the execution predicate)
    /// implies the Boolean statement `wanted`.
    fn atom_implies(&self, known: StmtId, wanted: StmtId) -> bool {
        if known == wanted {
            return true;
        }
        if let (Some(k), Some(w)) = (self.int_order(known), self.int_order(wanted)) {
            return Self::order_implies(k, w);
        }
        let (Some(k), Some(w)) = (self.stmts.get(known as usize), self.stmts.get(wanted as usize))
        else {
            return false;
        };
        // `a != b` / `a == b` in either operand order.
        if k.ir == w.ir && matches!(k.ir, IR::NeI | IR::EqI) && k.arguments.len() == 2 {
            let [k0, k1] = [k.arguments[0], k.arguments[1]].map(|a| self.operand_of_stmt(a));
            let [w0, w1] = [w.arguments[0], w.arguments[1]].map(|a| self.operand_of_stmt(a));
            return (k0 == w0 && k1 == w1) || (k0 == w1 && k1 == w0);
        }
        false
    }

    /// `true` when the Boolean `check` holds wherever `p` holds, decided
    /// syntactically from `p`'s conjuncts (no solver call): `check` is one
    /// of them, or follows from a comparison among them.
    fn predicate_implies_check(&mut self, p: &Value, check: &Value) -> bool {
        if static_truth(check) == Some(true) {
            return true;
        }
        let Some(wanted) = check.stmt_id() else {
            return false;
        };
        if let Some(stmt) = self.stmts.get(wanted as usize) {
            if matches!(stmt.ir, IR::LogicalAnd) && stmt.arguments.len() == 2 {
                let (a, b) = (stmt.arguments[0], stmt.arguments[1]);
                let (a, b) = (self.value_of_stmt(a), self.value_of_stmt(b));
                return self.predicate_implies_check(p, &a) && self.predicate_implies_check(p, &b);
            }
        }
        let atoms = self.predicate_atoms(p);
        atoms.iter().any(|a| self.atom_implies(*a, wanted))
    }

    /// `true` when `p` implies the integer comparison `lhs + gap <= rhs`.
    fn predicate_implies_order(&mut self, p: &Value, wanted: IntOrder) -> bool {
        if let (Operand::Const(l), Operand::Const(r)) = (wanted.lhs, wanted.rhs) {
            return l.checked_add(wanted.gap).is_some_and(|v| v <= r);
        }
        let atoms = self.predicate_atoms(p);
        atoms
            .iter()
            .any(|a| self.int_order(*a).is_some_and(|known| Self::order_implies(known, wanted)))
    }

    /// `true` when `p` implies `0 <= x`.
    fn predicate_implies_non_negative(&mut self, p: &Value, x: &Value) -> bool {
        let Some(x) = self.operand_of_value(x) else {
            return false;
        };
        self.predicate_implies_order(p, IntOrder { lhs: Operand::Const(0), rhs: x, gap: 0 })
    }

    /// `true` when `p` implies `0 <= idx < hi`.
    fn predicate_implies_index_below(&mut self, p: &Value, idx: &Value, hi: &Value) -> bool {
        let (Some(i), Some(h)) = (self.operand_of_value(idx), self.operand_of_value(hi)) else {
            return false;
        };
        self.predicate_implies_non_negative(p, idx)
            && self.predicate_implies_order(p, IntOrder { lhs: i, rhs: h, gap: 1 })
    }

    /// `true` when the code being emitted runs only where `lo <= idx < hi`
    /// (decided syntactically from the execution predicate's conjuncts, with
    /// no solver call; `false` for code that runs on every path). A loop
    /// item of `for i in range(len(x))` runs under `k < len(x)`, which
    /// discharges the check of `x[k]`.
    pub(crate) fn execution_implies_index_below(&mut self, idx: &Value, lo: i64, hi: &Value) -> bool {
        match self.operand_of_value(hi) {
            Some(h) => self.execution_implies_bound(idx, lo, h),
            None => false,
        }
    }

    /// [`Self::execution_implies_index_below`] for a constant `hi`.
    pub(crate) fn execution_implies_index_in(&mut self, idx: &Value, lo: i64, hi: i64) -> bool {
        self.execution_implies_bound(idx, lo, Operand::Const(hi))
    }

    fn execution_implies_bound(&mut self, idx: &Value, lo: i64, hi: Operand) -> bool {
        let Execution::When(p) = self.execution() else {
            return false;
        };
        let Some(i) = self.operand_of_value(idx) else {
            return false;
        };
        self.predicate_implies_order(&p, IntOrder { lhs: Operand::Const(lo), rhs: i, gap: 0 })
            && self.predicate_implies_order(&p, IntOrder { lhs: i, rhs: hi, gap: 1 })
    }

    /// `true` when the check `c` is already `select(q, x, true)` for a
    /// predicate `q` that implies `p`, i.e. already conditional on the
    /// current code running (the IR generator emits user `assert`s this way).
    fn is_conditioned_on(&mut self, c: &Value, p: &Value) -> bool {
        let Some(stmt) = c.stmt_id().and_then(|sid| self.stmts.get(sid as usize)) else {
            return false;
        };
        if !matches!(stmt.ir, IR::SelectI | IR::SelectB) || stmt.arguments.len() != 3 {
            return false;
        }
        let (cond, otherwise) = (stmt.arguments[0], stmt.arguments[2]);
        if self.constant_of(otherwise) != Some(1) {
            return false;
        }
        let q = self.value_of_stmt(cond);
        self.predicate_implies(&q, p)
    }

    /// `v` where the current code runs and the constant `safe` elsewhere:
    /// `select(p, v, safe)` under a runtime execution predicate `p`, `v`
    /// itself on every path (and for a constant `v`), `safe` on code that
    /// never runs. Hand the result to an operation whose lowering has an
    /// implicit domain (a memory address, a range decomposition, an
    /// exponent's bit width) or to a check that must pass where the code
    /// does not run; `safe` must lie inside that domain.
    /// The key of a checked position of `idx` on an axis of `len` under the
    /// current execution predicate (`None` where the code never runs or
    /// `idx` has no statement).
    fn position_key(&self, idx: &Value, len: usize) -> Option<(Option<StmtId>, StmtId, usize)> {
        let predicate = match self.execution() {
            Execution::Always => None,
            Execution::Never => return None,
            Execution::When(p) => Some(p.stmt_id()?),
        };
        Some((predicate, idx.stmt_id()?, len))
    }

    /// The checked position of `idx` on an axis of `len` built earlier under
    /// the current execution predicate, if any.
    pub(crate) fn checked_position(&self, idx: &Value, len: usize) -> Option<Value> {
        let key = self.position_key(idx, len)?;
        self.guard_caches.positions.get(&key).cloned()
    }

    /// Record `position`, the checked position of `idx` on an axis of `len`
    /// under the current execution predicate.
    pub(crate) fn remember_checked_position(&mut self, idx: &Value, len: usize, position: &Value) {
        if let Some(key) = self.position_key(idx, len) {
            self.guard_caches.positions.insert(key, position.clone());
        }
    }

    pub(crate) fn guard_operand(&mut self, v: &Value, safe: i64) -> Value {
        let (Some(v_sid), None) = (v.stmt_id(), v.int_val()) else {
            return v.clone();
        };
        let p = match self.execution() {
            Execution::Always => return v.clone(),
            Execution::Never => return self.ir_constant_int(safe),
            Execution::When(p) => p,
        };
        if self.value_where_not_run(v, &p) == Some(safe) {
            return v.clone();
        }
        let p_sid = p.stmt_id().expect("a runtime predicate has a statement");
        if let Some(guarded) = self.guard_caches.operands.get(&(p_sid, v_sid, safe)) {
            return guarded.clone();
        }
        let safe_v = self.ir_constant_int(safe);
        let guarded = self.ir_select_i(&p, v, &safe_v);
        self.guard_caches.operands.insert((p_sid, v_sid, safe), guarded.clone());
        guarded
    }

    /// Float version of [`Self::guard_operand`] for a float domain (e.g. the
    /// positive argument of `log`).
    pub(crate) fn guard_float_operand(&mut self, v: &Value, safe: f64) -> Value {
        if v.float_val().is_some() || v.stmt_id().is_none() {
            return v.clone();
        }
        let p = match self.execution() {
            Execution::Always => return v.clone(),
            Execution::Never => return self.ir_constant_float(safe),
            Execution::When(p) => p,
        };
        let safe_v = self.ir_constant_float(safe);
        self.ir_select_f(&p, v, &safe_v)
    }

    /// The address an access under the current execution predicate uses.
    /// Every backend constrains a dynamic memory address to lie inside its
    /// segment (or reads a fixed cell for a constant one), so where the code
    /// does not run the address must be in the segment without help from the
    /// (conditional) index check. It is when the address is built from
    /// guarded indices (its value there is a known in-range constant) or its
    /// interval lies inside `[0, capacity)` on every path; otherwise the
    /// access uses `select(p, addr, 0)`.
    pub(crate) fn guard_address(&mut self, addr: &Value, capacity: Option<u32>) -> Value {
        let p = match self.execution() {
            Execution::Always => return addr.clone(),
            Execution::Never => return self.guard_operand(addr, 0),
            Execution::When(p) => p,
        };
        if addr.int_val().is_some() || capacity == Some(0) {
            return addr.clone();
        }
        let within = |a: i64| capacity.is_some_and(|cap| 0 <= a && a < cap as i64);
        if self.value_where_not_run(addr, &p).is_some_and(within) {
            return addr.clone();
        }
        let interval = addr.value_id().and_then(|vid| self.provenance.interval_of(vid));
        if interval.is_some_and(|iv| within(iv.lo) && within(iv.hi)) {
            return addr.clone();
        }
        self.guard_operand(addr, 0)
    }

    /// The index an access uses after its bound `lo <= idx < hi` has been
    /// established where the current code runs (by a check or by the
    /// facts): `idx` itself when it is a constant, the code runs on every
    /// path, or `idx`'s interval lies in `[lo, hi)` on every path;
    /// `select(p, idx, lo)` otherwise, which is in range everywhere.
    pub(crate) fn index_in_range_everywhere(&mut self, idx: &Value, lo: i64, hi: i64) -> Value {
        let p = match self.execution() {
            Execution::Always => return idx.clone(),
            Execution::Never => return self.guard_operand(idx, lo),
            Execution::When(p) => p,
        };
        let within = |v: i64| lo <= v && v < hi;
        if idx.int_val().is_some() || self.value_where_not_run(idx, &p).is_some_and(within) {
            return idx.clone();
        }
        let interval = idx.value_id().and_then(|vid| self.provenance.interval_of(vid));
        if interval.is_some_and(|iv| within(iv.lo) && within(iv.hi)) {
            return idx.clone();
        }
        self.guard_operand(idx, lo)
    }

    /// Run `f` as code that executes only where `condition` holds as well:
    /// the execution predicate becomes `p ∧ condition` (`condition` alone on
    /// code that runs on every path), so the checks `f` emits hold only
    /// there, and its memory accesses stay inside their segments elsewhere.
    /// Writes `f` makes are predicated the same way.
    pub(crate) fn under_condition<T>(&mut self, condition: &Value, f: impl FnOnce(&mut Self) -> T) -> T {
        let narrowed = match self.write_predicate.clone() {
            None => condition.clone(),
            Some(p) => self.ir_logical_and(&p, condition),
        };
        let previous = self.swap_write_predicate(Some(narrowed));
        let result = f(self);
        self.swap_write_predicate(previous);
        result
    }

    /// Note that the current code must not run because a check reached here
    /// can never hold. Under a runtime predicate `p` this asserts `¬p` and
    /// returns `true`; on a path that always runs it returns `false` and the
    /// caller reports the compile error.
    pub(crate) fn assert_never_runs(&mut self) -> bool {
        match self.execution() {
            Execution::Always => false,
            Execution::Never => true,
            Execution::When(p) => {
                let not_run = self.ir_logical_not(&p);
                self.create_ir(&IR::Assert, &[not_run]);
                true
            }
        }
    }

    // ── Checks ────────────────────────────────────────────────────────

    /// Assert the Boolean `c` where the current code runs: `c` itself on
    /// every path, `p → c` (as `assert(select(p, c, true))`) under a runtime
    /// predicate `p`. Nothing is emitted when `p` already implies `c`, or
    /// when the code never runs.
    pub fn ir_assert(&mut self, c: &Value) -> Value {
        match self.execution() {
            Execution::Always => self.create_ir(&IR::Assert, &[c.clone()]),
            Execution::Never => Value::None,
            Execution::When(p) => {
                if self.predicate_implies_check(&p, c) {
                    return Value::None;
                }
                if self.is_conditioned_on(c, &p) {
                    return self.create_ir(&IR::Assert, &[c.clone()]);
                }
                let true_v = self.ir_constant_bool(true);
                let conditioned = self.ir_select_i(&p, c, &true_v);
                self.create_ir(&IR::Assert, &[conditioned])
            }
        }
    }

    /// Assert `a ∈ {0, 1}` where the current code runs.
    pub fn ir_assert_bit(&mut self, a: &Value) -> Value {
        match self.execution() {
            Execution::Always => self.create_ir(&IR::AssertBit, &[a.clone()]),
            Execution::Never => Value::None,
            Execution::When(_) => {
                let guarded = self.guard_operand(a, 0);
                self.create_ir(&IR::AssertBit, &[guarded])
            }
        }
    }

    /// Assert `0 <= a` (signed i64 domain) where the current code runs.
    pub fn ir_assert_non_neg_i(&mut self, a: &Value) -> Value {
        match self.execution() {
            Execution::Always => self.create_ir(&IR::AssertNonNegI, &[a.clone()]),
            Execution::Never => Value::None,
            Execution::When(p) => {
                if let Some(c) = a.int_val() {
                    if c < 0 {
                        self.assert_never_runs();
                    }
                    return Value::None;
                }
                if self.predicate_implies_non_negative(&p, a) {
                    return Value::None;
                }
                let guarded = match self.value_where_not_run(a, &p) {
                    Some(off) if off >= 0 => a.clone(),
                    _ => self.guard_operand(a, 0),
                };
                self.create_ir(&IR::AssertNonNegI, &[guarded])
            }
        }
    }

    /// Two-operand assert `0 <= idx < hi` where the current code runs.
    /// Backends pick the cheapest sound implementation; on Noir this lowers
    /// to a single `assert((idx as u128) < (hi as u128))` (~30 ACIR vs ~49
    /// for the 2-cmp + and + assert decomposition). See `IR::AssertU128Lt`.
    ///
    /// Under a runtime predicate `p` both operands are guarded: `idx` is 0
    /// (or its constant) where the code does not run and `hi` is one above
    /// that, so the constraint holds there.
    pub fn ir_assert_u128_lt(&mut self, idx: &Value, hi: &Value) -> Value {
        let p = match self.execution() {
            Execution::Always => return self.create_ir(&IR::AssertU128Lt, &[idx.clone(), hi.clone()]),
            Execution::Never => return Value::None,
            Execution::When(p) => p,
        };
        let never_valid = match (idx.int_val(), hi.int_val()) {
            (Some(i), Some(h)) => Some(!(0 <= i && i < h)),
            (Some(i), None) if i < 0 => Some(true),
            (_, Some(h)) if h <= 0 => Some(true),
            _ => None,
        };
        if let Some(never_valid) = never_valid {
            if never_valid {
                self.assert_never_runs();
            }
            return Value::None;
        }
        if self.predicate_implies_index_below(&p, idx, hi) {
            return Value::None;
        }
        // Where the code does not run, the checked index must be some `o`
        // with `0 <= o < hi`: its own value there when that already is in
        // range (an address built from guarded indices), else 0.
        let hi_off = match hi.int_val() {
            Some(h) => Some(h),
            None => self.value_where_not_run(hi, &p),
        };
        let idx_g = match self.value_where_not_run(idx, &p) {
            Some(o) if o >= 0 && hi_off.is_none_or(|h| o < h) => idx.clone(),
            _ if idx.stmt_id().is_none() => {
                // Nothing to guard with: check on every path.
                return self.create_ir(&IR::AssertU128Lt, &[idx.clone(), hi.clone()]);
            }
            _ => self.guard_operand(idx, 0),
        };
        let idx_off = self.value_where_not_run(&idx_g, &p).unwrap_or(0);
        let hi_g = if hi_off.is_some_and(|h| h > idx_off) {
            hi.clone()
        } else {
            self.guard_operand(hi, idx_off + 1)
        };
        self.create_ir(&IR::AssertU128Lt, &[idx_g, hi_g])
    }

    /// Constrain `0 <= idx < hi` where the current code runs and return the
    /// low decomposition bits of the index (of `select(p, idx, 0)` under a
    /// runtime predicate `p`, which equals `idx` wherever the bits are used).
    ///
    /// This is intentionally value-producing: users such as small static
    /// gathers can route directly on the constrained bits instead of
    /// rebuilding one-hot equality masks from the original index.
    pub fn ir_assert_u128_lt_bits(&mut self, idx: &Value, hi: usize) -> Vec<Value> {
        assert!(hi > 0, "assert_u128_lt_bits: hi must be positive");
        let idx = self.guard_operand(idx, 0);
        let bits = usize::BITS - (hi - 1).leading_zeros();
        (0..bits)
            .map(|bit| {
                self.create_ir(
                    &IR::AssertU128LtBit {
                        hi: hi as u64,
                        bit,
                        bits,
                    },
                    &[idx.clone()],
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::{CircuitInputs, ResolvedWitness};
    use crate::prove::mock::MockProverBackend;
    use crate::prove::traits::ProverBackend;
    use crate::prove::types::ProvingParams;
    use crate::types::ScalarValue;

    /// A value the builder treats as unknown at compile time (no static
    /// value) but the mock prover evaluates: a constant statement whose
    /// handle drops its static value, like a witness input.
    fn opaque_int(b: &mut IRBuilder, v: i64) -> Value {
        let c = b.create_ir(&IR::ConstantInt { value: v }, &[]);
        Value::Integer(ScalarValue { static_val: None, stmt_id: c.stmt_id(), value_id: c.value_id().unwrap() })
    }

    fn opaque_bool(b: &mut IRBuilder, v: bool) -> Value {
        let c = b.create_ir(&IR::ConstantBool { value: v }, &[]);
        Value::Boolean(ScalarValue { static_val: None, stmt_id: c.stmt_id(), value_id: c.value_id().unwrap() })
    }

    fn mock_satisfied(b: IRBuilder) -> bool {
        let graph = b.export_ir_graph();
        let params = ProvingParams::default();
        let witness = ResolvedWitness::new(CircuitInputs::new(), params.precision_bits);
        MockProverBackend.prove(&graph, &witness, &params).unwrap().proof_bytes == "mock_satisfied"
    }

    /// `0 <= idx < 4` checked while the code runs exactly when `runs`.
    fn bound_check_under(runs: Option<bool>, idx: i64) -> IRBuilder {
        let mut b = IRBuilder::new();
        let p = runs.map(|r| opaque_bool(&mut b, r));
        b.swap_write_predicate(p);
        let idx = opaque_int(&mut b, idx);
        let hi = b.ir_constant_int(4);
        b.ir_assert_u128_lt(&idx, &hi);
        b
    }

    #[test]
    fn check_on_every_path_is_unchanged() {
        for predicate in [None, Some(true)] {
            let mut b = IRBuilder::new();
            if let Some(v) = predicate {
                let p = b.ir_constant_bool(v);
                b.swap_write_predicate(Some(p));
            }
            let c = opaque_bool(&mut b, true);
            let before = b.stmts.len();
            b.ir_assert(&c);
            let emitted: Vec<&IR> = b.stmts[before..].iter().map(|s| &s.ir).collect();
            assert_eq!(emitted, vec![&IR::Assert]);
            assert_eq!(b.stmts.last().unwrap().arguments, vec![c.stmt_id().unwrap()]);
        }
    }

    #[test]
    fn check_holds_only_where_the_code_runs() {
        // Out of range: rejected where the code runs, accepted elsewhere.
        assert!(!mock_satisfied(bound_check_under(None, 9)));
        assert!(!mock_satisfied(bound_check_under(Some(true), 9)));
        assert!(mock_satisfied(bound_check_under(Some(false), 9)));
        assert!(!mock_satisfied(bound_check_under(Some(true), -1)));
        // In range: accepted either way.
        assert!(mock_satisfied(bound_check_under(Some(true), 3)));
        assert!(mock_satisfied(bound_check_under(Some(false), 3)));
    }

    #[test]
    fn check_implied_by_the_predicate_emits_nothing() {
        // A loop item active under `2 < n` reads `x[2]`.
        let mut b = IRBuilder::new();
        let n = opaque_int(&mut b, 3);
        let k = b.ir_constant_int(2);
        let active = b.ir_less_than_i(&k, &n);
        b.swap_write_predicate(Some(active));
        let before = b.stmts.len();
        b.ir_assert_u128_lt(&k, &n);
        let lt = b.ir_less_than_i(&k, &n);
        b.ir_assert(&lt);
        let one = b.ir_constant_int(1);
        let weaker = b.ir_less_than_i(&one, &n);
        b.ir_assert(&weaker);
        assert!(b.stmts[before..].iter().all(|s| !s.ir.is_fixed()));
        // A stronger bound is not implied.
        let three = b.ir_constant_int(3);
        let stronger = b.ir_less_than_i(&three, &n);
        b.ir_assert(&stronger);
        assert!(matches!(b.stmts.last().unwrap().ir, IR::Assert));
    }

    #[test]
    fn dynamic_address_is_in_its_segment_where_the_code_does_not_run() {
        let mut b = IRBuilder::new();
        let zero = b.ir_constant_int(0);
        let storage = b.alloc_storage(vec![zero; 4], crate::types::NumberType::Integer);
        let p = opaque_bool(&mut b, false);
        b.swap_write_predicate(Some(p.clone()));
        let addr = opaque_int(&mut b, 9);
        b.storage_read_at(storage, &addr);
        let read = b.stmts.last().unwrap().clone();
        assert!(matches!(read.ir, IR::ArrayRead(_)));
        let guarded = &b.stmts[read.arguments[1] as usize];
        assert!(matches!(guarded.ir, IR::SelectI));
        assert_eq!(guarded.arguments[0], p.stmt_id().unwrap());
        assert_eq!(guarded.arguments[1], addr.stmt_id().unwrap());
        assert_eq!(b.constant_of(guarded.arguments[2]), Some(0));
    }

    #[test]
    fn index_that_can_never_be_valid_asserts_the_code_does_not_run() {
        for runs in [false, true] {
            let mut b = IRBuilder::new();
            let p = opaque_bool(&mut b, runs);
            b.swap_write_predicate(Some(p));
            let idx = b.ir_constant_int(7);
            let used = crate::optim::resolver::discharge_index_in_range(&mut b, &idx, 0, 4, "test");
            assert_eq!(used.int_val(), Some(0));
            assert_eq!(mock_satisfied(b), !runs);
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn index_that_can_never_be_valid_on_every_path_is_a_compile_error() {
        let mut b = IRBuilder::new();
        let idx = b.ir_constant_int(7);
        crate::optim::resolver::discharge_index_in_range(&mut b, &idx, 0, 4, "test");
    }

    #[test]
    fn fact_learned_under_a_predicate_is_used_only_under_it() {
        use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
        let mut b = IRBuilder::new();
        let j = opaque_int(&mut b, 2);
        let j_vid = j.value_id().unwrap();
        let below_4 = ContractTerm::Cmp {
            op: CmpOp::Lt,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(j_vid))),
            rhs: Box::new(ContractTerm::LitInt(4)),
        };
        let input = |b: &mut IRBuilder, name: &str| {
            b.ir_read_integer(crate::circuit_input::InputPath::new(name.to_string(), vec![]), false)
        };
        let p = input(&mut b, "p");
        b.swap_write_predicate(Some(p.clone()));
        b.facts.insert_for(j_vid, below_4.clone());
        assert_eq!(b.prove(&below_4), crate::optim::prove::ProveOutcome::Proved);
        // A later statement under a stronger predicate may use it ...
        let q = input(&mut b, "q");
        let p_and_q = b.ir_logical_and(&p, &q);
        b.swap_write_predicate(Some(p_and_q));
        assert_eq!(b.prove(&below_4), crate::optim::prove::ProveOutcome::Proved);
        // ... code that runs elsewhere may not.
        b.swap_write_predicate(Some(q));
        assert_eq!(b.prove(&below_4), crate::optim::prove::ProveOutcome::Unknown);
        b.swap_write_predicate(None);
        assert_eq!(b.prove(&below_4), crate::optim::prove::ProveOutcome::Unknown);
    }
}
