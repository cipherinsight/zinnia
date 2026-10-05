//! What a `for` loop or a comprehension iterates over, one item at a time.
//!
//! Every loop is unrolled at compile time, so the number of items must be a
//! compile-time number. When Python's trip count is a runtime value (the
//! length of a bounded-dynamic array, or `range(n)` with a private `n`), the
//! loop is unrolled to a proven upper bound `B` and each item carries an
//! *activity* condition: item `k` exists in Python iff its condition holds.
//! Callers run the body of an inactive item under a false condition, so the
//! item has no effect (writes keep old cells, assertions are vacuous, values
//! are discarded by the surrounding selects).
//!
//! Soundness does not rest on the bound inference: for a runtime-bounded
//! `range`, the circuit also asserts that the item after the last unrolled
//! one is inactive. If an inferred bound were too small, an input that needs
//! more iterations makes the circuit unsatisfiable instead of truncating the
//! loop. When no bound can be established at all, the program is rejected.

use crate::ast::*;
use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};
use crate::types::{DynamicNDArrayData, Value};

use super::IRGenerator;

/// One item of an unrolled iteration.
#[derive(Debug, Clone)]
pub(crate) struct LoopItem {
    /// The value bound to the loop target for this item.
    pub value: Value,
    /// `None` when the item exists on every execution; otherwise the runtime
    /// condition under which Python produces this item.
    pub active: Option<Value>,
    /// Facts that hold whenever the item is active, used as path conditions
    /// so that bounds checks implied by the activity (e.g. `x[k]` with
    /// `k < len(x)`) can be discharged at compile time.
    pub active_facts: Vec<ContractTerm>,
}

impl LoopItem {
    fn always(value: Value) -> Self {
        LoopItem { value, active: None, active_facts: Vec::new() }
    }
}

/// Error text for a loop whose trip count is a runtime value without a
/// provable maximum. Shared by `range` and `while` so the message is uniform.
pub(crate) const UNBOUNDED_LOOP_ERROR: &str =
    "loop bound depends on a private value with no provable maximum";

fn value_term(v: &Value) -> Option<ContractTerm> {
    if let Some(n) = v.int_val() {
        return Some(ContractTerm::LitInt(n));
    }
    v.value_id().map(|vid| ContractTerm::Var(ContractVar::Value(vid)))
}

fn cmp_term(op: CmpOp, lhs: &Value, rhs: &Value) -> Option<ContractTerm> {
    Some(ContractTerm::Cmp {
        op,
        lhs: Box::new(value_term(lhs)?),
        rhs: Box::new(value_term(rhs)?),
    })
}

/// `true` for the AST of a call to the builtin `range`.
fn range_call(node: &ASTNode) -> Option<&ASTNamedAttribute> {
    match node {
        ASTNode::ASTNamedAttribute(call)
            if call.target.is_none()
                && call.member == "range"
                && call.kwargs.is_empty()
                && !call.args.iter().any(|a| matches!(a, ASTNode::ASTStarredExpr(_))) =>
        {
            Some(call)
        }
        _ => None,
    }
}

impl IRGenerator {
    /// Evaluate `iter_expr` and return the items a `for` loop or a
    /// comprehension generator iterates over.
    pub(crate) fn iteration_items(&mut self, iter_expr: &ASTNode) -> Vec<LoopItem> {
        if let Some(call) = range_call(iter_expr) {
            let args: Vec<Value> = call.args.iter().map(|a| self.visit(a)).collect();
            return self.range_items(&args);
        }
        let iterable = self.visit(iter_expr);
        self.value_items(&iterable)
    }

    /// Items of an already-evaluated iterable value.
    pub(crate) fn value_items(&mut self, iterable: &Value) -> Vec<LoopItem> {
        match iterable {
            Value::StaticArray { shape, .. } => (0..shape[0])
                .map(|i| {
                    LoopItem::always(crate::helpers::static_array::read::iter_element(
                        &mut self.builder,
                        iterable,
                        i,
                    ))
                })
                .collect(),
            Value::List(data) | Value::Tuple(data) => {
                data.values.iter().cloned().map(LoopItem::always).collect()
            }
            Value::DynamicNDArray(data) => self.dynamic_array_items(data),
            _ => panic!("'{}' object is not iterable", iterable.zinnia_type()),
        }
    }

    /// Items of a rank-1 bounded-dynamic array: one per slot of the public
    /// capacity, active while the slot index is below the runtime length.
    fn dynamic_array_items(&mut self, data: &DynamicNDArrayData) -> Vec<LoopItem> {
        if data.envelope.dims.len() != 1 {
            panic!(
                "iteration over rank-{} DynamicNDArray is not supported (only rank-1 for now)",
                data.envelope.dims.len()
            );
        }
        // Contiguous, offset-0 layout is the only case the current
        // bounded-dynamic producers emit. A view with non-trivial strides or
        // a non-zero offset would need address arithmetic per cell.
        if data.meta.logical_offset != 0 || data.meta.logical_strides.first().copied() != Some(1) {
            panic!(
                "iteration over a strided / offset DynamicNDArray view is not supported (offset={}, stride={:?})",
                data.meta.logical_offset, data.meta.logical_strides
            );
        }
        let dim = data.envelope.dims[0];
        let length = Value::Integer(data.meta.runtime_length.clone());
        let axis_length = data.meta.runtime_shape.first().cloned().map(Value::Integer);
        (0..dim.max)
            .map(|k| {
                let address = self.builder.ir_constant_int(k as i64);
                let value = crate::helpers::segment::read_typed(
                    &mut self.builder,
                    data.segment_id,
                    &address,
                    data.dtype,
                );
                if k < dim.min {
                    return LoopItem::always(value);
                }
                let active = self.builder.ir_less_than_i(&address, &length);
                let mut active_facts: Vec<ContractTerm> =
                    cmp_term(CmpOp::Lt, &address, &length).into_iter().collect();
                if let Some(axis_length) = &axis_length {
                    if axis_length.value_id() != length.value_id() {
                        active_facts.extend(cmp_term(CmpOp::Lt, &address, axis_length));
                    }
                }
                LoopItem { value, active: Some(active), active_facts }
            })
            .collect()
    }

    /// Items of `range(...)` with integer arguments. A constant `range` is
    /// enumerated exactly; a `range` with a runtime start or stop is
    /// unrolled to a proven bound with per-item activity conditions.
    fn range_items(&mut self, args: &[Value]) -> Vec<LoopItem> {
        let (start, stop, step) = self.range_arguments(args);
        let step = step.int_val().unwrap_or_else(|| {
            panic!("range() step must be a compile-time constant integer")
        });
        if step == 0 {
            panic!("range() arg 3 must not be zero");
        }
        if let (Some(start), Some(stop)) = (start.int_val(), stop.int_val()) {
            return static_range(start, stop, step)
                .into_iter()
                .map(|v| LoopItem::always(self.builder.ir_constant_int(v)))
                .collect();
        }
        let count = self.range_count_bound(&start, &stop, step).unwrap_or_else(|| {
            panic!(
                "{}: cannot bound the number of iterations of range() because its start or stop \
                 is a runtime value without known bounds. Give the bound a maximum, e.g. a \
                 bounded array annotation (Integer[Array, \"N<=8\"]) or a precondition \
                 (@zk_circuit(requires=[lambda n: 0 <= n <= 8]))",
                UNBOUNDED_LOOP_ERROR
            )
        });
        // Items are start, start+step, ...; item k is active iff it lies
        // before `stop` in the direction of `step`.
        let (active_op, active_cmp) = if step > 0 {
            (IRBuilderCmp::Lt, CmpOp::Lt)
        } else {
            (IRBuilderCmp::Gt, CmpOp::Gt)
        };
        let item_value = |gen: &mut Self, k: i64| -> Value {
            let offset = gen.builder.ir_constant_int(k * step);
            if k == 0 {
                start.clone()
            } else {
                gen.builder.ir_add_i(&start, &offset)
            }
        };
        // Assert that the range has no item beyond the unrolled ones, so a
        // bound that is too small makes the circuit unsatisfiable rather
        // than silently dropping iterations.
        let next = item_value(self, count);
        let beyond = active_op.apply(&mut self.builder, &next, &stop);
        let exhausted = self.builder.ir_logical_not(&beyond);
        self.assert_value(&exhausted);

        let stop_bounds = self.int_bounds(&stop);
        (0..count)
            .map(|k| {
                let value = item_value(self, k);
                // Statically active when even the extreme stop admits it.
                let always_active = match (value.int_val(), stop_bounds) {
                    (Some(v), Some((lo, hi))) => {
                        if step > 0 { v < lo } else { v > hi }
                    }
                    _ => false,
                };
                if always_active {
                    return LoopItem::always(value);
                }
                let active = active_op.apply(&mut self.builder, &value, &stop);
                let active_facts = cmp_term(active_cmp, &value, &stop).into_iter().collect();
                LoopItem { value, active: Some(active), active_facts }
            })
            .collect()
    }

    /// Recognise a counted `while` loop and return its counter name and the
    /// items of the equivalent `range`.
    ///
    /// `while i < bound: ...` (also `<=`, and the decreasing forms
    /// `i > bound` / `i >= bound`) whose body changes `i` only through one
    /// top-level `i += c` (`i -= c` when decreasing) iterates exactly like
    /// `for i in range(i, bound', c): <same body>` when `i` currently holds a
    /// compile-time integer, the body has no `continue` of this loop, and
    /// changes nothing the bound reads: in both, iteration k starts with
    /// `i = i0 + k*c`, and after the loop `i` holds its value at the end of
    /// the last iteration that ran. Unrolling it that way keeps `i` a
    /// compile-time constant inside the body (a general `while` merges `i`
    /// after every guarded iteration, which makes it a runtime value), and
    /// bounds the trip count like any runtime-bounded `range`.
    pub(crate) fn counted_while_items(
        &mut self,
        n: &ASTWhileStatement,
    ) -> Option<(String, Vec<LoopItem>)> {
        let ASTNode::ASTBinaryOperator(test) = &*n.test_expr else { return None };
        let ASTNode::ASTLoad(counter) = &*test.lhs else { return None };
        let counter = counter.name.as_str();
        let increment = n.block.iter().position(|stmt| counter_increment(stmt, counter).is_some())?;
        let step = counter_increment(&n.block[increment], counter)?;
        let rest: Vec<&ASTNode> = n
            .block
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != increment)
            .map(|(_, stmt)| stmt)
            .collect();
        let increasing = match test.operator.as_str() {
            "lt" | "lte" => true,
            "gt" | "gte" => false,
            _ => return None,
        };
        if (step > 0) != increasing {
            return None;
        }
        let mut assigned = std::collections::BTreeSet::new();
        for stmt in &rest {
            Self::collect_assigned_names_from_stmt(stmt, &mut assigned);
        }
        let body = serde_json::to_value(&n.block).ok()?;
        let bound_ast = serde_json::to_value(&*test.rhs).ok()?;
        let bound_names = loaded_names(&bound_ast);
        if assigned.contains(counter)
            || bound_names.iter().any(|name| name == counter || assigned.contains(name))
            || json_any(&body, &|node| is_method_call_on(node, &bound_names))
            || n.block.iter().any(statement_continues_loop)
        {
            return None;
        }
        let start = self.ctx.get(counter)?;
        let start = self.value_on_current_path(&start);
        if start.int_val().is_none() || !matches!(start, Value::Integer(_)) {
            return None;
        }
        let bound = self.visit(&test.rhs);
        if !matches!(bound, Value::Integer(_) | Value::Boolean(_)) {
            return None;
        }
        let stop = match test.operator.as_str() {
            "lte" => {
                let one = self.builder.ir_constant_int(1);
                self.builder.ir_add_i(&bound, &one)
            }
            "gte" => {
                let one = self.builder.ir_constant_int(1);
                self.builder.ir_sub_i(&bound, &one)
            }
            _ => bound,
        };
        let step = self.builder.ir_constant_int(step);
        Some((counter.to_string(), self.range_items(&[start, stop, step])))
    }

    /// The value `v` has on every path where the current statement executes.
    /// A name assigned inside a branch is stored pre-merged as
    /// `select(c, new, old)` so that the branch join is correct; while the
    /// current execution condition implies `c` (its conjuncts include all of
    /// `c`'s), that select equals `new`.
    pub(crate) fn value_on_current_path(&mut self, v: &Value) -> Value {
        use crate::ir_defs::IR;
        let Some(stmt) = v.stmt_id().and_then(|id| self.builder.stmts.get(id as usize)).cloned()
        else {
            return v.clone();
        };
        if !matches!(stmt.ir, IR::SelectI | IR::SelectBitI) || stmt.arg_values.len() != 3 {
            return v.clone();
        }
        let (Some(select_cond), Some(then_value)) = (
            self.builder.value_of_vid(stmt.arg_values[0]),
            self.builder.value_of_vid(stmt.arg_values[1]),
        ) else {
            return v.clone();
        };
        let current = self.ctx.get_condition_value(&mut self.builder);
        let required = self.conjuncts(&select_cond);
        let holding = self.conjuncts(&current);
        if required.is_subset(&holding) {
            self.value_on_current_path(&then_value)
        } else {
            v.clone()
        }
    }

    /// Statement ids of the conjuncts of a Boolean (`a & b & ...`).
    fn conjuncts(&self, condition: &Value) -> std::collections::BTreeSet<crate::types::StmtId> {
        use crate::ir_defs::IR;
        let mut out = std::collections::BTreeSet::new();
        let mut stack: Vec<crate::types::StmtId> = condition.stmt_id().into_iter().collect();
        while let Some(id) = stack.pop() {
            match self.builder.stmts.get(id as usize) {
                Some(stmt) if matches!(stmt.ir, IR::LogicalAnd) => stack.extend(stmt.arguments.iter().copied()),
                Some(stmt) if matches!(stmt.ir, IR::ConstantBool { value: true }) => {}
                _ => {
                    out.insert(id);
                }
            }
        }
        out
    }

    /// Split `range` arguments into integer (start, stop, step) values.
    fn range_arguments(&mut self, args: &[Value]) -> (Value, Value, Value) {
        for arg in args {
            if !matches!(arg, Value::Integer(_) | Value::Boolean(_)) {
                panic!(
                    "TypeError: '{}' object cannot be interpreted as an integer (range argument)",
                    arg.zinnia_type()
                );
            }
        }
        let zero = self.builder.ir_constant_int(0);
        let one = self.builder.ir_constant_int(1);
        match args {
            [stop] => (zero, stop.clone(), one),
            [start, stop] => (start.clone(), stop.clone(), one),
            [start, stop, step] => (start.clone(), stop.clone(), step.clone()),
            _ => panic!("range expected 1 to 3 arguments, got {}", args.len()),
        }
    }

    /// An upper bound on `len(range(start, stop, step))` over all values the
    /// compiler can prove `start` and `stop` take, or `None` if either end
    /// has no provable bound in the needed direction.
    fn range_count_bound(&mut self, start: &Value, stop: &Value, step: i64) -> Option<i64> {
        let (start_lo, start_hi) = self.int_bounds(start)?;
        let (stop_lo, stop_hi) = self.int_bounds(stop)?;
        let span = if step > 0 {
            stop_hi.checked_sub(start_lo)?
        } else {
            start_hi.checked_sub(stop_lo)?
        };
        let magnitude = step.checked_abs()?;
        Some(if span <= 0 { 0 } else { (span + magnitude - 1) / magnitude })
    }

    /// Proven `(min, max)` of an integer value: exact for constants,
    /// otherwise from the builder's facts and bound analysis.
    pub(crate) fn int_bounds(&mut self, v: &Value) -> Option<(i64, i64)> {
        if let Some(n) = v.int_val() {
            return Some((n, n));
        }
        self.builder.ask_bounds(v)
    }

    /// Proven `(min, max)` of an integer value from constants, facts and
    /// arithmetic on them only. Unlike [`Self::int_bounds`] it never queries
    /// the (possibly SMT-backed) resolver, so it is cheap enough to ask once
    /// per unrolled `while` iteration.
    fn fact_bounds(&self, v: &Value) -> Option<(i64, i64)> {
        if let Some(n) = v.int_val() {
            return Some((n, n));
        }
        self.builder.deep_bounds_walk(v)
    }

    /// Record that a runtime extent of a bounded-dynamic array lies in
    /// `[0, max]`. This is the array type's invariant (input extents are
    /// checked by the dynamic-metadata assertions; derived arrays never
    /// exceed their envelope), stated as facts so that loops over the extent
    /// can be bounded. Constant extents need no fact.
    pub(crate) fn note_extent_bounds(&mut self, extent: &Value, max: usize) {
        let Some(vid) = extent.value_id() else { return };
        if extent.int_val().is_some() {
            return;
        }
        let known = crate::optim::resolver::derive_bounds_from_facts(&self.builder.facts, vid);
        if known.is_some_and(|(lo, hi)| lo >= 0 && hi <= max as i64) {
            return;
        }
        let var = || Box::new(ContractTerm::Var(ContractVar::Value(vid)));
        self.builder.facts.insert_for(
            vid,
            ContractTerm::Cmp { op: CmpOp::Ge, lhs: var(), rhs: Box::new(ContractTerm::LitInt(0)) },
        );
        self.builder.facts.insert_for(
            vid,
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: var(),
                rhs: Box::new(ContractTerm::LitInt(max as i64)),
            },
        );
    }

    /// Record the extent facts (see [`Self::note_extent_bounds`]) for the
    /// runtime length of `data` along axis 0, as returned by `len()`.
    pub(crate) fn note_len_bounds(&mut self, data: &DynamicNDArrayData) {
        if let (Some(extent), Some(dim)) = (data.meta.runtime_shape.first(), data.envelope.dims.first()) {
            self.note_extent_bounds(&Value::Integer(extent.clone()), dim.max);
        }
    }

    /// Record extent facts for the result of `.shape` (a tuple of per-axis
    /// extents) or `.size` (the total length) of a bounded-dynamic array.
    pub(crate) fn note_metadata_bounds(&mut self, data: &DynamicNDArrayData, member: &str, result: &Value) {
        match (member, result) {
            ("shape", Value::Tuple(extents)) => {
                for (extent, dim) in extents.values.iter().zip(data.envelope.dims.iter()) {
                    self.note_extent_bounds(extent, dim.max);
                }
            }
            ("size", size) => self.note_extent_bounds(size, data.envelope.total_bound),
            _ => {}
        }
    }

    /// Whether the compiler can prove that the Boolean `condition` is false,
    /// by comparing the proven bounds of the operands of the comparisons it
    /// is built from. Used to stop unrolling a `while` loop whose guard can
    /// no longer hold.
    pub(crate) fn provably_false(&self, condition: &Value) -> bool {
        use crate::ir_defs::IR;
        if let Some(b) = condition.bool_val().or_else(|| condition.int_val().map(|v| v != 0)) {
            return !b;
        }
        let Some(stmt) = condition.stmt_id().and_then(|id| self.builder.stmts.get(id as usize))
        else {
            return false;
        };
        let operand = |i: usize| -> Option<Value> {
            self.builder.value_of_vid(*stmt.arg_values.get(i)?)
        };
        match stmt.ir {
            IR::LogicalAnd => (0..2).any(|i| operand(i).is_some_and(|v| self.provably_false(&v))),
            IR::BoolCast => operand(0).is_some_and(|v| self.provably_false(&v)),
            IR::LtI | IR::LtIBounded | IR::LteI | IR::GtI | IR::GtIBounded | IR::GteI => {
                let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) = (
                    operand(0).and_then(|a| self.fact_bounds(&a)),
                    operand(1).and_then(|b| self.fact_bounds(&b)),
                ) else {
                    return false;
                };
                match stmt.ir {
                    IR::LtI | IR::LtIBounded => a_lo >= b_hi,
                    IR::LteI => a_lo > b_hi,
                    IR::GtI | IR::GtIBounded => a_hi <= b_lo,
                    _ => a_hi < b_lo, // GteI
                }
            }
            _ => false,
        }
    }

    /// Facts that hold whenever `condition` is true, derived from the
    /// comparisons it is built from (conjunctions are split). Used as path
    /// conditions for code that runs only when `condition` holds.
    pub(crate) fn condition_facts(&self, condition: &Value) -> Vec<ContractTerm> {
        use crate::ir_defs::IR;
        let mut facts = Vec::new();
        let Some(stmt) = condition.stmt_id().and_then(|id| self.builder.stmts.get(id as usize))
        else {
            return facts;
        };
        let operand = |i: usize| -> Option<Value> {
            self.builder.value_of_vid(*stmt.arg_values.get(i)?)
        };
        let op = match stmt.ir {
            IR::LtI | IR::LtIBounded => Some(CmpOp::Lt),
            IR::LteI => Some(CmpOp::Le),
            IR::GtI | IR::GtIBounded => Some(CmpOp::Gt),
            IR::GteI => Some(CmpOp::Ge),
            IR::EqI => Some(CmpOp::Eq),
            IR::NeI => Some(CmpOp::Ne),
            _ => None,
        };
        if let Some(op) = op {
            if let (Some(a), Some(b)) = (operand(0), operand(1)) {
                facts.extend(cmp_term(op, &a, &b));
            }
        } else if matches!(stmt.ir, IR::LogicalAnd | IR::BoolCast) {
            for i in 0..stmt.arg_values.len() {
                if let Some(v) = operand(i) {
                    facts.extend(self.condition_facts(&v));
                }
            }
        }
        facts
    }

    /// Run `f` as code that executes only when `guard` holds (a comprehension
    /// filter, one arm of a conditional expression): inside, the execution
    /// condition, the write predicate, chip entry conditions, and assertions
    /// all include `guard`, and the comparisons in `guard` are available as
    /// path facts. Bindings made inside do not escape.
    pub(crate) fn with_guard<T>(&mut self, guard: &Value, f: impl FnOnce(&mut Self) -> T) -> T {
        if guard.bool_val() == Some(true) {
            return f(self);
        }
        self.ctx.if_enter(guard.clone(), &mut self.builder);
        let facts = self.condition_facts(guard);
        self.builder.facts.enter(
            crate::optim::predicates::ScopeKind::Conditional,
            conjunction(facts),
        );
        let condition = self.ctx.get_condition_value(&mut self.builder);
        let previous = self.builder.swap_write_predicate(Some(condition));
        let result = f(self);
        self.builder.swap_write_predicate(previous);
        let _guard_facts = self.builder.facts.leave();
        self.ctx.if_leave();
        result
    }
}

/// The conjunction of `facts`, or `None` when there are none.
pub(crate) fn conjunction(mut facts: Vec<ContractTerm>) -> Option<ContractTerm> {
    match facts.len() {
        0 => None,
        1 => facts.pop(),
        _ => Some(ContractTerm::BoolComb { op: BoolOp::And, operands: facts }),
    }
}

/// Python's `range(start, stop, step)` for compile-time arguments.
fn static_range(start: i64, stop: i64, step: i64) -> Vec<i64> {
    let mut values = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        values.push(i);
        i += step;
    }
    values
}

/// Which comparison makes a range item active.
#[derive(Clone, Copy)]
enum IRBuilderCmp {
    Lt,
    Gt,
}

impl IRBuilderCmp {
    fn apply(self, b: &mut crate::builder::IRBuilder, lhs: &Value, rhs: &Value) -> Value {
        match self {
            IRBuilderCmp::Lt => b.ir_less_than_i(lhs, rhs),
            IRBuilderCmp::Gt => b.ir_greater_than_i(lhs, rhs),
        }
    }
}

/// The constant `c` of a final `name += c` / `name -= c` statement (the
/// transformer lowers both to `name = name ± c`).
fn counter_increment(stmt: &ASTNode, name: &str) -> Option<i64> {
    let ASTNode::ASTAssignStatement(assign) = stmt else { return None };
    let [ASTNode::ASTNameAssignTarget(target)] = assign.targets.as_slice() else { return None };
    let ASTNode::ASTBinaryOperator(update) = &*assign.value else { return None };
    let (ASTNode::ASTLoad(load), ASTNode::ASTConstantInteger(c)) = (&*update.lhs, &*update.rhs)
    else {
        return None;
    };
    if target.name != name || load.name != name || c.value == 0 {
        return None;
    }
    match update.operator.as_str() {
        "add" => Some(c.value),
        "sub" => c.value.checked_neg(),
        _ => None,
    }
}

/// Whether `stmt` contains a `continue` of the loop it belongs to (one in a
/// nested loop belongs to that loop).
fn statement_continues_loop(stmt: &ASTNode) -> bool {
    match stmt {
        ASTNode::ASTContinueStatement(_) => true,
        ASTNode::ASTCondStatement(cond) => cond
            .t_block
            .iter()
            .chain(cond.f_block.iter())
            .any(statement_continues_loop),
        _ => false,
    }
}

/// Depth-first search over a serialized AST for a node matching `pred`.
fn json_any(node: &serde_json::Value, pred: &dyn Fn(&serde_json::Value) -> bool) -> bool {
    if pred(node) {
        return true;
    }
    match node {
        serde_json::Value::Array(items) => items.iter().any(|item| json_any(item, pred)),
        serde_json::Value::Object(map) => map.values().any(|value| json_any(value, pred)),
        _ => false,
    }
}

/// Names loaded anywhere in a serialized expression.
fn loaded_names(node: &serde_json::Value) -> Vec<String> {
    let mut names = Vec::new();
    collect_loaded_names(node, &mut names);
    names
}

fn collect_loaded_names(node: &serde_json::Value, names: &mut Vec<String>) {
    match node {
        serde_json::Value::Array(items) => {
            items.iter().for_each(|item| collect_loaded_names(item, names))
        }
        serde_json::Value::Object(map) => {
            if map.get("__class__").and_then(|c| c.as_str()) == Some("ASTLoad") {
                if let Some(name) = map.get("name").and_then(|n| n.as_str()) {
                    names.push(name.to_string());
                }
            }
            map.values().for_each(|value| collect_loaded_names(value, names));
        }
        _ => {}
    }
}

/// Whether a serialized AST node calls a method on one of `names`
/// (`lst.append(v)`), which may change what the name's length reads.
fn is_method_call_on(node: &serde_json::Value, names: &[String]) -> bool {
    let Some(map) = node.as_object() else { return false };
    match map.get("__class__").and_then(|c| c.as_str()) {
        Some("ASTNamedAttribute") => map
            .get("target")
            .and_then(|t| t.as_str())
            .is_some_and(|t| names.iter().any(|n| n == t)),
        Some("ASTExprAttribute") => map
            .get("target")
            .and_then(|t| t.as_object())
            .filter(|t| t.get("__class__").and_then(|c| c.as_str()) == Some("ASTLoad"))
            .and_then(|t| t.get("name").and_then(|n| n.as_str()))
            .is_some_and(|t| names.iter().any(|n| n == t)),
        _ => false,
    }
}
