use std::collections::{BTreeMap, HashMap};

use crate::builder::IRBuilder;
use crate::scope::*;
use crate::types::Value;

/// IRContext manages the scope stack and variable bindings during IR generation.
/// Mirrors Python `IRContext` from `zinnia/compile/ir/ir_ctx.py`.
pub struct IRContext {
    scopes: Vec<Scope>,
    unreachable_loop_bindings: HashMap<String, String>,
    /// Per function (innermost last): the names that may be unbound where
    /// they are read, each with the execution predicates of the assignments
    /// to it so far. Python binds a local name exactly where one of its
    /// assignments ran, and reading it elsewhere raises UnboundLocalError.
    /// A name first bound under a runtime predicate (in a branch, after a
    /// `break` or `continue`, in a loop that may run no iteration) is
    /// recorded here; a branch join replaces the arms' predicates by the
    /// join's own where both arms bind the name on every path through them
    /// (`IRGenerator::join_assignment_records`), and an assignment on every
    /// path ends the record (reviews 3 and 4, B4, R2, FR4).
    possibly_unbound: Vec<BTreeMap<String, Vec<Value>>>,
}

impl Default for IRContext {
    fn default() -> Self {
        Self::new()
    }
}

impl IRContext {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::Master(MasterScopeData {
                var_table: VarTable::new(),
            })],
            unreachable_loop_bindings: HashMap::new(),
            possibly_unbound: vec![BTreeMap::new()],
        }
    }

    // ── Variable access ───────────────────────────────────────────────

    /// Set a variable in the current scope.
    ///
    /// Inside a loop, an assignment to a name bound outside the loop updates
    /// that binding (Python has no per-iteration scope), so the next
    /// iteration and the code after the loop see it. The update goes up the
    /// scope chain to the scope holding the name, but never past a
    /// conditional scope: a branch's bindings are joined with the other
    /// branch's when the `if` ends (`merge_conditional_branch_bindings`), so
    /// a write inside a branch must land in that branch's own scope. (It
    /// used to go straight to the scope outside the `if`; the join then
    /// replaced it with the other branch's value, so `if c: s = s + 100
    /// else: for k in range(2): s = s + 1` gave 0, not 2, where `c` is
    /// false: review finding N6.)
    pub fn set(&mut self, key: &str, val: Value) {
        self.unreachable_loop_bindings.remove(key);
        let store = ValueStore::new(val);
        // A comprehension's scope holds its targets only: another name
        // assigned while evaluating it (a list of the function mutated by a
        // method call) is the enclosing scope's.
        let mut n = self.scopes.len();
        while n > 1
            && matches!(self.scopes[n - 1], Scope::Generator(_))
            && !self.scopes[n - 1].exists_local(key)
            && self.exists_below(n - 1, key)
        {
            n -= 1;
        }

        if matches!(&self.scopes[n - 1], Scope::Loop(_)) {
            // The innermost conditional scope between the loop and the scope
            // holding the name, if any.
            let mut branch: Option<usize> = None;
            for i in (0..n - 1).rev() {
                if self.scopes[i].exists_local(key) {
                    let target = branch.unwrap_or(i);
                    self.scopes[target].set_local(key.to_string(), store);
                    return;
                }
                if branch.is_none() && matches!(self.scopes[i], Scope::Conditional(_)) {
                    branch = Some(i);
                }
                if self.scopes[i].is_boundary() {
                    break;
                }
            }
        }

        self.scopes[n - 1].set_local(key.to_string(), store);
    }

    /// Make `key` a name of the innermost comprehension (its target), so
    /// that binding it does not touch a name of the enclosing scope.
    pub fn declare_comprehension_target(&mut self, key: &str) {
        let top = self.scopes.last_mut().expect("a scope");
        assert!(matches!(top, Scope::Generator(_)), "a comprehension target outside a comprehension");
        if !top.exists_local(key) {
            top.set_local(key.to_string(), ValueStore::new(Value::None));
        }
    }

    /// Whether `key` is bound in a scope below index `top` of the current
    /// function.
    fn exists_below(&self, top: usize, key: &str) -> bool {
        for scope in self.scopes[..top].iter().rev() {
            if scope.exists_local(key) {
                return true;
            }
            if scope.is_boundary() {
                break;
            }
        }
        false
    }

    /// Get a variable value from the scope chain.
    pub fn get(&self, key: &str) -> Option<Value> {
        let n = self.scopes.len();

        // Search from current scope upward
        for i in (0..n).rev() {
            if let Some(store) = self.scopes[i].get_local(key) {
                return Some(store.value.clone());
            }
            if self.scopes[i].is_boundary() {
                break;
            }
        }
        None
    }

    /// The name of a binding other than `name`'s own (in any scope, including
    /// callers' scopes outside the current chip) whose value satisfies
    /// `holds`. Used to find a second way to reach a Python list.
    ///
    /// `name` in any scope of the current function is `name` itself: a
    /// branch or loop scope holds the variable's value on its paths, the
    /// enclosing scope its value before them (a loop's writes inside a
    /// branch land in the branch's scope, see [`Self::set`]).
    pub fn find_other_binding(&self, name: &str, holds: impl Fn(&Value) -> bool) -> Option<String> {
        let function = (0..self.scopes.len()).rev().find(|&i| self.scopes[i].is_boundary()).unwrap_or(0);
        self.scopes.iter().enumerate().find_map(|(i, scope)| {
            scope.var_table().iter().find_map(|(other, store)| {
                let is_own = i >= function && other == name;
                (!is_own && holds(&store.value)).then(|| other.clone())
            })
        })
    }

    /// Mark a name that is syntactically introduced only inside a loop body
    /// that was proven not to execute. This is intentionally narrow: it is
    /// not a general definite-assignment analysis, just a better diagnostic
    /// for post-loop loads that would otherwise become a generic missing var.
    pub fn mark_unreachable_loop_binding(&mut self, key: &str, reason: String) {
        if !self.exists(key) {
            self.unreachable_loop_bindings
                .entry(key.to_string())
                .or_insert(reason);
        }
    }

    pub fn unreachable_loop_binding_reason(&self, key: &str) -> Option<&str> {
        if self.exists(key) {
            return None;
        }
        self.unreachable_loop_bindings.get(key).map(String::as_str)
    }

    /// Record an assignment to `key` that runs under `predicate` (`None`:
    /// on every path). An assignment on every path binds the name from then
    /// on; a first binding under a runtime predicate starts a record, and a
    /// later assignment to a recorded name adds its predicate (see
    /// `possibly_unbound`).
    pub fn note_assignment(&mut self, key: &str, predicate: Option<&Value>) {
        let everywhere = predicate.is_none_or(|p| p.bool_val() == Some(true) || p.int_val().is_some_and(|v| v != 0));
        let first = !self.exists(key);
        let names = self.possibly_unbound.last_mut().expect("a function scope");
        if everywhere {
            names.remove(key);
        } else if let Some(p) = predicate {
            if let Some(predicates) = names.get_mut(key) {
                predicates.push(p.clone());
            } else if first {
                names.insert(key.to_string(), vec![p.clone()]);
            }
        }
    }

    /// The records of possibly unbound names of the current function (see
    /// `possibly_unbound`), to save before a branch arm.
    pub fn assignment_records(&self) -> BTreeMap<String, Vec<Value>> {
        self.possibly_unbound.last().cloned().unwrap_or_default()
    }

    /// Replace the records of the current function (the second arm of a
    /// branch starts from the records before the first arm; the join sets
    /// the merged records).
    pub fn set_assignment_records(&mut self, records: BTreeMap<String, Vec<Value>>) {
        *self.possibly_unbound.last_mut().expect("a function scope") = records;
    }

    /// The innermost loop's condition for its current iteration to run on
    /// (not broken, not skipped), `None` when it runs on every path where
    /// the loop does.
    pub fn looping_condition(&self) -> Option<Value> {
        self.find_condition(|s| s.get_looping_condition())
    }

    /// Whether the innermost loop may have been left by a `break` so far.
    pub fn loop_may_break(&self) -> bool {
        self.scopes.iter().rev().find_map(|s| match s {
            Scope::Loop(d) => Some(d.break_condition.as_ref().is_some_and(|not_broken| {
                not_broken.bool_val() != Some(true) && not_broken.int_val().is_none_or(|v| v == 0)
            })),
            _ => None,
        }) == Some(true)
    }

    /// The predicates of the assignments to `key` when it may be unbound
    /// (see [`Self::note_assignment`]); it is bound where one of them held.
    pub fn possibly_unbound(&self, key: &str) -> Option<&[Value]> {
        // A comprehension's own target is bound wherever it is read.
        for scope in self.scopes.iter().rev() {
            if matches!(scope, Scope::Generator(_)) && scope.exists_local(key) {
                return None;
            }
            if scope.is_boundary() {
                break;
            }
        }
        self.possibly_unbound.last()?.get(key).map(Vec::as_slice)
    }

    /// Check if a variable exists in the scope chain.
    pub fn exists(&self, key: &str) -> bool {
        let n = self.scopes.len();
        for i in (0..n).rev() {
            if self.scopes[i].exists_local(key) {
                return true;
            }
            if self.scopes[i].is_boundary() {
                break;
            }
        }
        false
    }

    // ── Scope enter/leave ─────────────────────────────────────────────

    /// Enter a chip body. `entry_condition` is the caller's execution
    /// condition at the call site ([`Self::get_condition_value`] evaluated
    /// before entering); pass `None` when it is statically true. The chip's
    /// own conditions are combined with it, never substituted for it, so a
    /// chip called inside a private `if`, after a private `break`, or after
    /// an early `return` cannot write, assert, or return on paths where the
    /// call does not happen.
    pub fn chip_enter(&mut self, entry_condition: Option<Value>) {
        self.possibly_unbound.push(BTreeMap::new());
        self.scopes.push(Scope::Chip(ChipScopeData {
            var_table: VarTable::new(),
            return_guaranteed: false,
            returns_with_conditions: Vec::new(),
            calculated_returning_condition: None,
            entry_condition,
            has_none_return: false,
            has_value_return: false,
        }));
    }

    pub fn circuit_enter(&mut self) {
        self.possibly_unbound.push(BTreeMap::new());
        self.scopes.push(Scope::Circuit(CircuitScopeData {
            var_table: VarTable::new(),
            return_guaranteed: false,
            returns_with_conditions: Vec::new(),
            calculated_returning_condition: None,
            has_none_return: false,
            has_value_return: false,
        }));
    }

    pub fn circuit_leave(&mut self) -> Scope {
        assert!(matches!(self.scopes.last(), Some(Scope::Circuit(_))));
        self.possibly_unbound.pop();
        self.scopes.pop().unwrap()
    }

    pub fn chip_leave(&mut self) -> Scope {
        assert!(matches!(self.scopes.last(), Some(Scope::Chip(_))));
        self.possibly_unbound.pop();
        self.scopes.pop().unwrap()
    }

    pub fn loop_enter(&mut self) {
        // Capture the parent's looping condition at construction time
        let super_looping = self.find_condition(|s| s.get_looping_condition());
        self.scopes.push(Scope::Loop(Box::new(LoopScopeData {
            var_table: VarTable::new(),
            continue_condition: None,
            break_condition: None,
            return_guaranteed: false,
            loop_terminated_guaranteed: false,
            calculated_looping_condition: None,
            super_looping_condition: super_looping,
        })));
    }

    pub fn loop_leave(&mut self) -> Scope {
        assert!(matches!(self.scopes.last(), Some(Scope::Loop(_))));
        let popped = self.scopes.pop().unwrap();

        // Python loop-scope semantics: names assigned in the body remain
        // visible after the loop terminates (no fresh scope per iteration).
        // `set()` already promotes writes that target a parent-chain name
        // into that parent, so the loop's local `var_table` by construction
        // holds only names newly introduced inside the loop body. Leak
        // those out to the immediate enclosing scope on exit. If that
        // parent is itself a Loop / Conditional / Generator, the binding
        // continues to propagate / merge on its own leave / merge path.
        //
        // Card: compiler.loop-variable-scope-leak (intake, p3, 2026-05-16).
        if let Scope::Loop(data) = &popped {
            if let Some(parent) = self.scopes.last_mut() {
                for (name, store) in data.var_table.iter() {
                    parent.set_local(name.clone(), store.clone());
                }
            }
        }

        popped
    }

    pub fn if_enter(&mut self, condition: Value, builder: &mut IRBuilder) {
        // Compute calculated_branching_condition
        let super_branching = self.find_condition(|s| s.get_branching_condition());
        let calc = match super_branching {
            Some(sb) => Some(builder.ir_logical_and(&condition, &sb)),
            None => Some(condition.clone()),
        };

        self.scopes.push(Scope::Conditional(ConditionalScopeData {
            var_table: VarTable::new(),
            return_guaranteed: false,
            loop_terminated_guaranteed: false,
            calculated_branching_condition: calc,
        }));
    }

    pub fn if_leave(&mut self) -> Scope {
        assert!(matches!(self.scopes.last(), Some(Scope::Conditional(_))));
        self.scopes.pop().unwrap()
    }

    pub fn generator_enter(&mut self) {
        self.scopes.push(Scope::Generator(GeneratorScopeData {
            var_table: VarTable::new(),
        }));
    }

    pub fn generator_leave(&mut self) -> Scope {
        assert!(matches!(self.scopes.last(), Some(Scope::Generator(_))));
        self.scopes.pop().unwrap()
    }

    // ── Query helpers ─────────────────────────────────────────────────

    pub fn is_in_chip(&self) -> bool {
        self.find_in_scope_chain(|s| s.is_in_chip())
    }

    pub fn is_in_loop(&self) -> bool {
        self.find_in_scope_chain(|s| s.is_in_loop())
    }

    pub fn is_in_conditional(&self) -> bool {
        matches!(self.scopes.last(), Some(Scope::Conditional(_)))
    }

    /// Returns true if there's any active condition (branching, looping, or returning)
    /// that makes simple assignment incorrect (i.e., we need conditional select).
    pub fn has_nontrivial_condition(&self) -> bool {
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::Conditional(_) | Scope::Loop(_) => return true,
                Scope::Chip(_) => return false, // Stop at chip boundary
                _ => {}
            }
        }
        false
    }

    // ── Condition values ──────────────────────────────────────────────

    /// The condition under which the statement being generated executes:
    /// the enclosing loop's looping condition (not broken, not continued),
    /// the enclosing branch conditions, the "not yet returned" condition,
    /// and, inside a chip, the caller's condition at the call site. Returns
    /// a constant `true` when no condition is active.
    pub fn get_condition_value(&self, builder: &mut IRBuilder) -> Value {
        let conditions: Vec<Value> = [
            self.find_condition(|s| s.get_looping_condition()),
            self.find_condition(|s| s.get_branching_condition()),
            self.find_condition(|s| s.get_returning_condition()),
            self.find_condition(|s| s.get_entry_condition()),
        ]
        .into_iter()
        .flatten()
        .collect();
        let mut iter = conditions.into_iter();
        let Some(first) = iter.next() else {
            return builder.ir_constant_bool(true);
        };
        iter.fold(first, |acc, c| builder.ir_logical_and(&acc, &c))
    }

    // ── Loop control ──────────────────────────────────────────────────

    pub fn loop_reiter(&mut self, builder: &mut IRBuilder) {
        let n = self.scopes.len();
        // Find the nearest LoopScope
        for i in (0..n).rev() {
            if matches!(&self.scopes[i], Scope::Loop(_)) {
                self.scopes[i].loop_reiterate(builder);
                return;
            }
        }
        panic!("loop_reiter called outside of loop");
    }

    pub fn loop_break(&mut self, condition: Option<Value>, builder: &mut IRBuilder) {
        let cond = condition.unwrap_or_else(|| self.get_condition_value(builder));
        let n = self.scopes.len();
        for i in (0..n).rev() {
            if matches!(&self.scopes[i], Scope::Loop(_)) {
                self.scopes[i].loop_break(cond, builder);
                return;
            }
        }
        panic!("loop_break called outside of loop");
    }

    /// Skip the rest of the current iteration of the nearest loop when
    /// `condition` holds, exactly like a `continue` under that condition:
    /// the skip ends at the next [`Self::loop_reiter`], and it does not count
    /// as a `break` for a `for ... else` block.
    pub fn loop_skip_iteration_if(&mut self, condition: Value, builder: &mut IRBuilder) {
        let n = self.scopes.len();
        for i in (0..n).rev() {
            if matches!(&self.scopes[i], Scope::Loop(_)) {
                self.scopes[i].loop_continue(condition, builder);
                return;
            }
        }
        panic!("loop_skip_iteration_if called outside of loop");
    }

    pub fn loop_continue(&mut self, builder: &mut IRBuilder) {
        let cond = self.get_condition_value(builder);
        let n = self.scopes.len();
        for i in (0..n).rev() {
            if matches!(&self.scopes[i], Scope::Loop(_)) {
                self.scopes[i].loop_continue(cond, builder);
                return;
            }
        }
        panic!("loop_continue called outside of loop");
    }

    // ── Return management ─────────────────────────────────────────────

    pub fn register_return(&mut self, value: Value, is_bare_return: bool, builder: &mut IRBuilder) {
        let cond = self.get_condition_value(builder);
        // Find the nearest ChipScope
        let n = self.scopes.len();
        for i in (0..n).rev() {
            if matches!(&self.scopes[i], Scope::Chip(_) | Scope::Circuit(_)) {
                self.scopes[i].register_return(value, cond, is_bare_return, builder);
                return;
            }
        }
        panic!("register_return called outside of chip");
    }

    pub fn get_returns_with_conditions(&self) -> Vec<(Value, Value)> {
        for scope in self.scopes.iter().rev() {
            let ret = scope.get_returns_with_conditions();
            if !ret.is_empty() {
                return ret.to_vec();
            }
            if matches!(scope, Scope::Chip(_) | Scope::Circuit(_)) {
                return ret.to_vec();
            }
        }
        Vec::new()
    }

    pub fn has_mixed_bare_and_value_returns(&self) -> bool {
        for scope in self.scopes.iter().rev() {
            if matches!(scope, Scope::Chip(_) | Scope::Circuit(_)) {
                return scope.has_mixed_bare_and_value_returns();
            }
        }
        false
    }

    pub fn check_return_guaranteed(&self) -> bool {
        self.scopes.last().is_some_and(|s| s.is_return_guaranteed())
    }

    pub fn set_return_guarantee(&mut self) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.set_return_guarantee();
        }
    }

    pub fn check_loop_terminated_guaranteed(&self) -> bool {
        self.scopes
            .last()
            .is_some_and(|s| s.is_terminated_guaranteed())
    }

    pub fn set_terminated_guarantee(&mut self) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.set_terminated_guarantee();
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Search the scope chain for a condition, walking up through delegating scopes.
    fn find_condition(&self, f: impl Fn(&Scope) -> Option<&Value>) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = f(scope) {
                return Some(val.clone());
            }
            // Only Conditional, Loop, and Generator delegate to parent
            if scope.is_boundary() {
                break;
            }
        }
        None
    }

    /// Search the scope chain for a boolean property.
    fn find_in_scope_chain(&self, f: impl Fn(&Scope) -> bool) -> bool {
        for scope in self.scopes.iter().rev() {
            if f(scope) {
                return true;
            }
            if scope.is_boundary() {
                break;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ScalarValue;

    #[test]
    fn test_basic_variable_access() {
        let mut ctx = IRContext::new();
        let val = Value::Integer(ScalarValue::known(42, 0));
        ctx.set("x", val);

        assert!(ctx.exists("x"));
        assert!(!ctx.exists("y"));
        assert_eq!(ctx.get("x").unwrap().int_val(), Some(42));
    }

    #[test]
    fn test_chip_scope() {
        let mut ctx = IRContext::new();

        ctx.set("global_var", Value::Integer(ScalarValue::known(1, 0)));

        ctx.chip_enter(None);
        ctx.set("local_var", Value::Integer(ScalarValue::known(2, 1)));

        assert!(ctx.exists("local_var"));
        assert!(!ctx.exists("global_var")); // ChipScope is a boundary
        assert!(ctx.is_in_chip());

        let _chip = ctx.chip_leave();
        assert!(!ctx.is_in_chip());
        assert!(!ctx.exists("local_var"));
        assert!(ctx.exists("global_var"));
    }

    #[test]
    fn test_loop_scope_parent_write() {
        let mut ctx = IRContext::new();

        // Set var in master scope
        ctx.set("i", Value::Integer(ScalarValue::known(0, 0)));

        // Enter chip scope (creates boundary)
        ctx.chip_enter(None);
        ctx.set("i", Value::Integer(ScalarValue::known(0, 0)));

        // Enter loop scope
        ctx.loop_enter();

        // Writing to "i" should update the parent (chip scope) since it exists there.
        ctx.set("i", Value::Integer(ScalarValue::known(1, 1)));

        // A newly-introduced name lives in the loop's local var_table during the
        // body, then leaks to the parent on loop exit (Python flat-scope semantics).
        ctx.set("loop_local", Value::Integer(ScalarValue::known(99, 2)));

        ctx.loop_leave();

        // "i" should have been updated in chip scope
        assert_eq!(ctx.get("i").unwrap().int_val(), Some(1));
        // "loop_local" leaked to the chip scope on loop exit
        // (compiler.loop-variable-scope-leak fix).
        assert!(ctx.exists("loop_local"));
        assert_eq!(ctx.get("loop_local").unwrap().int_val(), Some(99));
    }

    #[test]
    fn test_loop_introduced_binding_leaks_to_parent() {
        // Regression: compiler.loop-variable-scope-leak.
        // Names assigned only inside a loop body must remain visible after
        // the loop terminates, mirroring Python's flat loop scope.
        let mut ctx = IRContext::new();

        ctx.chip_enter(None);
        ctx.loop_enter();

        // `x` is new (not in chip / master) — stays in loop var_table during body.
        ctx.set("x", Value::Integer(ScalarValue::known(42, 0)));

        ctx.loop_leave();

        // After loop exit, `x` should still resolve from the enclosing chip scope.
        assert!(ctx.exists("x"));
        assert_eq!(ctx.get("x").unwrap().int_val(), Some(42));
    }

    #[test]
    fn test_loop_leak_nested_through_outer_loop() {
        // Nested loops: a binding introduced in the inner loop should leak
        // first into the outer loop, and then into the chip scope when the
        // outer loop terminates.
        let mut ctx = IRContext::new();

        ctx.chip_enter(None);
        ctx.loop_enter(); // outer
        ctx.loop_enter(); // inner

        ctx.set("x", Value::Integer(ScalarValue::known(7, 0)));

        ctx.loop_leave(); // inner -> outer
                          // Mid-leak invariant: `x` is now visible from the outer loop scope.
        assert_eq!(ctx.get("x").unwrap().int_val(), Some(7));

        ctx.loop_leave(); // outer -> chip
        assert!(ctx.exists("x"));
        assert_eq!(ctx.get("x").unwrap().int_val(), Some(7));
    }

    #[test]
    fn test_loop_leak_does_not_overwrite_existing_parent_binding() {
        // `set()` promotes loop writes into the parent if the name already
        // exists there, so loop_leave has nothing to leak for shadowed names
        // and must not clobber the parent's already-updated value.
        let mut ctx = IRContext::new();

        ctx.chip_enter(None);
        ctx.set("i", Value::Integer(ScalarValue::known(0, 0)));

        ctx.loop_enter();
        // Update of existing name is promoted to chip by `set()`.
        ctx.set("i", Value::Integer(ScalarValue::known(5, 1)));
        // New name leaks on exit.
        ctx.set("acc", Value::Integer(ScalarValue::known(99, 2)));
        ctx.loop_leave();

        assert_eq!(ctx.get("i").unwrap().int_val(), Some(5));
        assert_eq!(ctx.get("acc").unwrap().int_val(), Some(99));
    }

    #[test]
    fn test_conditional_scope() {
        let mut ctx = IRContext::new();
        let mut builder = IRBuilder::new();

        ctx.chip_enter(None);
        ctx.set("x", Value::Integer(ScalarValue::known(0, 0)));

        let cond = builder.ir_constant_bool(true);
        ctx.if_enter(cond, &mut builder);

        // Write in conditional scope stays local
        ctx.set("x", Value::Integer(ScalarValue::known(1, 1)));
        ctx.set("y", Value::Integer(ScalarValue::known(2, 2)));

        assert!(ctx.is_in_conditional());

        let _scope = ctx.if_leave();
        assert!(!ctx.is_in_conditional());

        // "x" in chip scope is still the old value (conditional writes are local)
        assert_eq!(ctx.get("x").unwrap().int_val(), Some(0));
        // "y" doesn't exist in chip scope
        assert!(!ctx.exists("y"));
    }

    #[test]
    fn test_condition_value_composition() {
        let mut ctx = IRContext::new();
        let mut builder = IRBuilder::new();

        // No conditions → constant true
        let cond = ctx.get_condition_value(&mut builder);
        assert_eq!(cond.bool_val(), Some(true));

        ctx.chip_enter(None);

        // Still no conditions in chip scope
        let cond = ctx.get_condition_value(&mut builder);
        assert_eq!(cond.bool_val(), Some(true));

        ctx.chip_leave();
    }

    #[test]
    fn test_chip_entry_condition_is_combined_not_replaced() {
        // Regression (audit X1): a chip called under a private condition must
        // keep that condition for every statement in its body, combined with
        // the chip's own branch conditions.
        let mut ctx = IRContext::new();
        let mut builder = IRBuilder::new();
        let caller_cond = builder.ir_read_integer(
            crate::circuit_input::InputPath::new("c", vec![]),
            false,
        );
        ctx.chip_enter(Some(caller_cond.clone()));
        let at_entry = ctx.get_condition_value(&mut builder);
        assert_eq!(at_entry.stmt_id(), caller_cond.stmt_id());

        let local = builder.ir_read_integer(
            crate::circuit_input::InputPath::new("d", vec![]),
            false,
        );
        ctx.if_enter(local.clone(), &mut builder);
        let in_branch = ctx.get_condition_value(&mut builder);
        assert_ne!(in_branch.stmt_id(), local.stmt_id());
        assert_ne!(in_branch.stmt_id(), caller_cond.stmt_id());
        let stmt = &builder.stmts[in_branch.stmt_id().unwrap() as usize];
        assert!(matches!(stmt.ir, crate::ir_defs::IR::LogicalAnd));
        ctx.if_leave();
        ctx.chip_leave();
    }
}
