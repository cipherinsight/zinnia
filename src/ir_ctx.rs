use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::scope::*;
use crate::types::{Value, ZinniaType};

/// IRContext manages the scope stack and variable bindings during IR generation.
/// Mirrors Python `IRContext` from `zinnia/compile/ir/ir_ctx.py`.
pub struct IRContext {
    scopes: Vec<Scope>,
    pub recursion_depth: i32,
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
                var_table: HashMap::new(),
            })],
            recursion_depth: 0,
        }
    }

    // ── Variable access ───────────────────────────────────────────────

    /// Set a variable in the current scope.
    /// For LoopScope: if the variable exists in a parent scope, update it there.
    pub fn set(&mut self, key: &str, val: Value) {
        let store = ValueStore::new(val);
        let n = self.scopes.len();

        // LoopScope and ConditionalScope: if var exists in parent, set it in parent
        // This ensures mutations inside if/while blocks propagate
        if matches!(&self.scopes[n - 1], Scope::Loop(_) | Scope::Conditional(_)) {
            for i in (0..n - 1).rev() {
                if self.scopes[i].exists_local(key) {
                    self.scopes[i].set_local(key.to_string(), store);
                    return;
                }
                if self.scopes[i].is_boundary() {
                    break;
                }
            }
        }

        self.scopes.last_mut().unwrap().set_local(key.to_string(), store);
    }

    /// Get a variable value from the scope chain.
    pub fn get(&self, key: &str) -> Option<Value> {
        let n = self.scopes.len();
        let _type_locked = self.scopes[n - 1].lock_parent_variable_types();

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

    pub fn chip_enter(
        &mut self,
        return_dtype: ZinniaType,
        assertion_condition: Option<Value>,
    ) {
        self.scopes.push(Scope::Chip(ChipScopeData {
            var_table: HashMap::new(),
            return_guaranteed: false,
            return_dtype,
            returns_with_conditions: Vec::new(),
            calculated_returning_condition: None,
            assertion_condition,
        }));
    }

    pub fn chip_leave(&mut self) -> Scope {
        assert!(matches!(self.scopes.last(), Some(Scope::Chip(_))));
        self.scopes.pop().unwrap()
    }

    pub fn loop_enter(&mut self) {
        // Capture the parent's looping condition at construction time
        let super_looping = self.find_condition(|s| s.get_looping_condition());
        self.scopes.push(Scope::Loop(Box::new(LoopScopeData {
            var_table: HashMap::new(),
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
        self.scopes.pop().unwrap()
    }

    pub fn if_enter(&mut self, condition: Value, builder: &mut IRBuilder) {
        // Compute calculated_branching_condition
        let super_branching = self.find_condition(|s| s.get_branching_condition());
        let calc = match super_branching {
            Some(sb) => Some(builder.ir_logical_and(&condition, &sb)),
            None => Some(condition.clone()),
        };

        self.scopes.push(Scope::Conditional(ConditionalScopeData {
            var_table: HashMap::new(),
            condition,
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
            var_table: HashMap::new(),
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

    /// Get the combined execution condition (looping AND branching AND returning).
    /// If no conditions are active, returns a constant `true`.
    pub fn get_condition_value(&self, builder: &mut IRBuilder) -> Value {
        let cond_loop = self.find_condition(|s| s.get_looping_condition());
        let cond_branch = self.find_condition(|s| s.get_branching_condition());
        let cond_return = self.find_condition(|s| s.get_returning_condition());

        let mut conditions: Vec<Value> = Vec::new();
        if let Some(c) = cond_loop { conditions.push(c); }
        if let Some(c) = cond_branch { conditions.push(c); }
        if let Some(c) = cond_return { conditions.push(c); }

        match conditions.len() {
            0 => builder.ir_constant_bool(true),
            1 => conditions.into_iter().next().unwrap(),
            2 => builder.ir_logical_and(&conditions[0], &conditions[1]),
            3 => {
                let inner = builder.ir_logical_and(&conditions[1], &conditions[2]);
                builder.ir_logical_and(&conditions[0], &inner)
            }
            _ => unreachable!(),
        }
    }

    /// Get condition value including assertion condition.
    pub fn get_condition_value_for_assertion(&self, builder: &mut IRBuilder) -> Value {
        let assertion = self.find_condition(|s| s.get_assertion_condition());
        let base = self.get_condition_value(builder);
        match assertion {
            None => base,
            Some(a) => builder.ir_logical_and(&a, &base),
        }
    }

    pub fn get_break_condition_value(&self, builder: &mut IRBuilder) -> Value {
        match self.find_condition(|s| s.get_breaking_condition()) {
            Some(c) => c,
            None => builder.ir_constant_bool(true),
        }
    }

    pub fn get_return_condition_value(&self, builder: &mut IRBuilder) -> Value {
        match self.find_condition(|s| s.get_returning_condition()) {
            Some(c) => c,
            None => builder.ir_constant_bool(true),
        }
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

    pub fn register_return(&mut self, value: Value, builder: &mut IRBuilder) {
        let cond = self.get_condition_value(builder);
        // Find the nearest ChipScope
        let n = self.scopes.len();
        for i in (0..n).rev() {
            if matches!(&self.scopes[i], Scope::Chip(_)) {
                self.scopes[i].register_return(value, cond, builder);
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
            if let Scope::Chip(_) = scope {
                return ret.to_vec();
            }
        }
        Vec::new()
    }

    pub fn get_return_dtype(&self) -> Option<ZinniaType> {
        for scope in self.scopes.iter().rev() {
            if let Some(dt) = scope.get_return_dtype() {
                return Some(dt.clone());
            }
        }
        None
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
        self.scopes.last().is_some_and(|s| s.is_terminated_guaranteed())
    }

    pub fn set_terminated_guarantee(&mut self) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.set_terminated_guarantee();
        }
    }

    // ── Recursion depth ───────────────────────────────────────────────

    pub fn add_recursion_depth(&mut self) {
        self.recursion_depth += 1;
    }

    pub fn sub_recursion_depth(&mut self) {
        self.recursion_depth -= 1;
    }

    pub fn get_recursion_depth(&self) -> i32 {
        self.recursion_depth
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

    /// Get a reference to the current (top) scope.
    pub fn current_scope(&self) -> &Scope {
        self.scopes.last().unwrap()
    }

    /// Get a mutable reference to the current (top) scope.
    pub fn current_scope_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().unwrap()
    }

    /// Get the number of scopes on the stack.
    pub fn scope_depth(&self) -> usize {
        self.scopes.len()
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

        ctx.chip_enter(ZinniaType::Integer, None);
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
        ctx.chip_enter(ZinniaType::None, None);
        ctx.set("i", Value::Integer(ScalarValue::known(0, 0)));

        // Enter loop scope
        ctx.loop_enter();

        // Writing to "i" should update the parent (chip scope) since it exists there
        ctx.set("i", Value::Integer(ScalarValue::known(1, 1)));

        // New var should stay in loop scope
        ctx.set("loop_local", Value::Integer(ScalarValue::known(99, 2)));

        ctx.loop_leave();

        // "i" should have been updated in chip scope
        assert_eq!(ctx.get("i").unwrap().int_val(), Some(1));
        // "loop_local" should NOT exist (was in loop's local table)
        assert!(!ctx.exists("loop_local"));
    }

    #[test]
    fn test_conditional_scope() {
        let mut ctx = IRContext::new();
        let mut builder = IRBuilder::new();

        ctx.chip_enter(ZinniaType::None, None);
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

        ctx.chip_enter(ZinniaType::None, None);

        // Still no conditions in chip scope
        let cond = ctx.get_condition_value(&mut builder);
        assert_eq!(cond.bool_val(), Some(true));

        ctx.chip_leave();
    }

    #[test]
    fn test_recursion_depth() {
        let mut ctx = IRContext::new();
        assert_eq!(ctx.get_recursion_depth(), 0);
        ctx.add_recursion_depth();
        assert_eq!(ctx.get_recursion_depth(), 1);
        ctx.sub_recursion_depth();
        assert_eq!(ctx.get_recursion_depth(), 0);
    }
}
