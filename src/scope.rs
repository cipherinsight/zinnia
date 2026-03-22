use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{Value, ZinniaType};

// ---------------------------------------------------------------------------
// ValueStore — wraps a Value for scope variable storage
// ---------------------------------------------------------------------------

/// A stored variable binding. In Python this was `ValueStore` / `ValueTriplet`.
/// In Rust, `Value` already carries both the static value and IR pointer,
/// so `ValueStore` is a thin wrapper that enables interior mutability via
/// clone-on-write semantics.
#[derive(Debug, Clone)]
pub struct ValueStore {
    pub value: Value,
    pub type_locked: bool,
}

impl ValueStore {
    pub fn new(value: Value) -> Self {
        Self {
            value,
            type_locked: false,
        }
    }

    pub fn new_locked(value: Value) -> Self {
        Self {
            value,
            type_locked: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Scope enum — replaces the Python AbstractScope hierarchy
// ---------------------------------------------------------------------------

/// All scope types unified as a Rust enum.
/// Each variant holds the scope-specific state.
#[derive(Debug)]
pub enum Scope {
    Master(MasterScopeData),
    Chip(ChipScopeData),
    Conditional(ConditionalScopeData),
    Loop(Box<LoopScopeData>),
    Generator(GeneratorScopeData),
}

// ---------------------------------------------------------------------------
// MasterScope — root scope
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MasterScopeData {
    pub var_table: HashMap<String, ValueStore>,
}

// ---------------------------------------------------------------------------
// ChipScope — function/chip scope
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ChipScopeData {
    pub var_table: HashMap<String, ValueStore>,
    pub return_guaranteed: bool,
    pub return_dtype: ZinniaType,
    pub returns_with_conditions: Vec<(Value, Value)>, // (return_value, condition)
    pub calculated_returning_condition: Option<Value>, // BooleanValue
    pub assertion_condition: Option<Value>,             // BooleanValue
}

// ---------------------------------------------------------------------------
// ConditionalScope — if/else branch scope
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ConditionalScopeData {
    pub var_table: HashMap<String, ValueStore>,
    pub condition: Value, // BooleanValue
    pub return_guaranteed: bool,
    pub loop_terminated_guaranteed: bool,
    pub calculated_branching_condition: Option<Value>, // BooleanValue
}

// ---------------------------------------------------------------------------
// LoopScope — loop iteration scope
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct LoopScopeData {
    pub var_table: HashMap<String, ValueStore>,
    pub continue_condition: Option<Value>, // BooleanValue
    pub break_condition: Option<Value>,    // BooleanValue
    pub return_guaranteed: bool,
    pub loop_terminated_guaranteed: bool,
    pub calculated_looping_condition: Option<Value>,    // BooleanValue
    pub super_looping_condition: Option<Value>,         // BooleanValue (captured at construction)
}

// ---------------------------------------------------------------------------
// GeneratorScope — generator/comprehension scope
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GeneratorScopeData {
    pub var_table: HashMap<String, ValueStore>,
}

// ---------------------------------------------------------------------------
// Scope implementation
// ---------------------------------------------------------------------------

impl Scope {
    // ── Variable access ───────────────────────────────────────────────

    /// Set a variable in this scope.
    /// LoopScope: if variable exists in parent chain, it must be set there instead.
    /// This is handled by IRContext which checks parent first for LoopScope.
    pub fn set_local(&mut self, name: String, store: ValueStore) {
        match self {
            Scope::Master(d) => { d.var_table.insert(name, store); }
            Scope::Chip(d) => { d.var_table.insert(name, store); }
            Scope::Conditional(d) => { d.var_table.insert(name, store); }
            Scope::Loop(d) => { d.var_table.insert(name, store); }
            Scope::Generator(d) => { d.var_table.insert(name, store); }
        }
    }

    /// Get a variable from this scope's local table only.
    pub fn get_local(&self, name: &str) -> Option<&ValueStore> {
        match self {
            Scope::Master(d) => d.var_table.get(name),
            Scope::Chip(d) => d.var_table.get(name),
            Scope::Conditional(d) => d.var_table.get(name),
            Scope::Loop(d) => d.var_table.get(name),
            Scope::Generator(d) => d.var_table.get(name),
        }
    }

    /// Check if variable exists in this scope's local table.
    pub fn exists_local(&self, name: &str) -> bool {
        match self {
            Scope::Master(d) => d.var_table.contains_key(name),
            Scope::Chip(d) => d.var_table.contains_key(name),
            Scope::Conditional(d) => d.var_table.contains_key(name),
            Scope::Loop(d) => d.var_table.contains_key(name),
            Scope::Generator(d) => d.var_table.contains_key(name),
        }
    }

    /// Whether this scope delegates variable lookup to parent.
    pub fn delegates_to_parent(&self) -> bool {
        matches!(self, Scope::Conditional(_) | Scope::Loop(_) | Scope::Generator(_))
    }

    /// Whether this scope is a "boundary" (no parent lookup for get/exists).
    pub fn is_boundary(&self) -> bool {
        matches!(self, Scope::Master(_) | Scope::Chip(_))
    }

    // ── Control flow conditions ───────────────────────────────────────

    pub fn is_in_chip(&self) -> bool {
        matches!(self, Scope::Chip(_))
    }

    pub fn is_in_loop(&self) -> bool {
        matches!(self, Scope::Loop(_))
    }

    pub fn get_branching_condition(&self) -> Option<&Value> {
        match self {
            Scope::Conditional(d) => d.calculated_branching_condition.as_ref(),
            _ => None,
        }
    }

    pub fn get_looping_condition(&self) -> Option<&Value> {
        match self {
            Scope::Loop(d) => d.calculated_looping_condition.as_ref(),
            _ => None,
        }
    }

    pub fn get_breaking_condition(&self) -> Option<&Value> {
        match self {
            Scope::Loop(d) => d.break_condition.as_ref(),
            _ => None,
        }
    }

    pub fn get_returning_condition(&self) -> Option<&Value> {
        match self {
            Scope::Chip(d) => d.calculated_returning_condition.as_ref(),
            _ => None,
        }
    }

    pub fn get_assertion_condition(&self) -> Option<&Value> {
        match self {
            Scope::Chip(d) => d.assertion_condition.as_ref(),
            _ => None,
        }
    }

    // ── Return management ─────────────────────────────────────────────

    pub fn is_return_guaranteed(&self) -> bool {
        match self {
            Scope::Chip(d) => d.return_guaranteed,
            Scope::Conditional(d) => d.return_guaranteed,
            Scope::Loop(d) => d.return_guaranteed,
            _ => false,
        }
    }

    pub fn set_return_guarantee(&mut self) {
        match self {
            Scope::Chip(d) => d.return_guaranteed = true,
            Scope::Conditional(d) => d.return_guaranteed = true,
            Scope::Loop(d) => d.return_guaranteed = true,
            _ => {}
        }
    }

    pub fn is_terminated_guaranteed(&self) -> bool {
        match self {
            Scope::Conditional(d) => d.loop_terminated_guaranteed,
            Scope::Loop(d) => d.loop_terminated_guaranteed,
            _ => false,
        }
    }

    pub fn set_terminated_guarantee(&mut self) {
        match self {
            Scope::Conditional(d) => d.loop_terminated_guaranteed = true,
            Scope::Loop(d) => d.loop_terminated_guaranteed = true,
            _ => {}
        }
    }

    pub fn get_returns_with_conditions(&self) -> &[(Value, Value)] {
        match self {
            Scope::Chip(d) => &d.returns_with_conditions,
            _ => &[],
        }
    }

    pub fn get_return_dtype(&self) -> Option<&ZinniaType> {
        match self {
            Scope::Chip(d) => Some(&d.return_dtype),
            _ => None,
        }
    }

    /// Register a return value with the given condition. Only valid on ChipScope.
    pub fn register_return(&mut self, value: Value, condition: Value, builder: &mut IRBuilder) {
        if let Scope::Chip(d) = self {
            d.returns_with_conditions.push((value, condition.clone()));
            let not_cond = builder.ir_logical_not(&condition);
            d.calculated_returning_condition = Some(match &d.calculated_returning_condition {
                None => not_cond,
                Some(existing) => builder.ir_logical_and(existing, &not_cond),
            });
        }
    }

    // ── Loop control ──────────────────────────────────────────────────

    pub fn loop_continue(&mut self, condition: Value, builder: &mut IRBuilder) {
        if let Scope::Loop(d) = self {
            let not_cond = builder.ir_logical_not(&condition);
            d.continue_condition = Some(match &d.continue_condition {
                None => not_cond.clone(),
                Some(existing) => builder.ir_logical_and(existing, &not_cond),
            });
            // Recompute looping condition
            d.calculated_looping_condition = Some(match &d.break_condition {
                None => d.continue_condition.clone().unwrap(),
                Some(brk) => builder.ir_logical_and(d.continue_condition.as_ref().unwrap(), brk),
            });
            if let Some(ref sup) = d.super_looping_condition {
                let current = d.calculated_looping_condition.clone().unwrap();
                d.calculated_looping_condition = Some(builder.ir_logical_and(&current, sup));
            }
        }
    }

    pub fn loop_break(&mut self, condition: Value, builder: &mut IRBuilder) {
        if let Scope::Loop(d) = self {
            let not_cond = builder.ir_logical_not(&condition);
            d.break_condition = Some(match &d.break_condition {
                None => not_cond.clone(),
                Some(existing) => builder.ir_logical_and(existing, &not_cond),
            });
            // Recompute looping condition
            d.calculated_looping_condition = Some(match &d.continue_condition {
                None => d.break_condition.clone().unwrap(),
                Some(cont) => builder.ir_logical_and(cont, d.break_condition.as_ref().unwrap()),
            });
            if let Some(ref sup) = d.super_looping_condition {
                let current = d.calculated_looping_condition.clone().unwrap();
                d.calculated_looping_condition = Some(builder.ir_logical_and(&current, sup));
            }
        }
    }

    pub fn loop_reiterate(&mut self, builder: &mut IRBuilder) {
        if let Scope::Loop(d) = self {
            d.continue_condition = None;
            d.calculated_looping_condition = d.break_condition.clone();
            if let Some(ref sup) = d.super_looping_condition {
                d.calculated_looping_condition = Some(match &d.calculated_looping_condition {
                    None => sup.clone(),
                    Some(current) => builder.ir_logical_and(current, sup),
                });
            }
        }
    }

    /// Whether variable types from parent should be locked (cannot change type).
    /// Python: `lock_parent_variable_types()`
    pub fn lock_parent_variable_types(&self) -> bool {
        match self {
            Scope::Conditional(d) => {
                // Lock if condition is not statically known
                d.condition.bool_val().is_none() && d.condition.int_val().is_none()
            }
            Scope::Loop(d) => {
                let brk_unknown = d.break_condition.as_ref().is_some_and(|v| {
                    v.bool_val().is_none() || v.bool_val() == Some(false)
                });
                let cont_unknown = d.continue_condition.as_ref().is_some_and(|v| {
                    v.bool_val().is_none() || v.bool_val() == Some(false)
                });
                brk_unknown || cont_unknown
            }
            _ => false,
        }
    }

    /// Get the local variable table (for conditional scope merging).
    pub fn get_local_var_table(&self) -> Option<&HashMap<String, ValueStore>> {
        match self {
            Scope::Conditional(d) => Some(&d.var_table),
            _ => None,
        }
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
    fn test_master_scope_basics() {
        let mut scope = Scope::Master(MasterScopeData {
            var_table: HashMap::new(),
        });

        let val = Value::Integer(ScalarValue::known(42, 0));
        scope.set_local("x".to_string(), ValueStore::new(val.clone()));

        assert!(scope.exists_local("x"));
        assert!(!scope.exists_local("y"));
        assert_eq!(scope.get_local("x").unwrap().value.int_val(), Some(42));
        assert!(scope.is_boundary());
        assert!(!scope.delegates_to_parent());
    }

    #[test]
    fn test_chip_scope_return() {
        let mut builder = IRBuilder::new();
        let cond = builder.ir_constant_bool(true);

        let mut scope = Scope::Chip(ChipScopeData {
            var_table: HashMap::new(),
            return_guaranteed: false,
            return_dtype: ZinniaType::Integer,
            returns_with_conditions: Vec::new(),
            calculated_returning_condition: None,
            assertion_condition: None,
        });

        let ret_val = Value::Integer(ScalarValue::known(10, 0));
        scope.register_return(ret_val, cond, &mut builder);

        assert_eq!(scope.get_returns_with_conditions().len(), 1);
        assert!(scope.get_returning_condition().is_some());
    }

    #[test]
    fn test_scope_type_queries() {
        let master = Scope::Master(MasterScopeData {
            var_table: HashMap::new(),
        });
        assert!(!master.is_in_chip());
        assert!(!master.is_in_loop());

        let chip = Scope::Chip(ChipScopeData {
            var_table: HashMap::new(),
            return_guaranteed: false,
            return_dtype: ZinniaType::None,
            returns_with_conditions: Vec::new(),
            calculated_returning_condition: None,
            assertion_condition: None,
        });
        assert!(chip.is_in_chip());
        assert!(!chip.is_in_loop());

        let loop_scope = Scope::Loop(Box::new(LoopScopeData {
            var_table: HashMap::new(),
            continue_condition: None,
            break_condition: None,
            return_guaranteed: false,
            loop_terminated_guaranteed: false,
            calculated_looping_condition: None,
            super_looping_condition: None,
        }));
        assert!(!loop_scope.is_in_chip());
        assert!(loop_scope.is_in_loop());
    }
}
