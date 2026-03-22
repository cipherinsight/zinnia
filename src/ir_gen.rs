//! IR Generator — visitor pattern over Zinnia AST, produces IRGraph.
//! Ports `zinnia/compile/ir/ir_gen.py` (854 lines).

use std::collections::HashMap;

use crate::ast::*;
use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_ctx::IRContext;
use crate::types::{CompositeData, Value, ZinniaType, DTDescriptorDict};

/// Check if an assignment target node has the `star` flag set.
fn is_starred_target(node: &ASTNode) -> bool {
    match node {
        ASTNode::ASTNameAssignTarget(t) => t.star,
        ASTNode::ASTSubscriptAssignTarget(t) => t.star,
        ASTNode::ASTStarredExpr(_) => true,
        _ => false,
    }
}

/// Represents a slice index: either a single value or a range (start, stop, step).
enum SliceIndex {
    Single(Value),
    Range(Option<Value>, Option<Value>, Option<Value>),
}

/// Configuration for the IR generator.
#[derive(Debug, Clone)]
pub struct IRGenConfig {
    pub loop_limit: u32,
    pub recursion_limit: u32,
}

impl Default for IRGenConfig {
    fn default() -> Self {
        Self {
            loop_limit: 256,
            recursion_limit: 16,
        }
    }
}

/// Registered chip information for the IR generator.
#[derive(Debug, Clone)]
pub struct RegisteredChip {
    pub chip_ast: serde_json::Value,  // The ASTChip dict
    pub return_dt: serde_json::Value, // Return type descriptor
}

/// Registered external function information for the IR generator.
#[derive(Debug, Clone)]
pub struct RegisteredExternal {
    pub return_dt: serde_json::Value, // Return type descriptor
}

/// The IR Generator — walks the AST and produces an IRGraph.
pub struct IRGenerator {
    pub builder: IRBuilder,
    pub ctx: IRContext,
    config: IRGenConfig,
    registered_chips: HashMap<String, RegisteredChip>,
    registered_externals: HashMap<String, RegisteredExternal>,
    recursion_depth: u32,
    /// Next available memory segment ID for dynamic array allocations.
    next_segment_id: u32,
    /// Next available array ID for dynamic ndarray metadata.
    next_array_id: u32,
}

impl IRGenerator {
    pub fn new(config: IRGenConfig) -> Self {
        Self {
            builder: IRBuilder::new(),
            ctx: IRContext::new(),
            config,
            registered_chips: HashMap::new(),
            registered_externals: HashMap::new(),
            next_segment_id: 0,
            next_array_id: 0,
            recursion_depth: 0,
        }
    }

    /// Main entry point: generate an IRGraph from an AST circuit.
    pub fn generate(mut self, ast: &ASTCircuit) -> IRGraph {
        self.visit_circuit(ast);
        self.builder.export_ir_graph()
    }

    /// Generate from a JSON string (the bridge entry point).
    pub fn generate_from_json(config: IRGenConfig, ast_json: &str) -> Result<IRGraph, String> {
        let node: ASTNode = serde_json::from_str(ast_json)
            .map_err(|e| format!("AST parse error: {}", e))?;
        match node {
            ASTNode::ASTCircuit(circuit) => {
                let gen = IRGenerator::new(config);
                Ok(gen.generate(&circuit))
            }
            _ => Err("Expected ASTCircuit at top level".to_string()),
        }
    }

    /// Generate from JSON with chips and externals.
    pub fn generate_from_json_with_chips(
        config: IRGenConfig,
        ast_json: &str,
        chips: &HashMap<String, serde_json::Value>,
        externals: &HashMap<String, serde_json::Value>,
    ) -> Result<IRGraph, String> {
        let node: ASTNode = serde_json::from_str(ast_json)
            .map_err(|e| format!("AST parse error: {}", e))?;
        match node {
            ASTNode::ASTCircuit(circuit) => {
                let mut gen = IRGenerator::new(config);
                // Register chips
                for (name, chip_data) in chips {
                    gen.registered_chips.insert(name.clone(), RegisteredChip {
                        chip_ast: chip_data["chip_ast"].clone(),
                        return_dt: chip_data["return_dt"].clone(),
                    });
                }
                // Register externals
                for (name, ext_data) in externals {
                    gen.registered_externals.insert(name.clone(), RegisteredExternal {
                        return_dt: ext_data["return_dt"].clone(),
                    });
                }
                Ok(gen.generate(&circuit))
            }
            _ => Err("Expected ASTCircuit at top level".to_string()),
        }
    }

    // ── Visitor dispatch ──────────────────────────────────────────────

    fn visit(&mut self, node: &ASTNode) -> Value {
        match node {
            ASTNode::ASTCircuit(n) => { self.visit_circuit(n); Value::None }
            ASTNode::ASTAssignStatement(n) => { self.visit_assign(n); Value::None }
            ASTNode::ASTAugAssignStatement(n) => { self.visit_aug_assign(n); Value::None }
            ASTNode::ASTCondStatement(n) => { self.visit_cond(n); Value::None }
            ASTNode::ASTForInStatement(n) => { self.visit_for_in(n); Value::None }
            ASTNode::ASTWhileStatement(n) => { self.visit_while(n); Value::None }
            ASTNode::ASTBreakStatement(_) => { self.visit_break(); Value::None }
            ASTNode::ASTContinueStatement(_) => { self.visit_continue(); Value::None }
            ASTNode::ASTReturnStatement(n) => { self.visit_return(n); Value::None }
            ASTNode::ASTAssertStatement(n) => { self.visit_assert(n); Value::None }
            ASTNode::ASTPassStatement(_) => Value::None,
            ASTNode::ASTExprStatement(n) => self.visit(&n.expr),
            ASTNode::ASTBinaryOperator(n) => self.visit_binary_op(n),
            ASTNode::ASTUnaryOperator(n) => self.visit_unary_op(n),
            ASTNode::ASTNamedAttribute(n) => self.visit_named_attr(n),
            ASTNode::ASTExprAttribute(n) => self.visit_expr_attr(n),
            ASTNode::ASTLoad(n) => self.visit_load(n),
            ASTNode::ASTSubscriptExp(n) => self.visit_subscript(n),
            ASTNode::ASTConstantFloat(n) => self.builder.ir_constant_float(n.value),
            ASTNode::ASTConstantInteger(n) => self.builder.ir_constant_int(n.value),
            ASTNode::ASTConstantBoolean(n) => self.builder.ir_constant_bool(n.value),
            ASTNode::ASTConstantNone(_) => Value::None,
            ASTNode::ASTConstantString(n) => self.builder.ir_constant_str(n.value.clone()),
            ASTNode::ASTSquareBrackets(n) => self.visit_square_brackets(n),
            ASTNode::ASTParenthesis(n) => self.visit_parenthesis(n),
            ASTNode::ASTGeneratorExp(n) => self.visit_generator_exp(n),
            ASTNode::ASTCondExp(n) => self.visit_cond_exp(n),
            ASTNode::ASTJoinedStr(n) => self.visit_joined_str(n),
            ASTNode::ASTFormattedValue(n) => self.visit_formatted_value(n),
            ASTNode::ASTStarredExpr(_) => panic!("Can't use starred expression here"),
            // Assignment targets are not visited as expressions
            ASTNode::ASTNameAssignTarget(_)
            | ASTNode::ASTSubscriptAssignTarget(_)
            | ASTNode::ASTTupleAssignTarget(_)
            | ASTNode::ASTListAssignTarget(_)
            | ASTNode::ASTChip(_) => {
                panic!("Cannot visit {:?} as expression", std::mem::discriminant(node))
            }
        }
    }

    // ── Circuit ───────────────────────────────────────────────────────

    fn visit_circuit(&mut self, n: &ASTCircuit) {
        // Process inputs
        for (i, inp) in n.inputs.iter().enumerate() {
            let dt = self.parse_dt_descriptor(&inp.annotation.dt);
            let kind = inp.annotation.kind.as_deref().unwrap_or("Private");
            let is_public = kind == "Public";
            let val = self.read_input_value(&dt, vec![0, i as u32], is_public);
            self.ctx.set(&inp.name, val);
        }

        self.register_global_datatypes();

        for stmt in &n.block {
            self.visit(stmt);
        }
    }

    // ── Statements ────────────────────────────────────────────────────

    fn visit_assign(&mut self, n: &ASTAssignStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let val = self.visit(&n.value);
        for target in &n.targets {
            self.do_recursive_assign(target, val.clone(), true);
        }
    }

    fn visit_aug_assign(&mut self, n: &ASTAugAssignStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        // AugAssign on subscript — delegate to builder
        let _val = self.visit(&n.value);
        // TODO: full aug assign implementation
    }

    fn visit_cond(&mut self, n: &ASTCondStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let cond_val = self.visit(&n.cond);
        // Check for multi-element array used as condition
        if let Value::List(data) | Value::Tuple(data) = &cond_val {
            if data.values.len() > 1 {
                // Check if elements are composites (multi-dim array)
                if data.values.iter().any(|v| matches!(v, Value::List(_) | Value::Tuple(_))) {
                    panic!("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                }
            }
        }
        let true_cond = self.to_scalar_bool(&cond_val);
        let false_cond = self.builder.ir_logical_not(&true_cond);

        let true_static = true_cond.bool_val().or_else(|| true_cond.int_val().map(|v| v != 0));
        let false_static = false_cond.bool_val().or_else(|| false_cond.int_val().map(|v| v != 0));

        // True branch
        let mut scope_true_ret_guaranteed = false;
        let mut scope_true_term_guaranteed = false;
        if true_static.is_none() || true_static == Some(true) {
            self.ctx.if_enter(true_cond.clone(), &mut self.builder);
            for stmt in &n.t_block {
                self.visit(stmt);
            }
            scope_true_ret_guaranteed = self.ctx.check_return_guaranteed();
            scope_true_term_guaranteed = self.ctx.check_loop_terminated_guaranteed();
            let _scope_true = self.ctx.if_leave();
        }

        // False branch
        let mut scope_false_ret_guaranteed = false;
        let mut scope_false_term_guaranteed = false;
        if false_static.is_none() || false_static == Some(true) {
            self.ctx.if_enter(false_cond.clone(), &mut self.builder);
            for stmt in &n.f_block {
                self.visit(stmt);
            }
            scope_false_ret_guaranteed = self.ctx.check_return_guaranteed();
            scope_false_term_guaranteed = self.ctx.check_loop_terminated_guaranteed();
            let _scope_false = self.ctx.if_leave();
        }

        // Update return guarantee
        if (true_static == Some(true) && scope_true_ret_guaranteed)
            || (false_static == Some(true) && scope_false_ret_guaranteed)
            || (scope_true_ret_guaranteed && scope_false_ret_guaranteed)
        {
            self.ctx.set_return_guarantee();
        }

        // Update terminated guarantee
        if self.ctx.is_in_loop()
            && ((true_static == Some(true)
                && (scope_true_term_guaranteed || scope_true_ret_guaranteed))
                || (false_static == Some(true)
                    && (scope_false_term_guaranteed || scope_false_ret_guaranteed))
                || ((scope_true_term_guaranteed || scope_true_ret_guaranteed)
                    && (scope_false_term_guaranteed || scope_false_ret_guaranteed)))
            {
                self.ctx.set_terminated_guarantee();
            }
    }

    fn visit_for_in(&mut self, n: &ASTForInStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }

        let iter_val = self.visit(&n.iter_expr);

        // Extract the iterable elements
        let elements: Vec<Value> = match &iter_val {
            Value::List(data) | Value::Tuple(data) => data.values.clone(),
            _ => {
                panic!("'{}' object is not iterable", iter_val.zinnia_type());
            }
        };

        // Unroll the loop: for each element, bind the target variable and run the block
        self.ctx.loop_enter();
        for elem in &elements {
            self.ctx.loop_reiter(&mut self.builder);

            // Bind the loop variable
            self.do_recursive_assign(&n.target, elem.clone(), false);

            let mut terminated = false;
            for stmt in &n.block {
                self.visit(stmt);
                if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
                    terminated = true;
                    break;
                }
            }
            if terminated {
                break;
            }
        }

        // Handle orelse block
        for stmt in &n.orelse {
            self.visit(stmt);
        }

        self.ctx.loop_leave();
    }

    fn visit_while(&mut self, n: &ASTWhileStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }

        let mut loop_quota = self.config.loop_limit + 1;
        self.ctx.loop_enter();

        loop {
            self.ctx.loop_reiter(&mut self.builder);
            let test = self.visit(&n.test_expr);
            let test_bool = self.to_scalar_bool(&test);
            loop_quota -= 1;

            if loop_quota == 0 {
                // Loop limit reached
                break;
            }

            let test_static = test_bool.bool_val().or_else(|| test_bool.int_val().map(|v| v != 0));
            if test_static == Some(false) {
                break;
            }

            if test_static.is_none() {
                let not_test = self.builder.ir_logical_not(&test_bool);
                self.ctx.loop_break(Some(not_test), &mut self.builder);
            }

            let mut terminated = false;
            for stmt in &n.block {
                self.visit(stmt);
                if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed()
                {
                    terminated = true;
                    break;
                }
            }
            if terminated {
                break;
            }
        }

        self.ctx.loop_leave();
    }

    fn visit_break(&mut self) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        self.ctx.loop_break(None, &mut self.builder);
        self.ctx.set_terminated_guarantee();
    }

    fn visit_continue(&mut self) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        self.ctx.loop_continue(&mut self.builder);
    }

    fn visit_return(&mut self, n: &ASTReturnStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let val = match &n.expr {
            Some(expr) => self.visit(expr),
            None => Value::None,
        };
        self.ctx.register_return(val, &mut self.builder);
        self.ctx.set_return_guarantee();
    }

    fn visit_assert(&mut self, n: &ASTAssertStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let test = self.visit(&n.expr);
        self.assert_value(&test);
    }

    /// Conditional select: if cond { tv } else { fv }, with element-wise support.
    fn select_value(&mut self, cond: &Value, tv: &Value, fv: &Value) -> Value {
        match (tv, fv) {
            (Value::List(td), Value::List(fd)) if td.values.len() == fd.values.len() => {
                let results: Vec<Value> = td.values.iter().zip(fd.values.iter())
                    .map(|(t, f)| self.select_value(cond, t, f))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: results })
            }
            (Value::Tuple(td), Value::Tuple(fd)) if td.values.len() == fd.values.len() => {
                let results: Vec<Value> = td.values.iter().zip(fd.values.iter())
                    .map(|(t, f)| self.select_value(cond, t, f))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::Tuple(CompositeData { elements_type: types, values: results })
            }
            // If types don't match (e.g., list vs scalar), just use the true value
            // (can't do conditional select across different structures)
            (Value::List(_) | Value::Tuple(_), _) | (_, Value::List(_) | Value::Tuple(_)) => {
                tv.clone()
            }
            _ => self.builder.ir_select_i(cond, tv, fv),
        }
    }

    /// Convert a value to a scalar boolean, reducing composites via AND.
    fn to_scalar_bool(&mut self, val: &Value) -> Value {
        match val {
            Value::List(data) | Value::Tuple(data) => {
                if data.values.is_empty() {
                    return self.builder.ir_constant_bool(true);
                }
                let mut acc = self.to_scalar_bool(&data.values[0]);
                for elem in &data.values[1..] {
                    let b = self.to_scalar_bool(elem);
                    acc = self.builder.ir_logical_and(&acc, &b);
                }
                acc
            }
            _ => self.builder.ir_bool_cast(val),
        }
    }

    /// Assert a value. For composites, reduces to scalar bool via AND, then asserts.
    /// The assert is conditioned on the current path condition — if the path condition
    /// is false (we're in an inactive branch), the assert is automatically satisfied.
    fn assert_value(&mut self, val: &Value) {
        let scalar = self.to_scalar_bool(val);
        // Get path condition: if we're inside a conditional/loop, the assert
        // should only fire when the path condition is true
        let cond = self.ctx.get_condition_value(&mut self.builder);
        // assert(cond → scalar) = assert(!cond || scalar) = assert(select(cond, scalar, true))
        let true_val = self.builder.ir_constant_bool(true);
        let conditioned = self.builder.ir_select_i(&cond, &scalar, &true_val);
        self.builder.ir_assert(&conditioned);
    }

    // ── Expressions ───────────────────────────────────────────────────

    fn visit_binary_op(&mut self, n: &ASTBinaryOperator) -> Value {
        let lhs = self.visit(&n.lhs);
        let rhs = self.visit(&n.rhs);
        self.apply_binary_op(n.operator.as_str(), &lhs, &rhs)
    }

    /// Apply a binary operation, with element-wise support for composite types.
    fn apply_binary_op(&mut self, op: &str, lhs: &Value, rhs: &Value) -> Value {
        // List/tuple concatenation via `+` (only for different-length composites
        // or when both are pure integer lists — same-length composites do element-wise)
        if op == "add" {
            match (lhs, rhs) {
                (Value::List(ld), Value::List(rd)) | (Value::Tuple(ld), Value::List(rd))
                | (Value::List(ld), Value::Tuple(rd)) | (Value::Tuple(ld), Value::Tuple(rd)) => {
                    // Same-length composites: element-wise addition (ndarray behavior)
                    if ld.values.len() == rd.values.len() && !ld.values.is_empty() {
                        let results: Vec<Value> = ld.values.iter().zip(rd.values.iter())
                            .map(|(l, r)| self.apply_binary_op("add", l, r))
                            .collect();
                        let types = results.iter().map(|v| v.zinnia_type()).collect();
                        return Value::List(CompositeData { elements_type: types, values: results });
                    }
                    // Different-length composites: concatenation (Python list behavior)
                    let mut values = ld.values.clone();
                    values.extend(rd.values.clone());
                    let types = values.iter().map(|v| v.zinnia_type()).collect();
                    let is_tuple = matches!(lhs, Value::Tuple(_));
                    return if is_tuple {
                        Value::Tuple(CompositeData { elements_type: types, values })
                    } else {
                        Value::List(CompositeData { elements_type: types, values })
                    };
                }
                _ => {}
            }
        }
        // List/tuple repetition via `*`
        if op == "mul" {
            match (lhs, rhs) {
                (Value::List(ld), _) | (Value::Tuple(ld), _) if rhs.int_val().is_some() => {
                    let n = rhs.int_val().unwrap().max(0) as usize;
                    let mut values = Vec::new();
                    for _ in 0..n {
                        values.extend(ld.values.clone());
                    }
                    let types = values.iter().map(|v| v.zinnia_type()).collect();
                    let is_tuple = matches!(lhs, Value::Tuple(_));
                    return if is_tuple {
                        Value::Tuple(CompositeData { elements_type: types, values })
                    } else {
                        Value::List(CompositeData { elements_type: types, values })
                    };
                }
                (_, Value::List(rd)) | (_, Value::Tuple(rd)) if lhs.int_val().is_some() => {
                    let n = lhs.int_val().unwrap().max(0) as usize;
                    let mut values = Vec::new();
                    for _ in 0..n {
                        values.extend(rd.values.clone());
                    }
                    let types = values.iter().map(|v| v.zinnia_type()).collect();
                    let is_tuple = matches!(rhs, Value::Tuple(_));
                    return if is_tuple {
                        Value::Tuple(CompositeData { elements_type: types, values })
                    } else {
                        Value::List(CompositeData { elements_type: types, values })
                    };
                }
                _ => {}
            }
        }
        // Composite comparison: handle eq/ne/lt/lte/gt/gte for composites
        if matches!(op, "eq" | "ne" | "lt" | "lte" | "gt" | "gte") {
            match (lhs, rhs) {
                (Value::List(ld), Value::List(rd)) | (Value::Tuple(ld), Value::List(rd))
                | (Value::List(ld), Value::Tuple(rd)) | (Value::Tuple(ld), Value::Tuple(rd)) => {
                    return self.composite_comparison(op, ld, rd);
                }
                _ => {}
            }
        }
        // Element-wise: both composite with matching length (for arithmetic ops)
        match (lhs, rhs) {
            (Value::List(ld), Value::List(rd)) | (Value::Tuple(ld), Value::List(rd))
            | (Value::List(ld), Value::Tuple(rd)) | (Value::Tuple(ld), Value::Tuple(rd))
                if ld.values.len() == rd.values.len() =>
            {
                let results: Vec<Value> = ld.values.iter().zip(rd.values.iter())
                    .map(|(l, r)| self.apply_binary_op(op, l, r))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                return Value::List(CompositeData { elements_type: types, values: results });
            }
            // Broadcasting: scalar op composite
            (_, Value::List(rd)) | (_, Value::Tuple(rd)) if lhs.is_number() => {
                let results: Vec<Value> = rd.values.iter()
                    .map(|r| self.apply_binary_op(op, lhs, r))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                return Value::List(CompositeData { elements_type: types, values: results });
            }
            (Value::List(ld), _) | (Value::Tuple(ld), _) if rhs.is_number() => {
                let results: Vec<Value> = ld.values.iter()
                    .map(|l| self.apply_binary_op(op, l, rhs))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                return Value::List(CompositeData { elements_type: types, values: results });
            }
            _ => {}
        }
        // Class (type) comparison
        if matches!((lhs, rhs), (Value::Class(_), Value::Class(_))) {
            if let (Value::Class(lt), Value::Class(rt)) = (lhs, rhs) {
                let types_equal = lt == rt;
                return match op {
                    "eq" => self.builder.ir_constant_bool(types_equal),
                    "ne" => self.builder.ir_constant_bool(!types_equal),
                    _ => self.builder.ir_constant_bool(false),
                };
            }
        }
        // Scalar operation
        self.apply_scalar_binary_op(op, lhs, rhs)
    }

    fn apply_scalar_binary_op(&mut self, op: &str, lhs: &Value, rhs: &Value) -> Value {
        let use_float = matches!(lhs, Value::Float(_)) || matches!(rhs, Value::Float(_));
        if use_float {
            // Float operations
            match op {
                "add" => self.builder.ir_add_f(lhs, rhs),
                "sub" => self.builder.ir_sub_f(lhs, rhs),
                "mul" => self.builder.ir_mul_f(lhs, rhs),
                "div" => self.builder.ir_div_f(lhs, rhs),
                "pow" => self.builder.ir_pow_f(lhs, rhs),
                "mod" | "floor_div" => {
                    // Fallback to integer ops for these
                    self.builder.ir_mod_i(lhs, rhs)
                }
                "eq" => self.builder.ir_equal_f(lhs, rhs),
                "ne" => {
                    let eq = self.builder.ir_equal_f(lhs, rhs);
                    self.builder.ir_logical_not(&eq)
                }
                "lt" => self.builder.ir_less_than_f(lhs, rhs),
                "lte" => self.builder.ir_less_than_or_equal_f(lhs, rhs),
                "gt" => self.builder.ir_greater_than_f(lhs, rhs),
                "gte" => self.builder.ir_greater_than_or_equal_f(lhs, rhs),
                "and" => self.builder.ir_logical_and(lhs, rhs),
                "or" => self.builder.ir_logical_or(lhs, rhs),
                "mat_mul" => self.matmul(lhs, rhs),
                _ => panic!("Unknown binary operator: {}", op),
            }
        } else {
            // Integer operations
            match op {
                "add" => self.builder.ir_add_i(lhs, rhs),
                "sub" => self.builder.ir_sub_i(lhs, rhs),
                "mul" => self.builder.ir_mul_i(lhs, rhs),
                "div" => self.builder.ir_div_i(lhs, rhs),
                "mod" => self.builder.ir_mod_i(lhs, rhs),
                "floor_div" => self.builder.ir_floor_div_i(lhs, rhs),
                "pow" => self.builder.ir_pow_i(lhs, rhs),
                "eq" => self.builder.ir_equal_i(lhs, rhs),
                "ne" => self.builder.ir_not_equal_i(lhs, rhs),
                "lt" => self.builder.ir_less_than_i(lhs, rhs),
                "lte" => self.builder.ir_less_than_or_equal_i(lhs, rhs),
                "gt" => self.builder.ir_greater_than_i(lhs, rhs),
                "gte" => self.builder.ir_greater_than_or_equal_i(lhs, rhs),
                "and" => self.builder.ir_logical_and(lhs, rhs),
                "or" => self.builder.ir_logical_or(lhs, rhs),
                "mat_mul" => self.matmul(lhs, rhs),
                _ => panic!("Unknown binary operator: {}", op),
            }
        }
    }

    fn visit_unary_op(&mut self, n: &ASTUnaryOperator) -> Value {
        let operand = self.visit(&n.operand);
        self.apply_unary_op(n.operator.as_str(), &operand)
    }

    /// Apply a unary operation, with element-wise support for composite types.
    fn apply_unary_op(&mut self, op: &str, operand: &Value) -> Value {
        match operand {
            Value::List(data) | Value::Tuple(data) => {
                let results: Vec<Value> = data.values.iter()
                    .map(|v| self.apply_unary_op(op, v))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: results })
            }
            _ => match op {
                "not" => self.builder.ir_logical_not(operand),
                "usub" => {
                    let zero = self.builder.ir_constant_int(0);
                    self.builder.ir_sub_i(&zero, operand)
                }
                "uadd" => operand.clone(),
                _ => panic!("Unknown unary operator: {}", op),
            }
        }
    }

    /// Composite comparison (eq, ne, lt, lte, gt, gte) — element-wise then reduce.
    fn composite_comparison(&mut self, op: &str, ld: &CompositeData, rd: &CompositeData) -> Value {
        let min_len = ld.values.len().min(rd.values.len());
        match op {
            "eq" => {
                if ld.values.len() != rd.values.len() {
                    return self.builder.ir_constant_bool(false);
                }
                let mut result = self.builder.ir_constant_bool(true);
                for i in 0..min_len {
                    let cmp = self.apply_binary_op("eq", &ld.values[i], &rd.values[i]);
                    let cmp_bool = self.to_scalar_bool(&cmp);
                    result = self.builder.ir_logical_and(&result, &cmp_bool);
                }
                result
            }
            "ne" => {
                let eq = self.composite_comparison("eq", ld, rd);
                self.builder.ir_logical_not(&eq)
            }
            "lt" | "lte" | "gt" | "gte" => {
                // Lexicographic comparison
                // For simplicity, compare element by element
                if min_len == 0 {
                    return match op {
                        "lt" => self.builder.ir_constant_bool(ld.values.len() < rd.values.len()),
                        "lte" => self.builder.ir_constant_bool(ld.values.len() <= rd.values.len()),
                        "gt" => self.builder.ir_constant_bool(ld.values.len() > rd.values.len()),
                        "gte" => self.builder.ir_constant_bool(ld.values.len() >= rd.values.len()),
                        _ => unreachable!(),
                    };
                }
                // Compare first elements
                let mut result = self.apply_binary_op(op, &ld.values[0], &rd.values[0]);
                for i in 1..min_len {
                    // If previous elements were equal, compare this element
                    let prev_eq = self.apply_binary_op("eq", &ld.values[i-1], &rd.values[i-1]);
                    let prev_eq_bool = self.to_scalar_bool(&prev_eq);
                    let this_cmp = self.apply_binary_op(op, &ld.values[i], &rd.values[i]);
                    let this_cmp_bool = self.to_scalar_bool(&this_cmp);
                    // result = prev_eq ? this_cmp : result
                    result = self.builder.ir_select_i(&prev_eq_bool, &this_cmp_bool, &result);
                }
                // Handle different lengths: if all common elements are equal
                if ld.values.len() != rd.values.len() {
                    let all_eq = self.composite_comparison("eq",
                        &CompositeData { elements_type: ld.elements_type[..min_len].to_vec(), values: ld.values[..min_len].to_vec() },
                        &CompositeData { elements_type: rd.elements_type[..min_len].to_vec(), values: rd.values[..min_len].to_vec() },
                    );
                    let all_eq_bool = self.to_scalar_bool(&all_eq);
                    let len_result = match op {
                        "lt" | "lte" => self.builder.ir_constant_bool(ld.values.len() < rd.values.len()),
                        "gt" | "gte" => self.builder.ir_constant_bool(ld.values.len() > rd.values.len()),
                        _ => unreachable!(),
                    };
                    result = self.builder.ir_select_i(&all_eq_bool, &len_result, &result);
                }
                result
            }
            _ => self.builder.ir_constant_bool(false),
        }
    }

    fn visit_named_attr(&mut self, n: &ASTNamedAttribute) -> Value {
        // Handle starred args: foo(1, *args, 4) → flatten starred into arg list
        let mut visited_args: Vec<Value> = Vec::new();
        for a in &n.args {
            if let ASTNode::ASTStarredExpr(se) = a {
                let inner = self.visit(&se.value);
                if let Value::List(data) | Value::Tuple(data) = inner {
                    visited_args.extend(data.values);
                } else {
                    visited_args.push(inner);
                }
            } else {
                visited_args.push(self.visit(a));
            }
        }
        let _visited_kwargs: HashMap<String, Value> = n
            .kwargs
            .iter()
            .map(|(k, v)| (k.clone(), self.visit(v)))
            .collect();

        let target = n.target.as_deref();
        let member = n.member.as_str();

        match (target, member) {
            // ── Built-in functions (no target) ─────────────────────────
            (None, "range") => self.builtin_range(&visited_args),
            (None, "len") => self.builtin_len(&visited_args),
            (None, "int") => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_int(0)
                } else {
                    self.builder.ir_int_cast(&visited_args[0])
                }
            }
            (None, "float") => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_float(0.0)
                } else {
                    self.builder.ir_float_cast(&visited_args[0])
                }
            }
            (None, "bool") => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_bool(false)
                } else {
                    self.builder.ir_bool_cast(&visited_args[0])
                }
            }
            (None, "abs") => {
                if !visited_args.is_empty() {
                    match &visited_args[0] {
                        Value::List(data) | Value::Tuple(data) => {
                            let results: Vec<Value> = data.values.iter()
                                .map(|v| self.builder.ir_abs_i(v))
                                .collect();
                            let types = results.iter().map(|v| v.zinnia_type()).collect();
                            Value::List(CompositeData { elements_type: types, values: results })
                        }
                        v => self.builder.ir_abs_i(v),
                    }
                } else {
                    Value::None
                }
            }
            (None, "print") => {
                if !visited_args.is_empty() {
                    // For composites, print each element
                    let arg = &visited_args[0];
                    let flat = self.flatten_composite(arg);
                    if let Some(first) = flat.first() {
                        let fmt = self.builder.ir_constant_str(String::new());
                        self.builder.ir_print(first, &fmt)
                    } else {
                        Value::None
                    }
                } else {
                    Value::None
                }
            }
            (None, "enumerate") => {
                // enumerate(iterable) → list of (index, element) tuples
                if let Some(iter_val) = visited_args.first() {
                    self.builtin_enumerate(iter_val)
                } else {
                    Value::None
                }
            }
            (None, "sum") => {
                if let Some(iter_val) = visited_args.first() {
                    // Python sum() iterates over the first level
                    let mut result = if let Value::List(data) | Value::Tuple(data) = iter_val {
                        // Check if elements are composites (2D+ array)
                        if !data.values.is_empty() && matches!(&data.values[0], Value::List(_) | Value::Tuple(_)) {
                            // Sum over first axis (element-wise add of rows)
                            let mut acc = data.values[0].clone();
                            for row in &data.values[1..] {
                                acc = self.elementwise_binary("add", &acc, row);
                            }
                            acc
                        } else {
                            self.builtin_reduce("sum", iter_val)
                        }
                    } else {
                        self.builtin_reduce("sum", iter_val)
                    };
                    // sum(iterable, start) — add start value
                    if let Some(start) = visited_args.get(1) {
                        result = self.apply_binary_op("add", start, &result);
                    }
                    result
                } else {
                    Value::None
                }
            }
            (None, "any") | (None, "all") => {
                if let Some(iter_val) = visited_args.first() {
                    // For 2D+ arrays, Python's any()/all() tries to evaluate truth of rows
                    // which is ambiguous for multi-element arrays
                    if let Value::List(data) | Value::Tuple(data) = iter_val {
                        if !data.values.is_empty() && matches!(&data.values[0], Value::List(_) | Value::Tuple(_)) {
                            panic!("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                        }
                    }
                    self.builtin_reduce(member, iter_val)
                } else {
                    Value::None
                }
            }
            (None, "min") | (None, "max") => {
                if let Some(iter_val) = visited_args.first() {
                    // For 2D+ arrays, reduce over the first axis
                    if let Value::List(data) | Value::Tuple(data) = iter_val {
                        if !data.values.is_empty() && matches!(&data.values[0], Value::List(_) | Value::Tuple(_)) {
                            let mut acc = data.values[0].clone();
                            for row in &data.values[1..] {
                                acc = self.elementwise_minmax(&acc, row, member == "max");
                            }
                            return acc;
                        }
                    }
                    self.builtin_reduce(member, iter_val)
                } else {
                    Value::None
                }
            }
            (None, "pow") => {
                if visited_args.len() >= 3 {
                    // pow(base, exp, mod) — modular exponentiation
                    let base_exp = self.apply_scalar_binary_op("pow", &visited_args[0], &visited_args[1]);
                    self.builder.ir_mod_i(&base_exp, &visited_args[2])
                } else if visited_args.len() >= 2 {
                    self.apply_scalar_binary_op("pow", &visited_args[0], &visited_args[1])
                } else {
                    Value::None
                }
            }
            (None, "list") => {
                // list(iterable) — convert to list, or empty list
                if visited_args.is_empty() {
                    Value::List(CompositeData { elements_type: vec![], values: vec![] })
                } else {
                    match &visited_args[0] {
                        Value::List(_) => visited_args[0].clone(),
                        Value::Tuple(data) => {
                            Value::List(CompositeData {
                                elements_type: data.elements_type.clone(),
                                values: data.values.clone(),
                            })
                        }
                        _ => visited_args[0].clone(),
                    }
                }
            }
            (None, "tuple") => {
                // tuple(iterable) — convert to tuple, or empty tuple
                if visited_args.is_empty() {
                    Value::Tuple(CompositeData { elements_type: vec![], values: vec![] })
                } else {
                    match &visited_args[0] {
                        Value::Tuple(_) => visited_args[0].clone(),
                        Value::List(data) => {
                            Value::Tuple(CompositeData {
                                elements_type: data.elements_type.clone(),
                                values: data.values.clone(),
                            })
                        }
                        _ => visited_args[0].clone(),
                    }
                }
            }

            // ── np.* (numpy-like operations) ───────────────────────────
            (Some("np"), "asarray") => {
                // np.asarray(x) — pass through if already a composite
                if !visited_args.is_empty() {
                    let val = &visited_args[0];
                    // Validate that all sub-elements have consistent shapes
                    if let Value::List(data) | Value::Tuple(data) = val {
                        if data.values.len() > 1 {
                            let first_is_composite = matches!(&data.values[0], Value::List(_) | Value::Tuple(_));
                            for v in &data.values[1..] {
                                let is_composite = matches!(v, Value::List(_) | Value::Tuple(_));
                                if is_composite != first_is_composite {
                                    panic!("To convert to NDArray, all sub-lists should be of the same shape");
                                }
                            }
                        }
                    }
                    // Handle dtype kwarg for type casting
                    if let Some(dtype) = _visited_kwargs.get("dtype") {
                        let to_float = matches!(dtype, Value::Class(ZinniaType::Float));
                        self.cast_composite(val, to_float)
                    } else {
                        val.clone()
                    }
                } else {
                    Value::None
                }
            }
            (Some("np"), "zeros") => self.np_fill(&visited_args, &_visited_kwargs, 0),
            (Some("np"), "ones") => self.np_fill(&visited_args, &_visited_kwargs, 1),
            (Some("np"), "identity") => self.np_identity(&visited_args),
            (Some("np"), "arange") => self.np_arange(&visited_args),
            (Some("np"), "linspace") => self.np_linspace(&visited_args, &_visited_kwargs),
            (Some("np"), "allclose") => self.np_allclose(&visited_args, &_visited_kwargs),
            (Some("np"), "concatenate") => self.np_concatenate(&visited_args, &_visited_kwargs),
            (Some("np"), "stack") => self.np_stack(&visited_args, &_visited_kwargs),

            // ── DynamicNDArray class methods ────────────────────────────
            (Some("DynamicNDArray"), "zeros") | (Some("zinnia"), "zeros") => {
                self.dyn_zeros(&visited_args, &_visited_kwargs)
            }
            (Some("DynamicNDArray"), "ones") | (Some("zinnia"), "ones") => {
                self.dyn_ones(&visited_args, &_visited_kwargs)
            }
            (Some("DynamicNDArray"), "eye") | (Some("zinnia"), "eye") => {
                self.dyn_eye(&visited_args, &_visited_kwargs)
            }
            (Some("DynamicNDArray"), "concatenate") | (Some("zinnia"), "concatenate") => {
                self.dyn_concatenate(&visited_args, &_visited_kwargs)
            }
            (Some("DynamicNDArray"), "stack") | (Some("zinnia"), "stack") => {
                self.dyn_stack(&visited_args, &_visited_kwargs)
            }

            // ── List methods (target is a variable name) ───────────────
            (Some(var), "append") if self.ctx.exists(var) => {
                self.list_method_append(var, &visited_args)
            }
            (Some(var), "extend") if self.ctx.exists(var) => {
                self.list_method_extend(var, &visited_args)
            }
            (Some(var), "pop") if self.ctx.exists(var) => {
                self.list_method_pop(var, &visited_args)
            }
            (Some(var), "remove") if self.ctx.exists(var) => {
                self.list_method_remove(var, &visited_args)
            }
            (Some(var), "clear") if self.ctx.exists(var) => {
                self.list_method_clear(var)
            }
            (Some(var), "copy") if self.ctx.exists(var) => {
                self.ctx.get(var).unwrap_or(Value::None)
            }
            (Some(var), "reverse") if self.ctx.exists(var) => {
                self.list_method_reverse(var)
            }
            (Some(var), "count") if self.ctx.exists(var) => {
                self.list_method_count(var, &visited_args)
            }
            (Some(var), "index") if self.ctx.exists(var) => {
                self.list_method_index(var, &visited_args)
            }
            (Some(var), "insert") if self.ctx.exists(var) => {
                self.list_method_insert(var, &visited_args)
            }

            // ── DynamicNDArray method dispatch ────────────────────────
            (Some(var), method) if self.ctx.exists(var) && matches!(self.ctx.get(var), Some(Value::DynamicNDArray(_))) => {
                let val = self.ctx.get(var).unwrap();
                self.dispatch_dyn_ndarray_method(val, method, &visited_args, &_visited_kwargs)
            }

            // ── Method calls on expr attributes ────────────────────────
            (Some(var), "sum") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    self.reduce_with_axis("sum", &val, ax)
                } else {
                    self.builtin_reduce("sum", &val)
                }
            }
            (Some(var), "any") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    self.reduce_with_axis("any", &val, ax)
                } else {
                    self.builtin_reduce("any", &val)
                }
            }
            (Some(var), "all") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    self.reduce_with_axis("all", &val, ax)
                } else {
                    self.builtin_reduce("all", &val)
                }
            }
            (Some(var), "transpose") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // Check for axes keyword argument
                let args = if let Some(axes_val) = _visited_kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    visited_args.clone()
                };
                self.ndarray_transpose(&val, &args)
            }
            (Some(var), "T") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                self.ndarray_transpose(&val, &[])
            }
            (Some(var), "tolist") if self.ctx.exists(var) => {
                self.ctx.get(var).unwrap_or(Value::None)
            }
            (Some(var), "astype") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // Determine target type from the argument (int or float class)
                let target_float = if let Some(Value::Class(ZinniaType::Float)) = visited_args.first() {
                    true
                } else {
                    false
                };
                self.cast_composite(&val, target_float)
            }
            (Some(var), "argmax") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    self.ndarray_argmax_argmin_with_axis(&val, ax, true)
                } else {
                    self.ndarray_argmax_argmin(&val, &visited_args, true)
                }
            }
            (Some(var), "argmin") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    self.ndarray_argmax_argmin_with_axis(&val, ax, false)
                } else {
                    self.ndarray_argmax_argmin(&val, &visited_args, false)
                }
            }

            // ── NDArray property access ──────────────────────────────
            (Some(var), "shape") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let shape = self.get_composite_shape(&val);
                let shape_vals: Vec<Value> = shape.iter()
                    .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                    .collect();
                let types = vec![ZinniaType::Integer; shape_vals.len()];
                Value::Tuple(CompositeData { elements_type: types, values: shape_vals })
            }
            (Some(var), "dtype") if self.ctx.exists(var) => {
                // Infer dtype from element types
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let flat = self.flatten_composite(&val);
                let has_float = flat.iter().any(|v| matches!(v, Value::Float(_)));
                if has_float {
                    Value::Class(ZinniaType::Float)
                } else {
                    Value::Class(ZinniaType::Integer)
                }
            }
            (Some(var), "min") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    self.reduce_with_axis("min", &val, ax)
                } else {
                    self.builtin_reduce("min", &val)
                }
            }
            (Some(var), "max") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    self.reduce_with_axis("max", &val, ax)
                } else {
                    self.builtin_reduce("max", &val)
                }
            }
            (Some(var), "prod") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    self.reduce_with_axis("prod", &val, ax)
                } else {
                    self.builtin_reduce("prod", &val)
                }
            }

            // ── NDArray ndim, size, flatten, flat, reshape, moveaxis, repeat, filter ─
            (Some(var), "ndim") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let shape = self.get_composite_shape(&val);
                self.builder.ir_constant_int(shape.len() as i64)
            }
            (Some(var), "size") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let shape = self.get_composite_shape(&val);
                let total: usize = shape.iter().product();
                self.builder.ir_constant_int(total as i64)
            }
            (Some(var), "flatten") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let flat = self.flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            (Some(var), "flat") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let flat = self.flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            (Some(var), "reshape") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                self.ndarray_reshape(&val, &visited_args)
            }
            (Some(var), "moveaxis") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                self.ndarray_moveaxis(&val, &visited_args)
            }
            (Some(var), "repeat") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                self.ndarray_repeat(&val, &visited_args, &_visited_kwargs)
            }
            (Some(var), "filter") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                self.ndarray_filter(&val, &visited_args)
            }

            // ── Chip calls (no target or target not a known variable) ─
            (None, name) if self.registered_chips.contains_key(name) => {
                self.visit_chip_call(name, &visited_args, &_visited_kwargs)
            }

            // ── External function calls ─────────────────────────────────
            (None, name) if self.registered_externals.contains_key(name) => {
                self.visit_external_call(name, &visited_args)
            }

            // ── Fallback ───────────────────────────────────────────────
            _ => {
                panic!(
                    "Named attribute `{}.{}` not yet implemented in Rust IR generator",
                    target.unwrap_or(""),
                    member
                )
            }
        }
    }

    fn visit_expr_attr(&mut self, n: &ASTExprAttribute) -> Value {
        let target = self.visit(&n.target);
        let mut visited_args: Vec<Value> = Vec::new();
        for a in &n.args {
            if let ASTNode::ASTStarredExpr(se) = a {
                let inner = self.visit(&se.value);
                if let Value::List(data) | Value::Tuple(data) = inner {
                    visited_args.extend(data.values);
                } else {
                    visited_args.push(inner);
                }
            } else {
                visited_args.push(self.visit(a));
            }
        }
        let visited_kwargs: HashMap<String, Value> = n
            .kwargs
            .iter()
            .map(|(k, v)| (k.clone(), self.visit(v)))
            .collect();

        // DynamicNDArray dispatch — route to dedicated handler
        if matches!(target, Value::DynamicNDArray(_)) {
            return self.dispatch_dyn_ndarray_method(
                target, n.member.as_str(), &visited_args, &visited_kwargs,
            );
        }

        match n.member.as_str() {
            "sum" => self.builtin_reduce("sum", &target),
            "any" => self.builtin_reduce("any", &target),
            "all" => self.builtin_reduce("all", &target),
            "transpose" => {
                let args = if let Some(axes_val) = visited_kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    visited_args.clone()
                };
                self.ndarray_transpose(&target, &args)
            }
            "T" => self.ndarray_transpose(&target, &[]),
            "tolist" => target,
            "astype" => {
                let target_float = if let Some(Value::Class(ZinniaType::Float)) = visited_args.first() {
                    true
                } else {
                    false
                };
                self.cast_composite(&target, target_float)
            }
            "argmax" => self.ndarray_argmax_argmin(&target, &visited_args, true),
            "argmin" => self.ndarray_argmax_argmin(&target, &visited_args, false),
            "prod" => self.builtin_reduce("prod", &target),
            "min" => self.builtin_reduce("min", &target),
            "max" => self.builtin_reduce("max", &target),
            "ndim" => {
                let shape = self.get_composite_shape(&target);
                self.builder.ir_constant_int(shape.len() as i64)
            }
            "size" => {
                let shape = self.get_composite_shape(&target);
                let total: usize = shape.iter().product();
                self.builder.ir_constant_int(total as i64)
            }
            "flatten" | "flat" => {
                let flat = self.flatten_composite(&target);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            "reshape" => self.ndarray_reshape(&target, &visited_args),
            "moveaxis" => self.ndarray_moveaxis(&target, &visited_args),
            "repeat" => self.ndarray_repeat(&target, &visited_args, &visited_kwargs),
            "filter" => self.ndarray_filter(&target, &visited_args),
            "shape" => {
                let shape = self.get_composite_shape(&target);
                let shape_vals: Vec<Value> = shape.iter()
                    .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                    .collect();
                let types = vec![ZinniaType::Integer; shape_vals.len()];
                Value::Tuple(CompositeData { elements_type: types, values: shape_vals })
            }
            "dtype" => {
                let flat = self.flatten_composite(&target);
                let has_float = flat.iter().any(|v| matches!(v, Value::Float(_)));
                if has_float { Value::Class(ZinniaType::Float) } else { Value::Class(ZinniaType::Integer) }
            }
            "append" | "extend" | "pop" | "remove" | "clear" |
            "copy" | "reverse" | "count" | "index" => {
                // These should be handled via visit_named_attr with a target variable name
                panic!("Expr attribute `.{}` on non-variable target not supported", n.member)
            }
            _ => {
                panic!(
                    "Expr attribute `.{}` not yet implemented in Rust IR generator",
                    n.member
                )
            }
        }
    }

    fn visit_load(&mut self, n: &ASTLoad) -> Value {
        self.ctx
            .get(&n.name)
            .unwrap_or_else(|| panic!("Variable `{}` not found", n.name))
    }

    fn visit_subscript(&mut self, n: &ASTSubscriptExp) -> Value {
        let val = self.visit(&n.val);
        // Evaluate slice indices by visiting them as AST nodes
        let slice_values = self.eval_slice_indices(&n.slicing);

        match &val {
            Value::List(data) | Value::Tuple(data) => {
                if slice_values.len() == 1 {
                    let idx_val = &slice_values[0];
                    match idx_val {
                        SliceIndex::Single(idx_value) => {
                            // Try to resolve statically
                            if let Some(idx) = idx_value.int_val() {
                                let idx = if idx < 0 { (data.values.len() as i64 + idx) as usize } else { idx as usize };
                                if idx < data.values.len() {
                                    data.values[idx].clone()
                                } else {
                                    Value::None
                                }
                            } else {
                                // Dynamic index: generate a chain of SelectI to pick the right element
                                self.dynamic_list_subscript(data, idx_value)
                            }
                        }
                        SliceIndex::Range(_, _, _) => {
                            // Range slicing on list/tuple
                            self.list_slice_range(&val, data, idx_val)
                        }
                    }
                } else {
                    // Multi-dimensional ndarray-style indexing
                    self.multidim_subscript(data, &slice_values)
                }
            }
            _ => Value::None,
        }
    }

    /// Evaluate slice indices by visiting AST nodes (not just extracting constants).
    fn eval_slice_indices(&mut self, slice: &ASTSlice) -> Vec<SliceIndex> {
        let mut indices = Vec::new();
        for d in &slice.data {
            if d.is_array() {
                // Range slice [start, stop, step]
                let arr = d.as_array().unwrap();
                let start = if arr.len() > 0 && !arr[0].is_null() {
                    if let Ok(node) = serde_json::from_value::<ASTNode>(arr[0].clone()) {
                        Some(self.visit(&node))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let stop = if arr.len() > 1 && !arr[1].is_null() {
                    if let Ok(node) = serde_json::from_value::<ASTNode>(arr[1].clone()) {
                        Some(self.visit(&node))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let step = if arr.len() > 2 && !arr[2].is_null() {
                    if let Ok(node) = serde_json::from_value::<ASTNode>(arr[2].clone()) {
                        Some(self.visit(&node))
                    } else {
                        None
                    }
                } else {
                    None
                };
                indices.push(SliceIndex::Range(start, stop, step));
            } else if let Ok(node) = serde_json::from_value::<ASTNode>(d.clone()) {
                let visited = self.visit(&node);
                indices.push(SliceIndex::Single(visited));
            } else {
                indices.push(SliceIndex::Single(Value::None));
            }
        }
        indices
    }

    /// Generate a chain of SelectI instructions to handle dynamic list indexing.
    /// Allocate a fresh segment ID.
    fn alloc_segment_id(&mut self) -> u32 {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        id
    }

    /// Allocate a fresh array ID.
    fn alloc_array_id(&mut self) -> u32 {
        let id = self.next_array_id;
        self.next_array_id += 1;
        id
    }

    /// Dynamic read from a composite using a runtime index.
    /// For small arrays (< MUX_THRESHOLD=100): uses SelectI chains.
    /// For larger arrays: uses DynamicNDArrayGetItem IR (lowered to memory by opt pass).
    fn dynamic_list_subscript(&mut self, data: &CompositeData, idx: &Value) -> Value {
        if data.values.is_empty() {
            return Value::None;
        }
        let n = data.values.len();

        if n < 100 {
            // Mux path: SelectI chain
            let mut result = data.values.last().unwrap().clone();
            for i in (0..n - 1).rev() {
                let const_i = self.builder.ir_constant_int(i as i64);
                let cmp = self.builder.ir_equal_i(idx, &const_i);
                result = self.builder.ir_select_i(&cmp, &data.values[i], &result);
            }
            result
        } else {
            // Memory path: allocate segment, write all values, read at dynamic index
            let seg_id = self.alloc_segment_id();
            let arr_id = self.alloc_array_id();

            // Allocate memory segment
            self.builder.ir_allocate_memory(seg_id, n as u32, 0);

            // Write all values to the segment
            for (i, val) in data.values.iter().enumerate() {
                let addr = self.builder.ir_constant_int(i as i64);
                self.builder.ir_write_memory(seg_id, &addr, val);
            }

            // Read at dynamic index using DynamicNDArrayGetItem
            self.builder.ir_dynamic_ndarray_get_item(arr_id, seg_id, idx)
        }
    }

    /// Dynamic write to a composite at a runtime index.
    /// Returns the updated composite.
    fn dynamic_list_set_item(&mut self, data: &CompositeData, idx: &Value, value: &Value) -> Value {
        let n = data.values.len();
        if n == 0 {
            return Value::List(data.clone());
        }

        if n < 100 {
            // Mux path: for each position, conditionally replace
            let mut new_values = Vec::new();
            let mut new_types = Vec::new();
            for i in 0..n {
                let const_i = self.builder.ir_constant_int(i as i64);
                let cmp = self.builder.ir_equal_i(idx, &const_i);
                let selected = self.builder.ir_select_i(&cmp, value, &data.values[i]);
                new_types.push(selected.zinnia_type());
                new_values.push(selected);
            }
            Value::List(CompositeData { elements_type: new_types, values: new_values })
        } else {
            // Memory path: allocate, write all, then overwrite at dynamic index, read all back
            let seg_id = self.alloc_segment_id();
            let arr_id = self.alloc_array_id();

            // Allocate memory segment
            self.builder.ir_allocate_memory(seg_id, n as u32, 0);

            // Write all original values
            for (i, val) in data.values.iter().enumerate() {
                let addr = self.builder.ir_constant_int(i as i64);
                self.builder.ir_write_memory(seg_id, &addr, val);
            }

            // Write the new value at the dynamic index
            self.builder.ir_dynamic_ndarray_set_item(arr_id, seg_id, idx, value);

            // Read all values back to reconstruct the list
            let mut new_values = Vec::new();
            let mut new_types = Vec::new();
            for i in 0..n {
                let addr = self.builder.ir_constant_int(i as i64);
                let read_val = self.builder.ir_read_memory(seg_id, &addr);
                new_types.push(read_val.zinnia_type());
                new_values.push(read_val);
            }
            Value::List(CompositeData { elements_type: new_types, values: new_values })
        }
    }

    /// Handle range slicing on lists/tuples.
    fn list_slice_range(&mut self, _val: &Value, data: &CompositeData, slice: &SliceIndex) -> Value {
        if let SliceIndex::Range(start, stop, step) = slice {
            let len = data.values.len() as i64;
            let start_idx = start.as_ref().and_then(|v| v.int_val()).unwrap_or(0);
            let stop_idx = stop.as_ref().and_then(|v| v.int_val()).unwrap_or(len);
            let step_val = step.as_ref().and_then(|v| v.int_val()).unwrap_or(1);
            let start_idx = if start_idx < 0 { (len + start_idx).max(0) } else { start_idx.min(len) };
            let stop_idx = if stop_idx < 0 { (len + stop_idx).max(0) } else { stop_idx.min(len) };

            let mut result_values = Vec::new();
            let mut i = start_idx;
            while (step_val > 0 && i < stop_idx) || (step_val < 0 && i > stop_idx) {
                if i >= 0 && (i as usize) < data.values.len() {
                    result_values.push(data.values[i as usize].clone());
                }
                i += step_val;
            }
            let types = result_values.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: result_values })
        } else {
            Value::None
        }
    }

    fn visit_square_brackets(&mut self, n: &ASTSquareBrackets) -> Value {
        let mut values = Vec::new();
        for v in &n.values {
            if let ASTNode::ASTStarredExpr(se) = v {
                // Unpack starred: [*a, 7, 8] → flatten a's elements into the list
                let inner = self.visit(&se.value);
                if let Value::List(data) | Value::Tuple(data) = inner {
                    values.extend(data.values);
                } else {
                    values.push(inner);
                }
            } else {
                values.push(self.visit(v));
            }
        }
        let types: Vec<ZinniaType> = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values,
        })
    }

    fn visit_parenthesis(&mut self, n: &ASTParenthesis) -> Value {
        let mut values = Vec::new();
        for v in &n.values {
            if let ASTNode::ASTStarredExpr(se) = v {
                // Unpack starred: (1, 2, *a) → flatten a's elements into the tuple
                let inner = self.visit(&se.value);
                if let Value::List(data) | Value::Tuple(data) = inner {
                    values.extend(data.values);
                } else {
                    values.push(inner);
                }
            } else {
                values.push(self.visit(v));
            }
        }
        let types: Vec<ZinniaType> = values.iter().map(|v| v.zinnia_type()).collect();
        Value::Tuple(CompositeData {
            elements_type: types,
            values,
        })
    }

    fn visit_generator_exp(&mut self, n: &ASTGeneratorExp) -> Value {
        // Simple implementation: expand generators into a flat list of values
        // Only handles single generator for now
        if n.generators.is_empty() {
            return Value::List(CompositeData { elements_type: vec![], values: vec![] });
        }

        let gen = &n.generators[0];
        let iter_val = self.visit(&gen.iter_expr);

        let elements: Vec<Value> = match &iter_val {
            Value::List(data) | Value::Tuple(data) => data.values.clone(),
            _ => return Value::None,
        };

        let mut result_values = Vec::new();
        for elem in &elements {
            // Bind the generator target variable
            self.do_recursive_assign(&gen.target, elem.clone(), false);

            // Check if conditions
            let mut passes = true;
            for if_expr in &gen.ifs {
                let cond = self.visit(if_expr);
                let cond_static = cond.bool_val().or_else(|| cond.int_val().map(|v| v != 0));
                if cond_static == Some(false) {
                    passes = false;
                    break;
                }
            }

            if passes {
                let val = self.visit(&n.elt);
                result_values.push(val);
            }
        }

        let types: Vec<ZinniaType> = result_values.iter().map(|v| v.zinnia_type()).collect();
        if n.kind == "list" {
            Value::List(CompositeData { elements_type: types, values: result_values })
        } else {
            Value::Tuple(CompositeData { elements_type: types, values: result_values })
        }
    }

    fn visit_cond_exp(&mut self, n: &ASTCondExp) -> Value {
        let cond = self.visit(&n.cond);
        let true_val = self.visit(&n.t_expr);
        let false_val = self.visit(&n.f_expr);
        let cond_bool = self.to_scalar_bool(&cond);
        self.select_value(&cond_bool, &true_val, &false_val)
    }

    fn visit_joined_str(&mut self, n: &ASTJoinedStr) -> Value {
        let values: Vec<Value> = n.values.iter().map(|v| self.visit(v)).collect();
        let mut result = self.builder.ir_constant_str(String::new());
        for val in values {
            result = self.builder.ir_add_str(&result, &val);
        }
        result
    }

    fn visit_formatted_value(&mut self, n: &ASTFormattedValue) -> Value {
        let val = self.visit(&n.value);
        // Convert to string
        match &val {
            Value::Integer(_) | Value::Boolean(_) => self.builder.ir_str_i(&val),
            Value::Float(_) => self.builder.ir_str_f(&val),
            Value::String(_) => val,
            _ => self.builder.ir_str_i(&val), // fallback
        }
    }

    // ── Assignment ────────────────────────────────────────────────────

    fn do_recursive_assign(&mut self, target: &ASTNode, value: Value, conditional_select: bool) {
        match target {
            ASTNode::ASTNameAssignTarget(t) => {
                // Only do conditional merge when inside a conditional/loop scope
                // that has a non-trivial condition
                let should_merge = self.ctx.exists(&t.name)
                    && conditional_select
                    && self.ctx.has_nontrivial_condition();
                if should_merge {
                    let orig = self.ctx.get(&t.name).unwrap();
                    let cond = self.ctx.get_condition_value(&mut self.builder);
                    let merged = self.select_value(&cond, &value, &orig);
                    self.ctx.set(&t.name, merged);
                } else {
                    self.ctx.set(&t.name, value);
                }
            }
            ASTNode::ASTTupleAssignTarget(t) => {
                if let Value::Tuple(data) | Value::List(data) = &value {
                    // Check for starred target — the `star` flag is on ASTNameAssignTarget
                    let star_idx = t.targets.iter().position(|tgt| is_starred_target(tgt));
                    if let Some(si) = star_idx {
                        // Starred unpacking: a, *b, c = (1, 2, 3, 4)
                        let n_before = si;
                        let n_after = t.targets.len() - si - 1;
                        let n_values = data.values.len();
                        if n_values < n_before + n_after {
                            panic!("UnpackingError: not enough values to unpack (expected at least {}, got {})",
                                n_before + n_after, n_values);
                        }
                        // Assign before-star targets
                        for i in 0..n_before {
                            self.do_recursive_assign(&t.targets[i], data.values[i].clone(), conditional_select);
                        }
                        // Assign starred target (collect middle into list)
                        let star_count = n_values - n_before - n_after;
                        let star_values: Vec<Value> = data.values[n_before..n_before + star_count].to_vec();
                        let star_types = star_values.iter().map(|v| v.zinnia_type()).collect();
                        let star_list = Value::List(CompositeData { elements_type: star_types, values: star_values });
                        // Assign to the starred target directly (star flag is on the target itself)
                        self.do_recursive_assign(&t.targets[si], star_list, conditional_select);
                        // Assign after-star targets
                        for i in 0..n_after {
                            self.do_recursive_assign(&t.targets[si + 1 + i], data.values[n_before + star_count + i].clone(), conditional_select);
                        }
                    } else {
                        // No starred target — exact match required
                        if t.targets.len() != data.values.len() {
                            panic!("UnpackingError: cannot unpack {} values into {} targets",
                                data.values.len(), t.targets.len());
                        }
                        for (i, tgt) in t.targets.iter().enumerate() {
                            if matches!(tgt, ASTNode::ASTTupleAssignTarget(_) | ASTNode::ASTListAssignTarget(_)) {
                                if !matches!(&data.values[i], Value::List(_) | Value::Tuple(_)) {
                                    let tp = data.values[i].zinnia_type();
                                    panic!("TypeInferenceError: {} is not iterable", tp);
                                }
                            }
                            self.do_recursive_assign(tgt, data.values[i].clone(), conditional_select);
                        }
                    }
                } else {
                    panic!("UnpackingError: cannot unpack non-iterable value");
                }
            }
            ASTNode::ASTListAssignTarget(t) => {
                if let Value::Tuple(data) | Value::List(data) = &value {
                    for (i, tgt) in t.targets.iter().enumerate() {
                        if i < data.values.len() {
                            self.do_recursive_assign(tgt, data.values[i].clone(), conditional_select);
                        }
                    }
                }
            }
            ASTNode::ASTSubscriptAssignTarget(t) => {
                // Subscript assignment: e.g. array[0, 1] = value or lst[0] = value
                // Get the variable name from target
                if let ASTNode::ASTLoad(load) = &*t.target {
                    let var_name = &load.name;
                    let indices = self.eval_slice_indices(&t.slicing);
                    if let Some(current) = self.ctx.get(var_name) {
                        // Check if target is a tuple — tuples don't support item assignment
                        if matches!(&current, Value::Tuple(_)) {
                            panic!("'tuple' object does not support item assignment");
                        }
                        let updated = self.set_nested_value(current, &indices, value);
                        self.ctx.set(var_name, updated);
                    }
                }
            }
            _ => {
                // Unsupported assignment target
            }
        }
    }

    // ── Built-in function helpers ────────────────────────────────────

    fn builtin_range(&mut self, args: &[Value]) -> Value {
        let (start, stop, step) = match args.len() {
            1 => (0i64, args[0].int_val().unwrap_or(0), 1i64),
            2 => (args[0].int_val().unwrap_or(0), args[1].int_val().unwrap_or(0), 1i64),
            3 => (args[0].int_val().unwrap_or(0), args[1].int_val().unwrap_or(0), args[2].int_val().unwrap_or(1)),
            _ => return Value::None,
        };
        if step == 0 { return Value::None; }
        let mut values = Vec::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            values.push(self.builder.ir_constant_int(i));
            i += step;
        }
        let types = vec![ZinniaType::Integer; values.len()];
        Value::List(CompositeData { elements_type: types, values })
    }

    fn builtin_len(&mut self, args: &[Value]) -> Value {
        if let Some(val) = args.first() {
            match val {
                Value::List(data) | Value::Tuple(data) => {
                    self.builder.ir_constant_int(data.values.len() as i64)
                }
                _ => self.builder.ir_constant_int(0),
            }
        } else {
            self.builder.ir_constant_int(0)
        }
    }

    fn builtin_enumerate(&mut self, iter_val: &Value) -> Value {
        match iter_val {
            Value::List(data) | Value::Tuple(data) => {
                let mut result = Vec::new();
                for (i, elem) in data.values.iter().enumerate() {
                    let idx = self.builder.ir_constant_int(i as i64);
                    result.push(Value::Tuple(CompositeData {
                        elements_type: vec![ZinniaType::Integer, elem.zinnia_type()],
                        values: vec![idx, elem.clone()],
                    }));
                }
                let types = result.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: result })
            }
            _ => Value::None,
        }
    }

    fn builtin_reduce(&mut self, op: &str, val: &Value) -> Value {
        // Flatten to scalar elements first (handles nested composites like 2D arrays)
        let elements = self.flatten_composite(val);
        if elements.is_empty() {
            return match op {
                "sum" => self.builder.ir_constant_int(0),
                "any" => self.builder.ir_constant_bool(false),
                "all" => self.builder.ir_constant_bool(true),
                "prod" => self.builder.ir_constant_int(1),
                "min" | "max" => Value::None,
                _ => Value::None,
            };
        }
        match op {
            "sum" => {
                let mut acc = elements[0].clone();
                for elem in &elements[1..] {
                    acc = self.builder.ir_add_i(&acc, elem);
                }
                acc
            }
            "any" => {
                let mut acc = self.to_scalar_bool(&elements[0]);
                for elem in &elements[1..] {
                    let b = self.to_scalar_bool(elem);
                    acc = self.builder.ir_logical_or(&acc, &b);
                }
                acc
            }
            "all" => {
                let mut acc = self.to_scalar_bool(&elements[0]);
                for elem in &elements[1..] {
                    let b = self.to_scalar_bool(elem);
                    acc = self.builder.ir_logical_and(&acc, &b);
                }
                acc
            }
            "min" => {
                let mut acc = elements[0].clone();
                for elem in &elements[1..] {
                    let cond = self.builder.ir_less_than_i(&acc, elem);
                    acc = self.builder.ir_select_i(&cond, &acc, elem);
                }
                acc
            }
            "max" => {
                let mut acc = elements[0].clone();
                for elem in &elements[1..] {
                    let cond = self.builder.ir_greater_than_i(&acc, elem);
                    acc = self.builder.ir_select_i(&cond, &acc, elem);
                }
                acc
            }
            "prod" => {
                let mut acc = elements[0].clone();
                for elem in &elements[1..] {
                    acc = self.builder.ir_mul_i(&acc, elem);
                }
                acc
            }
            _ => Value::None,
        }
    }

    /// Matrix multiplication with float-awareness and full matrix-matrix support.
    fn matmul(&mut self, lhs: &Value, rhs: &Value) -> Value {
        let lhs_shape = self.get_composite_shape(lhs);
        let rhs_shape = self.get_composite_shape(rhs);

        // Scalar case
        if lhs_shape.is_empty() || rhs_shape.is_empty() {
            return self.apply_binary_op("mul", lhs, rhs);
        }

        let lhs_cols = *lhs_shape.last().unwrap();
        let rhs_rows = if rhs_shape.len() >= 1 { rhs_shape[0] } else { 1 };

        if lhs_cols != rhs_rows {
            panic!("their shapes are not multiply compatible: ({}) and ({})",
                lhs_shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
                rhs_shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
            );
        }

        // Determine if we should use float ops (if either operand has float elements)
        let lhs_flat = self.flatten_composite(lhs);
        let rhs_flat = self.flatten_composite(rhs);
        let use_float = lhs_flat.iter().any(|v| matches!(v, Value::Float(_)))
            || rhs_flat.iter().any(|v| matches!(v, Value::Float(_)));

        if let (Value::List(ld), Value::List(rd)) = (lhs, rhs) {
            if rhs_shape.len() == 1 {
                // Matrix @ vector or vector @ vector
                if lhs_shape.len() == 1 {
                    // 1D @ 1D: dot product → scalar
                    return self.matmul_dot(&ld.values, &rd.values, use_float);
                }
                // 2D @ 1D: each row dot product with vector → 1D
                let mut results = Vec::new();
                for row in &ld.values {
                    if let Value::List(row_data) | Value::Tuple(row_data) = row {
                        results.push(self.matmul_dot(&row_data.values, &rd.values, use_float));
                    }
                }
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                return Value::List(CompositeData { elements_type: types, values: results });
            }

            if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
                // 2D @ 2D: full matrix multiply
                let m = lhs_shape[0]; // rows of lhs
                let k = lhs_shape[1]; // cols of lhs = rows of rhs
                let n = rhs_shape[1]; // cols of rhs

                let mut rows = Vec::new();
                for i in 0..m {
                    let lhs_row = match &ld.values[i] {
                        Value::List(r) | Value::Tuple(r) => &r.values,
                        _ => panic!("matmul: expected 2D array"),
                    };
                    let mut row_vals = Vec::new();
                    for j in 0..n {
                        // Compute dot product of lhs row i with rhs column j
                        let zero = if use_float {
                            self.builder.ir_constant_float(0.0)
                        } else {
                            self.builder.ir_constant_int(0)
                        };
                        let mut acc = zero;
                        for kk in 0..k {
                            let rhs_row = match &rd.values[kk] {
                                Value::List(r) | Value::Tuple(r) => &r.values,
                                _ => panic!("matmul: expected 2D array"),
                            };
                            let prod = if use_float {
                                let a = self.ensure_float(&lhs_row[kk]);
                                let b = self.ensure_float(&rhs_row[j]);
                                self.builder.ir_mul_f(&a, &b)
                            } else {
                                self.builder.ir_mul_i(&lhs_row[kk], &rhs_row[j])
                            };
                            acc = if use_float {
                                self.builder.ir_add_f(&acc, &prod)
                            } else {
                                self.builder.ir_add_i(&acc, &prod)
                            };
                        }
                        row_vals.push(acc);
                    }
                    let rtypes = row_vals.iter().map(|v| v.zinnia_type()).collect();
                    rows.push(Value::List(CompositeData { elements_type: rtypes, values: row_vals }));
                }
                let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
                return Value::List(CompositeData { elements_type: row_types, values: rows });
            }
        }

        // Fallback: scalar multiply
        self.apply_binary_op("mul", lhs, rhs)
    }

    /// Dot product helper for matmul.
    fn matmul_dot(&mut self, a: &[Value], b: &[Value], use_float: bool) -> Value {
        let zero = if use_float {
            self.builder.ir_constant_float(0.0)
        } else {
            self.builder.ir_constant_int(0)
        };
        let mut acc = zero;
        for (x, y) in a.iter().zip(b.iter()) {
            let prod = if use_float {
                let xf = self.ensure_float(x);
                let yf = self.ensure_float(y);
                self.builder.ir_mul_f(&xf, &yf)
            } else {
                self.builder.ir_mul_i(x, y)
            };
            acc = if use_float {
                self.builder.ir_add_f(&acc, &prod)
            } else {
                self.builder.ir_add_i(&acc, &prod)
            };
        }
        acc
    }

    /// Ensure a value is Float, casting from Int if needed.
    fn ensure_float(&mut self, val: &Value) -> Value {
        match val {
            Value::Float(_) => val.clone(),
            _ => self.builder.ir_float_cast(val),
        }
    }

    /// Multi-dimensional ndarray-style subscript: array[i, j], array[0, :], array[:, 0], etc.
    fn multidim_subscript(&mut self, data: &CompositeData, indices: &[SliceIndex]) -> Value {
        if indices.is_empty() {
            return Value::List(data.clone());
        }

        match &indices[0] {
            SliceIndex::Single(idx_value) => {
                if let Some(idx) = idx_value.int_val() {
                    let i = if idx < 0 { (data.values.len() as i64 + idx) as usize } else { idx as usize };
                    if i >= data.values.len() {
                        return Value::None;
                    }
                    if indices.len() == 1 {
                        return data.values[i].clone();
                    }
                    // Recurse into the selected element
                    match &data.values[i] {
                        Value::List(inner) | Value::Tuple(inner) => {
                            self.multidim_subscript(inner, &indices[1..])
                        }
                        _ => data.values[i].clone(),
                    }
                } else {
                    // Dynamic index
                    if indices.len() == 1 {
                        return self.dynamic_list_subscript(data, idx_value);
                    }
                    // Dynamic index with further dimensions: select from each possible row
                    // For each possible index value, apply the remaining indices
                    let mut results: Vec<Value> = Vec::new();
                    for elem in &data.values {
                        if let Value::List(inner) | Value::Tuple(inner) = elem {
                            results.push(self.multidim_subscript(inner, &indices[1..]));
                        } else {
                            results.push(elem.clone());
                        }
                    }
                    // Now select from results using the dynamic index
                    let result_data = CompositeData {
                        elements_type: results.iter().map(|v| v.zinnia_type()).collect(),
                        values: results,
                    };
                    self.dynamic_list_subscript(&result_data, idx_value)
                }
            }
            SliceIndex::Range(start, stop, step) => {
                let len = data.values.len() as i64;
                let s = start.as_ref().and_then(|v| v.int_val()).unwrap_or(0);
                let e = stop.as_ref().and_then(|v| v.int_val()).unwrap_or(len);
                let st = step.as_ref().and_then(|v| v.int_val()).unwrap_or(1);
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;

                let mut selected = Vec::new();
                let mut i = s;
                while (st > 0 && i < e) || (st < 0 && i > e) {
                    if i < data.values.len() {
                        if indices.len() == 1 {
                            selected.push(data.values[i].clone());
                        } else {
                            // Apply remaining indices to each selected element
                            match &data.values[i] {
                                Value::List(inner) | Value::Tuple(inner) => {
                                    selected.push(self.multidim_subscript(inner, &indices[1..]));
                                }
                                _ => selected.push(data.values[i].clone()),
                            }
                        }
                    }
                    i = (i as i64 + st) as usize;
                }
                let types = selected.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: selected })
            }
        }
    }

    /// Element-wise binary operation on two composites (bypasses list concatenation).
    fn elementwise_binary(&mut self, op: &str, a: &Value, b: &Value) -> Value {
        match (a, b) {
            (Value::List(ad), Value::List(bd)) | (Value::Tuple(ad), Value::List(bd))
            | (Value::List(ad), Value::Tuple(bd)) | (Value::Tuple(ad), Value::Tuple(bd))
                if ad.values.len() == bd.values.len() => {
                let results: Vec<Value> = ad.values.iter().zip(bd.values.iter())
                    .map(|(x, y)| self.elementwise_binary(op, x, y))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: results })
            }
            _ => self.apply_scalar_binary_op(op, a, b),
        }
    }

    /// Element-wise min or max of two composites.
    fn elementwise_minmax(&mut self, a: &Value, b: &Value, is_max: bool) -> Value {
        match (a, b) {
            (Value::List(ad), Value::List(bd)) | (Value::Tuple(ad), Value::List(bd))
            | (Value::List(ad), Value::Tuple(bd)) | (Value::Tuple(ad), Value::Tuple(bd))
                if ad.values.len() == bd.values.len() => {
                let results: Vec<Value> = ad.values.iter().zip(bd.values.iter())
                    .map(|(x, y)| self.elementwise_minmax(x, y, is_max))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: results })
            }
            _ => {
                let cond = if is_max {
                    self.builder.ir_greater_than_i(a, b)
                } else {
                    self.builder.ir_less_than_i(a, b)
                };
                self.builder.ir_select_i(&cond, a, b)
            }
        }
    }

    /// Reduce along a specific axis.
    /// For a 2D array with axis=0: reduce columns (result is 1D with same ncols)
    /// For a 2D array with axis=1: reduce rows (result is 1D with same nrows)
    fn reduce_with_axis(&mut self, op: &str, val: &Value, axis: i64) -> Value {
        if let Value::List(outer) | Value::Tuple(outer) = val {
            let ndim = self.get_composite_shape(val).len();
            let axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };

            if axis == 0 {
                // Reduce along axis 0: for each column position, reduce across rows
                if outer.values.is_empty() { return Value::None; }
                // Get the number of columns from the first row
                if let Value::List(first_row) | Value::Tuple(first_row) = &outer.values[0] {
                    let ncols = first_row.values.len();
                    let mut results = Vec::new();
                    for col in 0..ncols {
                        // Collect all values in this column
                        let mut col_vals = Vec::new();
                        for row in &outer.values {
                            if let Value::List(rd) | Value::Tuple(rd) = row {
                                if col < rd.values.len() {
                                    col_vals.push(rd.values[col].clone());
                                }
                            }
                        }
                        let col_list = Value::List(CompositeData {
                            elements_type: col_vals.iter().map(|v| v.zinnia_type()).collect(),
                            values: col_vals,
                        });
                        results.push(self.builtin_reduce(op, &col_list));
                    }
                    let types = results.iter().map(|v| v.zinnia_type()).collect();
                    Value::List(CompositeData { elements_type: types, values: results })
                } else {
                    // If first element is scalar, just reduce the whole thing
                    self.builtin_reduce(op, val)
                }
            } else if axis == 1 {
                // Reduce along axis 1: for each row, reduce to scalar
                let mut results = Vec::new();
                for row in &outer.values {
                    results.push(self.builtin_reduce(op, row));
                }
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: results })
            } else {
                self.builtin_reduce(op, val)
            }
        } else {
            self.builtin_reduce(op, val)
        }
    }

    // ── Numpy-like helpers ────────────────────────────────────────────

    fn np_fill(&mut self, args: &[Value], kwargs: &HashMap<String, Value>, fill_value: i64) -> Value {
        // np.zeros(shape, dtype=...) / np.ones(shape, dtype=...)
        let shape = if let Some(arg) = args.first() {
            match arg {
                Value::Integer(_) => vec![arg.int_val().unwrap_or(0) as usize],
                Value::Tuple(data) => data.values.iter().map(|v| v.int_val().unwrap_or(0) as usize).collect(),
                Value::List(data) => data.values.iter().map(|v| v.int_val().unwrap_or(0) as usize).collect(),
                _ => vec![0],
            }
        } else {
            return Value::None;
        };
        let use_float = matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Float)));
        let total: usize = shape.iter().product();
        let (fill, elem_type) = if use_float {
            (self.builder.ir_constant_float(fill_value as f64), ZinniaType::Float)
        } else {
            (self.builder.ir_constant_int(fill_value), ZinniaType::Integer)
        };
        let values = vec![fill; total];
        let types = vec![elem_type; total];
        self.build_ndarray_from_flat(values, types, &shape)
    }

    fn np_identity(&mut self, args: &[Value]) -> Value {
        let n = args.first().and_then(|a| a.int_val()).unwrap_or(0) as usize;
        let zero = self.builder.ir_constant_int(0);
        let one = self.builder.ir_constant_int(1);
        let mut rows = Vec::new();
        for i in 0..n {
            let mut row_vals = Vec::new();
            for j in 0..n {
                row_vals.push(if i == j { one.clone() } else { zero.clone() });
            }
            let row_types = vec![ZinniaType::Integer; n];
            rows.push(Value::List(CompositeData { elements_type: row_types, values: row_vals }));
        }
        let types = rows.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: types, values: rows })
    }

    fn np_arange(&mut self, args: &[Value]) -> Value {
        self.builtin_range(args)
    }

    fn np_linspace(&mut self, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
        if args.len() < 2 { return Value::None; }
        let start = args[0].float_val().or_else(|| args[0].int_val().map(|v| v as f64)).unwrap_or(0.0);
        let stop = args[1].float_val().or_else(|| args[1].int_val().map(|v| v as f64)).unwrap_or(0.0);
        let num = args.get(2).and_then(|v| v.int_val()).or_else(|| kwargs.get("num").and_then(|v| v.int_val())).unwrap_or(50) as usize;
        let endpoint = kwargs.get("endpoint").and_then(|v| v.bool_val()).unwrap_or(true);
        let use_int = matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Integer)));

        if num == 0 {
            return Value::List(CompositeData { elements_type: vec![], values: vec![] });
        }
        if num == 1 {
            let v = if use_int { self.builder.ir_constant_int(start as i64) } else { self.builder.ir_constant_float(start) };
            let t = if use_int { ZinniaType::Integer } else { ZinniaType::Float };
            return Value::List(CompositeData { elements_type: vec![t], values: vec![v] });
        }

        let divisor = if endpoint { (num - 1) as f64 } else { num as f64 };
        let step = (stop - start) / divisor;
        let mut values = Vec::new();
        for i in 0..num {
            let fval = start + step * i as f64;
            if use_int {
                values.push(self.builder.ir_constant_int(fval as i64));
            } else {
                values.push(self.builder.ir_constant_float(fval));
            }
        }
        let elem_type = if use_int { ZinniaType::Integer } else { ZinniaType::Float };
        let types = vec![elem_type; values.len()];
        Value::List(CompositeData { elements_type: types, values })
    }

    fn np_allclose(&mut self, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
        // np.allclose(a, b, rtol=1e-05, atol=1e-08) — check |a - b| <= atol + rtol * |b|
        if args.len() < 2 { return Value::None; }

        // Validate argument types
        fn is_valid_allclose_arg(val: &Value) -> bool {
            matches!(val, Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::List(_) | Value::Tuple(_))
        }
        if !is_valid_allclose_arg(&args[0]) {
            panic!("Unsupported argument type for `lhs`");
        }
        if !is_valid_allclose_arg(&args[1]) {
            panic!("Unsupported argument type for `rhs`");
        }
        if let Some(atol) = kwargs.get("atol").or_else(|| args.get(3)) {
            if !is_valid_allclose_arg(atol) {
                panic!("Unsupported argument type for `atol`");
            }
        }
        if let Some(rtol) = kwargs.get("rtol").or_else(|| args.get(2)) {
            if !is_valid_allclose_arg(rtol) {
                panic!("Unsupported argument type for `rtol`");
            }
        }

        // Extract atol and rtol from kwargs or positional args
        let default_rtol = 1e-5_f64;
        let default_atol = 1e-8_f64;

        fn extract_scalar_float(val: &Value) -> Option<f64> {
            match val {
                Value::Float(s) => s.static_val,
                Value::Integer(s) => s.static_val.map(|i| i as f64),
                Value::List(d) | Value::Tuple(d) if d.values.len() == 1 => {
                    extract_scalar_float(&d.values[0])
                }
                _ => None,
            }
        }

        let rtol = kwargs.get("rtol")
            .and_then(|v| extract_scalar_float(v))
            .or_else(|| args.get(2).and_then(|v| extract_scalar_float(v)))
            .unwrap_or(default_rtol);
        let atol = kwargs.get("atol")
            .and_then(|v| extract_scalar_float(v))
            .or_else(|| args.get(3).and_then(|v| extract_scalar_float(v)))
            .unwrap_or(default_atol);

        let a_flat = self.flatten_composite(&args[0]);
        let b_flat = self.flatten_composite(&args[1]);

        // Handle broadcasting: scalar vs array
        let (a_elems, b_elems) = if a_flat.len() == 1 && b_flat.len() > 1 {
            (vec![a_flat[0].clone(); b_flat.len()], b_flat)
        } else if b_flat.len() == 1 && a_flat.len() > 1 {
            let b_repeated = vec![b_flat[0].clone(); a_flat.len()];
            (a_flat, b_repeated)
        } else {
            (a_flat, b_flat)
        };

        if a_elems.len() != b_elems.len() {
            return self.builder.ir_constant_bool(false);
        }

        // For each element: |a - b| <= atol + rtol * |b|
        // Since we're in ZK, use static evaluation for compile-time known values
        let mut result = self.builder.ir_constant_bool(true);
        for (a, b) in a_elems.iter().zip(b_elems.iter()) {
            let a_f = a.float_val().or_else(|| a.int_val().map(|i| i as f64));
            let b_f = b.float_val().or_else(|| b.int_val().map(|i| i as f64));

            if let (Some(av), Some(bv)) = (a_f, b_f) {
                let diff = (av - bv).abs();
                let threshold = atol + rtol * bv.abs();
                let close = diff <= threshold;
                let close_val = self.builder.ir_constant_bool(close);
                result = self.builder.ir_logical_and(&result, &close_val);
            } else {
                // Dynamic: fall back to exact equality
                let eq = self.builder.ir_equal_i(a, b);
                result = self.builder.ir_logical_and(&result, &eq);
            }
        }
        result
    }

    fn np_concatenate(&mut self, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
        let axis = kwargs.get("axis")
            .or_else(|| args.get(1))
            .and_then(|v| v.int_val())
            .unwrap_or(0);

        if let Some(Value::List(data) | Value::Tuple(data)) = args.first() {
            // Validate axis bounds
            if !data.values.is_empty() {
                let ndim = self.get_composite_shape(&data.values[0]).len();
                let resolved_axis = if axis < 0 { ndim as i64 + axis } else { axis };
                if resolved_axis < 0 || resolved_axis >= ndim as i64 {
                    panic!("axis {} is out of bounds for array with {} dimensions", axis, ndim);
                }
            }
            if axis == 0 {
                // Concatenate along axis 0: just flatten one level
                let mut all_values = Vec::new();
                let mut all_types = Vec::new();
                for arr in &data.values {
                    match arr {
                        Value::List(d) | Value::Tuple(d) => {
                            all_values.extend(d.values.clone());
                            all_types.extend(d.elements_type.clone());
                        }
                        _ => { all_values.push(arr.clone()); all_types.push(arr.zinnia_type()); }
                    }
                }
                Value::List(CompositeData { elements_type: all_types, values: all_values })
            } else if axis == 1 {
                // Concatenate along axis 1: merge inner rows
                // [[1,2],[3,4]] + [[5,6],[7,8]] axis=1 → [[1,2,5,6],[3,4,7,8]]
                if data.values.is_empty() { return Value::None; }
                let num_arrays = data.values.len();
                // Get number of rows from first array
                let first = &data.values[0];
                if let Value::List(first_data) | Value::Tuple(first_data) = first {
                    let nrows = first_data.values.len();
                    let mut result_rows = Vec::new();
                    for row_idx in 0..nrows {
                        let mut row_values = Vec::new();
                        let mut row_types = Vec::new();
                        for arr_idx in 0..num_arrays {
                            if let Value::List(arr_data) | Value::Tuple(arr_data) = &data.values[arr_idx] {
                                if row_idx < arr_data.values.len() {
                                    match &arr_data.values[row_idx] {
                                        Value::List(rd) | Value::Tuple(rd) => {
                                            row_values.extend(rd.values.clone());
                                            row_types.extend(rd.elements_type.clone());
                                        }
                                        v => { row_values.push(v.clone()); row_types.push(v.zinnia_type()); }
                                    }
                                }
                            }
                        }
                        result_rows.push(Value::List(CompositeData { elements_type: row_types, values: row_values }));
                    }
                    let types = result_rows.iter().map(|v| v.zinnia_type()).collect();
                    Value::List(CompositeData { elements_type: types, values: result_rows })
                } else {
                    Value::None
                }
            } else {
                Value::None
            }
        } else {
            Value::None
        }
    }

    fn np_stack(&mut self, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
        // np.stack(arrays, axis=0) — stack arrays along a new axis
        let axis = kwargs.get("axis")
            .or_else(|| args.get(1))
            .and_then(|v| v.int_val())
            .unwrap_or(0);

        if let Some(Value::List(data) | Value::Tuple(data)) = args.first() {
            // Validate axis bounds
            if !data.values.is_empty() {
                let ndim = self.get_composite_shape(&data.values[0]).len() + 1;
                let resolved_axis = if axis < 0 { ndim as i64 + axis } else { axis };
                if resolved_axis < 0 || resolved_axis >= ndim as i64 {
                    panic!("axis {} is out of bounds for array of dimension {}", axis, ndim - 1);
                }
            }
            if axis == 0 {
                // Stack along axis 0: just wrap arrays as rows
                let types = data.values.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: data.values.clone() })
            } else if axis == 1 {
                // Stack along axis 1: transpose-like — zip elements from each array
                // e.g., stack([[1,2,3], [4,5,6]], axis=1) = [[1,4],[2,5],[3,6]]
                if let Some(Value::List(first) | Value::Tuple(first)) = data.values.first() {
                    let n_elements = first.values.len();
                    let mut result = Vec::new();
                    for i in 0..n_elements {
                        let mut row = Vec::new();
                        for arr in &data.values {
                            if let Value::List(d) | Value::Tuple(d) = arr {
                                if i < d.values.len() {
                                    row.push(d.values[i].clone());
                                }
                            }
                        }
                        let types = row.iter().map(|v| v.zinnia_type()).collect();
                        result.push(Value::List(CompositeData { elements_type: types, values: row }));
                    }
                    let types = result.iter().map(|v| v.zinnia_type()).collect();
                    Value::List(CompositeData { elements_type: types, values: result })
                } else {
                    Value::None
                }
            } else {
                // Higher axes — not common, fall back to axis=0
                let types = data.values.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: data.values.clone() })
            }
        } else {
            Value::None
        }
    }

    fn flatten_composite(&self, val: &Value) -> Vec<Value> {
        match val {
            Value::List(data) | Value::Tuple(data) => {
                let mut flat = Vec::new();
                for v in &data.values {
                    flat.extend(self.flatten_composite(v));
                }
                flat
            }
            other => vec![other.clone()],
        }
    }

    fn build_ndarray_from_flat(&mut self, values: Vec<Value>, types: Vec<ZinniaType>, shape: &[usize]) -> Value {
        if shape.len() == 1 {
            Value::List(CompositeData { elements_type: types, values })
        } else {
            // Build nested structure
            let inner_size: usize = shape[1..].iter().product();
            let mut rows = Vec::new();
            for chunk in values.chunks(inner_size) {
                let chunk_types = chunk.iter().map(|v| v.zinnia_type()).collect();
                rows.push(self.build_ndarray_from_flat(chunk.to_vec(), chunk_types, &shape[1..]));
            }
            let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: row_types, values: rows })
        }
    }

    // ── NDArray helpers ───────────────────────────────────────────────

    fn ndarray_transpose(&mut self, val: &Value, args: &[Value]) -> Value {
        // Determine the shape of the input
        let shape = self.get_composite_shape(val);
        let ndim = shape.len();
        if ndim <= 1 { return val.clone(); }

        // Determine axis permutation — check length before validating individual values
        let raw_axes: Vec<i64> = if args.is_empty() || matches!(args.first(), Some(Value::None)) {
            (0..ndim as i64).rev().collect()
        } else if let Some(Value::Tuple(perm_data)) | Some(Value::List(perm_data)) = args.first() {
            perm_data.values.iter().map(|v| v.int_val().unwrap_or(0)).collect()
        } else {
            args.iter().map(|v| v.int_val().unwrap_or(0)).collect()
        };

        // Check length first (before resolving individual values)
        if raw_axes.len() != ndim {
            panic!("Length of `axes` should be equal to the number of dimensions of the array (expected {}, got {})", ndim, raw_axes.len());
        }

        let axes: Vec<usize> = raw_axes.iter().map(|&a| {
            let resolved = if a < 0 { ndim as i64 + a } else { a };
            if resolved < 0 || resolved >= ndim as i64 {
                panic!("Invalid axis value: {} is out of bounds for array with {} dimensions", a, ndim);
            }
            resolved as usize
        }).collect();
        // Check for invalid axis values
        for &a in &axes {
            if a >= ndim {
                panic!("Invalid axis value: {} is out of bounds for array with {} dimensions", a, ndim);
            }
        }
        // Check for valid permutation (no duplicates)
        let mut seen = vec![false; ndim];
        for &a in &axes {
            if seen[a] {
                panic!("axes should be a permutation of 0 to {}", ndim - 1);
            }
            seen[a] = true;
        }

        // Calculate output shape
        let out_shape: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();

        // Flatten the input, then reassemble in transposed order
        let flat = self.flatten_composite(val);
        if flat.is_empty() { return val.clone(); }

        // Compute strides for input
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            in_strides[i] = in_strides[i + 1] * shape[i + 1];
        }
        // Compute strides for output
        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        let total: usize = shape.iter().product();
        let mut out_flat = vec![Value::None; total];

        // For each element in the flat array, compute its input index tuple,
        // permute it, and write to the output position
        for flat_idx in 0..total {
            // Compute input multi-index
            let mut remainder = flat_idx;
            let mut in_idx = vec![0usize; ndim];
            for d in 0..ndim {
                in_idx[d] = remainder / in_strides[d];
                remainder %= in_strides[d];
            }
            // Permute to get output multi-index
            let mut out_idx = vec![0usize; ndim];
            for d in 0..ndim {
                out_idx[d] = in_idx[axes[d]];
            }
            // Compute output flat index
            let mut out_flat_idx = 0;
            for d in 0..ndim {
                out_flat_idx += out_idx[d] * out_strides[d];
            }
            out_flat[out_flat_idx] = flat[flat_idx].clone();
        }

        // Rebuild nested structure from output shape
        let types = out_flat.iter().map(|v| v.zinnia_type()).collect();
        self.build_nested_value(out_flat, types, &out_shape)
    }

    /// Get the shape of a nested composite value.
    fn get_composite_shape(&self, val: &Value) -> Vec<usize> {
        match val {
            Value::List(data) | Value::Tuple(data) => {
                if data.values.is_empty() {
                    return vec![0];
                }
                let mut shape = vec![data.values.len()];
                // Recurse into first element to get inner dimensions
                let inner_shape = self.get_composite_shape(&data.values[0]);
                if inner_shape.len() > 0 && !matches!(&data.values[0], Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::String(_) | Value::None | Value::Class(_)) {
                    shape.extend(inner_shape);
                }
                shape
            }
            Value::NDArray(nd) => nd.shape.clone(),
            _ => vec![],
        }
    }

    /// NDArray reshape: flatten, then rebuild with new shape.
    fn ndarray_reshape(&mut self, val: &Value, args: &[Value]) -> Value {
        let flat = self.flatten_composite(val);
        let total = flat.len();

        // Parse new shape from args — single tuple arg or multiple int args
        let new_shape: Vec<usize> = if args.len() == 1 {
            match &args[0] {
                Value::Tuple(data) | Value::List(data) => {
                    data.values.iter().map(|v| v.int_val().expect("reshape: shape elements must be constant ints") as usize).collect()
                }
                Value::Integer(_) => vec![args[0].int_val().unwrap() as usize],
                _ => panic!("reshape: invalid shape argument"),
            }
        } else {
            args.iter().map(|v| v.int_val().expect("reshape: shape elements must be constant ints") as usize).collect()
        };

        // Handle -1 (infer one dimension)
        let neg_count = new_shape.iter().filter(|&&s| s == usize::MAX).count();
        let final_shape: Vec<usize> = if neg_count == 1 {
            let known_product: usize = new_shape.iter().filter(|&&s| s != usize::MAX).product();
            assert!(known_product > 0 && total % known_product == 0, "reshape: cannot infer dimension");
            new_shape.iter().map(|&s| if s == usize::MAX { total / known_product } else { s }).collect()
        } else {
            // Also handle -1 encoded as a large number from i64 cast
            let has_neg = args.iter().any(|a| a.int_val() == Some(-1));
            if has_neg {
                let known: Vec<usize> = new_shape.iter().copied().collect();
                let known_product: usize = known.iter().filter(|&&s| s < usize::MAX / 2).product();
                assert!(known_product > 0 && total % known_product == 0, "reshape: cannot infer dimension");
                known.iter().map(|&s| if s >= usize::MAX / 2 { total / known_product } else { s }).collect()
            } else {
                new_shape
            }
        };

        let shape_product: usize = final_shape.iter().product();
        assert_eq!(total, shape_product, "reshape: total size mismatch ({} vs {})", total, shape_product);

        let types = flat.iter().map(|v| v.zinnia_type()).collect();
        self.build_ndarray_from_flat(flat, types, &final_shape)
    }

    /// NDArray moveaxis: reorder axes by moving source axis to destination.
    fn ndarray_moveaxis(&mut self, val: &Value, args: &[Value]) -> Value {
        let shape = self.get_composite_shape(val);
        let ndim = shape.len();
        assert!(args.len() >= 2, "moveaxis: requires source and destination arguments");

        let src = {
            let s = args[0].int_val().expect("moveaxis: source must be constant int");
            if s < 0 { (ndim as i64 + s) as usize } else { s as usize }
        };
        let dst = {
            let d = args[1].int_val().expect("moveaxis: destination must be constant int");
            if d < 0 { (ndim as i64 + d) as usize } else { d as usize }
        };
        assert!(src < ndim && dst < ndim, "moveaxis: axis out of bounds");

        // Build permutation: remove src, insert at dst
        let mut order: Vec<usize> = (0..ndim).filter(|&i| i != src).collect();
        order.insert(dst, src);

        let axes_val: Vec<Value> = order.iter()
            .map(|&a| Value::Integer(crate::types::ScalarValue::new(Some(a as i64), None)))
            .collect();
        let axes_tuple = Value::Tuple(CompositeData {
            elements_type: vec![ZinniaType::Integer; order.len()],
            values: axes_val,
        });
        self.ndarray_transpose(val, &[axes_tuple])
    }

    /// NDArray repeat: repeat array elements along an axis.
    fn ndarray_repeat(&mut self, val: &Value, args: &[Value], kwargs: &std::collections::HashMap<String, Value>) -> Value {
        let repeats = args.first()
            .and_then(|v| v.int_val())
            .expect("repeat: repeats must be a constant integer");
        let axis = kwargs.get("axis")
            .or_else(|| args.get(1))
            .and_then(|v| v.int_val());

        if let Some(ax) = axis {
            // Repeat along specific axis
            let shape = self.get_composite_shape(val);
            let ndim = shape.len();
            let ax = if ax < 0 { (ndim as i64 + ax) as usize } else { ax as usize };

            if ax == 0 {
                if let Value::List(data) | Value::Tuple(data) = val {
                    let mut new_vals = Vec::new();
                    for v in &data.values {
                        for _ in 0..repeats {
                            new_vals.push(v.clone());
                        }
                    }
                    let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                    return Value::List(CompositeData { elements_type: types, values: new_vals });
                }
            }
            // For other axes, transpose so target axis is first, repeat, transpose back
            let mut fwd: Vec<usize> = (0..ndim).collect();
            fwd.swap(0, ax);
            let fwd_vals: Vec<Value> = fwd.iter().map(|&a| Value::Integer(crate::types::ScalarValue::new(Some(a as i64), None))).collect();
            let fwd_tuple = Value::Tuple(CompositeData { elements_type: vec![ZinniaType::Integer; ndim], values: fwd_vals });
            let transposed = self.ndarray_transpose(val, &[fwd_tuple.clone()]);
            let repeated = self.ndarray_repeat(&transposed, args, &std::collections::HashMap::new());
            self.ndarray_transpose(&repeated, &[fwd_tuple])
        } else {
            // No axis: flatten, then repeat each element
            let flat = self.flatten_composite(val);
            let mut new_vals = Vec::new();
            for v in &flat {
                for _ in 0..repeats {
                    new_vals.push(v.clone());
                }
            }
            let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: new_vals })
        }
    }

    /// NDArray filter: select elements where mask is true.
    fn ndarray_filter(&mut self, val: &Value, args: &[Value]) -> Value {
        let mask = args.first().expect("filter: requires a mask argument");
        let elements = self.flatten_composite(val);
        let mask_elements = self.flatten_composite(mask);
        assert_eq!(elements.len(), mask_elements.len(), "filter: array and mask must have same size");

        // For static arrays, we can build a filtered result at compile time
        // by using select chains. The result length depends on the mask values.
        // If mask values are all statically known, produce a fixed-size result.
        let mut static_result = Vec::new();
        let mut all_static = true;
        for (elem, m) in elements.iter().zip(mask_elements.iter()) {
            match m.int_val().or_else(|| if matches!(m, Value::Boolean(b) if b.static_val == Some(true)) { Some(1) } else if matches!(m, Value::Boolean(b) if b.static_val == Some(false)) { Some(0) } else { None }) {
                Some(v) if v != 0 => static_result.push(elem.clone()),
                Some(_) => {} // masked out
                None => { all_static = false; break; }
            }
        }

        if all_static {
            let types = static_result.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: static_result })
        } else {
            panic!("filter: dynamic masks require DynamicNDArray (not yet supported in Rust backend)");
        }
    }

    fn ndarray_argmax_argmin(&mut self, val: &Value, _args: &[Value], is_max: bool) -> Value {
        let elements = self.flatten_composite(val);
        if elements.is_empty() { return self.builder.ir_constant_int(0); }
        let mut best_idx = self.builder.ir_constant_int(0);
        let mut best_val = elements[0].clone();
        for (i, elem) in elements.iter().enumerate().skip(1) {
            let cond = if is_max {
                self.builder.ir_greater_than_i(elem, &best_val)
            } else {
                self.builder.ir_less_than_i(elem, &best_val)
            };
            let idx_val = self.builder.ir_constant_int(i as i64);
            best_idx = self.builder.ir_select_i(&cond, &idx_val, &best_idx);
            best_val = self.builder.ir_select_i(&cond, elem, &best_val);
        }
        best_idx
    }

    fn ndarray_argmax_argmin_with_axis(&mut self, val: &Value, axis: i64, is_max: bool) -> Value {
        if let Value::List(outer) | Value::Tuple(outer) = val {
            let ndim = self.get_composite_shape(val).len();
            let axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };

            if axis == 0 {
                // argmax along axis 0: for each column, find row with max/min
                if let Some(Value::List(first_row) | Value::Tuple(first_row)) = outer.values.first() {
                    let ncols = first_row.values.len();
                    let mut results = Vec::new();
                    for col in 0..ncols {
                        let mut best_idx = self.builder.ir_constant_int(0);
                        let mut best_val_opt: Option<Value> = None;
                        for (row_idx, row) in outer.values.iter().enumerate() {
                            if let Value::List(rd) | Value::Tuple(rd) = row {
                                if col < rd.values.len() {
                                    if let Some(ref best_val) = best_val_opt {
                                        let cond = if is_max {
                                            self.builder.ir_greater_than_i(&rd.values[col], best_val)
                                        } else {
                                            self.builder.ir_less_than_i(&rd.values[col], best_val)
                                        };
                                        let idx_val = self.builder.ir_constant_int(row_idx as i64);
                                        best_idx = self.builder.ir_select_i(&cond, &idx_val, &best_idx);
                                        best_val_opt = Some(self.builder.ir_select_i(&cond, &rd.values[col], best_val));
                                    } else {
                                        best_val_opt = Some(rd.values[col].clone());
                                    }
                                }
                            }
                        }
                        results.push(best_idx);
                    }
                    let types = vec![ZinniaType::Integer; results.len()];
                    return Value::List(CompositeData { elements_type: types, values: results });
                }
            } else if axis == 1 {
                // argmax along axis 1: for each row, find column index of max/min
                let mut results = Vec::new();
                for row in &outer.values {
                    results.push(self.ndarray_argmax_argmin(row, &[], is_max));
                }
                let types = vec![ZinniaType::Integer; results.len()];
                return Value::List(CompositeData { elements_type: types, values: results });
            }
        }
        self.ndarray_argmax_argmin(val, &[], is_max)
    }

    // ── List method helpers ───────────────────────────────────────────

    fn list_method_append(&mut self, var: &str, args: &[Value]) -> Value {
        if let Some(new_elem) = args.first() {
            if let Some(Value::List(mut data)) = self.ctx.get(var) {
                data.elements_type.push(new_elem.zinnia_type());
                data.values.push(new_elem.clone());
                self.ctx.set(var, Value::List(data));
            }
        }
        Value::None
    }

    fn list_method_extend(&mut self, var: &str, args: &[Value]) -> Value {
        if let Some(other) = args.first() {
            if let (Some(Value::List(mut data)), Value::List(ext) | Value::Tuple(ext)) =
                (self.ctx.get(var), other)
            {
                data.elements_type.extend(ext.elements_type.clone());
                data.values.extend(ext.values.clone());
                self.ctx.set(var, Value::List(data));
            }
        }
        Value::None
    }

    fn list_method_pop(&mut self, var: &str, args: &[Value]) -> Value {
        // Try to get static index from args
        let idx_val = args.first();
        // Only use static path if: (a) no args (default -1), or (b) arg is compile-time known
        let static_idx = match idx_val {
            None => Some(-1i64),  // default: pop last
            Some(v) => v.int_val(),  // None if dynamic
        };

        if let Some(idx) = static_idx {
            if let Some(Value::List(mut data)) = self.ctx.get(var) {
                let len = data.values.len() as i64;
                let i = if idx < 0 { len + idx } else { idx };
                if i < 0 || i >= len {
                    panic!("pop index out of range");
                }
                let i = i as usize;
                let removed = data.values.remove(i);
                data.elements_type.remove(i);
                self.ctx.set(var, Value::List(data));
                return removed;
            }
        }
        // Dynamic index case: select the popped value and rebuild list without it
        if let Some(idx_v) = idx_val {
            if let Some(Value::List(data)) = self.ctx.get(var) {
                let n = data.values.len();
                let len_const = self.builder.ir_constant_int(n as i64);
                let neg_len = self.builder.ir_constant_int(-(n as i64));

                // Normalize negative index: idx = idx < 0 ? idx + len : idx
                let zero = self.builder.ir_constant_int(0);
                let is_neg = self.builder.ir_less_than_i(idx_v, &zero);
                let normalized = self.builder.ir_add_i(idx_v, &len_const);
                let idx_norm = self.builder.ir_select_i(&is_neg, &normalized, idx_v);

                // Assert 0 <= idx_norm < len
                let ge_zero = self.builder.ir_greater_than_or_equal_i(&idx_norm, &zero);
                let lt_len = self.builder.ir_less_than_i(&idx_norm, &len_const);
                let in_bounds = self.builder.ir_logical_and(&ge_zero, &lt_len);
                self.builder.ir_assert(&in_bounds);

                let popped = self.dynamic_list_subscript(&data, &idx_norm);
                if n > 0 {
                    let mut past_idx = self.builder.ir_constant_bool(false);
                    let mut new_values = Vec::new();
                    let mut new_types = Vec::new();
                    for i in 0..n - 1 {
                        let i_const = self.builder.ir_constant_int(i as i64);
                        let is_idx = self.builder.ir_equal_i(&idx_norm, &i_const);
                        past_idx = self.builder.ir_logical_or(&past_idx, &is_idx);
                        // If past the popped index, take data[i+1], else data[i]
                        let shifted = self.builder.ir_select_i(&past_idx, &data.values[i + 1], &data.values[i]);
                        new_values.push(shifted);
                        new_types.push(data.elements_type[i].clone());
                    }
                    let new_list = Value::List(CompositeData { elements_type: new_types, values: new_values });
                    self.ctx.set(var, new_list);
                }
                return popped;
            }
        }
        Value::None
    }

    fn list_method_remove(&mut self, var: &str, args: &[Value]) -> Value {
        if let Some(target) = args.first() {
            if let Some(Value::List(data)) = self.ctx.get(var) {
                // Try static removal first
                let target_int = target.int_val();
                if let Some(target_val) = target_int {
                    let mut new_data = data.clone();
                    // Check if all elements have known values (static list)
                    let all_known = new_data.values.iter().all(|v| v.int_val().is_some());
                    if let Some(pos) = new_data.values.iter().position(|v| v.int_val() == Some(target_val)) {
                        new_data.values.remove(pos);
                        new_data.elements_type.remove(pos);
                        self.ctx.set(var, Value::List(new_data));
                        return Value::None;
                    } else if all_known {
                        // Static value not found in a fully known list
                        panic!("Value not found in list");
                    }
                }
                // Dynamic: generate a new list with the first matching element removed
                // Strategy: for each position, if we haven't removed yet and this matches,
                // skip it (use next element). Otherwise, keep current or shifted element.
                let n = data.values.len();
                let mut found = self.builder.ir_constant_bool(false);
                let mut new_values = Vec::new();
                let mut new_types = Vec::new();

                // Build shifted list: for each output position i,
                // if found_before_i: take data[i+1], else take data[i]
                for i in 0..n - 1 {
                    let eq = self.builder.ir_equal_i(&data.values[i], target);
                    let not_found = self.builder.ir_logical_not(&found);
                    let is_removal = self.builder.ir_logical_and(&eq, &not_found);
                    found = self.builder.ir_logical_or(&found, &is_removal);
                    // After this point, 'found' means we've removed an element at or before i
                    // If found, take data[i+1], else take data[i]
                    let shifted = self.builder.ir_select_i(&found, &data.values[i + 1], &data.values[i]);
                    new_values.push(shifted);
                    new_types.push(data.elements_type[i].clone());
                }
                // For the last element: if not found yet, check it too
                let eq_last = self.builder.ir_equal_i(&data.values[n - 1], target);
                let not_found_last = self.builder.ir_logical_not(&found);
                let is_removal_last = self.builder.ir_logical_and(&eq_last, &not_found_last);
                found = self.builder.ir_logical_or(&found, &is_removal_last);

                // Assert that the value was found
                self.builder.ir_assert(&found);

                // Update the list variable with the shorter list
                let new_list = Value::List(CompositeData { elements_type: new_types, values: new_values });
                self.ctx.set(var, new_list);
            }
        }
        Value::None
    }

    fn list_method_clear(&mut self, var: &str) -> Value {
        self.ctx.set(var, Value::List(CompositeData { elements_type: vec![], values: vec![] }));
        Value::None
    }

    fn list_method_reverse(&mut self, var: &str) -> Value {
        if let Some(Value::List(mut data)) = self.ctx.get(var) {
            data.values.reverse();
            data.elements_type.reverse();
            self.ctx.set(var, Value::List(data));
        }
        Value::None
    }

    fn list_method_insert(&mut self, var: &str, args: &[Value]) -> Value {
        assert!(args.len() >= 2, "list.insert requires index and object arguments");
        let idx_arg = &args[0];
        let new_elem = &args[1];

        if let Some(Value::List(mut data)) = self.ctx.get(var) {
            if let Some(idx) = idx_arg.int_val() {
                // Static index: insert at known position
                let len = data.values.len() as i64;
                let i = if idx < 0 {
                    (len + idx).max(0) as usize
                } else {
                    (idx as usize).min(data.values.len())
                };
                data.elements_type.insert(i, new_elem.zinnia_type());
                data.values.insert(i, new_elem.clone());
                self.ctx.set(var, Value::List(data));
            } else {
                // Dynamic index: build a select chain over all possible positions.
                // All elements must be the same type for the select chain to work.
                let len = data.values.len();
                // Build list with element inserted at each possible position 0..=len
                // and select the correct one based on the index value.
                let mut current_data = data.clone();
                // Start with insert at position `len` (append)
                current_data.elements_type.push(new_elem.zinnia_type());
                current_data.values.push(new_elem.clone());

                // For each position i from len-1 down to 0, conditionally swap
                for i in (0..len).rev() {
                    let i_const = self.builder.ir_constant_int(i as i64);
                    let should_insert_here = self.builder.ir_equal_i(idx_arg, &i_const);
                    // If inserting at i, shift elements right from position i
                    // Use select: for positions >= i, pick element from position-1 in original
                    let mut new_vals = current_data.values.clone();
                    // Swap: move new_elem to position i and shift others right
                    for j in (i + 1..=len).rev() {
                        new_vals[j] = self.builder.ir_select_i(&should_insert_here, &current_data.values[j - 1], &current_data.values[j]);
                    }
                    new_vals[i] = self.builder.ir_select_i(&should_insert_here, new_elem, &current_data.values[i]);
                    current_data.values = new_vals;
                }
                let types = current_data.values.iter().map(|v| v.zinnia_type()).collect();
                current_data.elements_type = types;
                self.ctx.set(var, Value::List(current_data));
            }
        }
        Value::None
    }

    fn list_method_count(&mut self, var: &str, args: &[Value]) -> Value {
        if let (Some(target), Some(Value::List(data) | Value::Tuple(data))) = (args.first(), self.ctx.get(var)) {
            // Generate dynamic count: sum(1 if elem == target else 0 for elem in list)
            let mut count = self.builder.ir_constant_int(0);
            for elem in &data.values {
                let eq = self.builder.ir_equal_i(elem, target);
                // Cast bool to int and add
                let one = self.builder.ir_constant_int(1);
                let zero = self.builder.ir_constant_int(0);
                let inc = self.builder.ir_select_i(&eq, &one, &zero);
                count = self.builder.ir_add_i(&count, &inc);
            }
            count
        } else {
            self.builder.ir_constant_int(0)
        }
    }

    fn list_method_index(&mut self, var: &str, args: &[Value]) -> Value {
        if let (Some(target), Some(Value::List(data) | Value::Tuple(data))) = (args.first(), self.ctx.get(var)) {
            let start = args.get(1).and_then(|a| a.int_val()).unwrap_or(0) as usize;
            // Generate dynamic index: find first match after start
            let mut found = self.builder.ir_constant_bool(false);
            let mut answer = self.builder.ir_constant_int(0);
            for (i, elem) in data.values.iter().enumerate() {
                if i < start { continue; }
                let eq = self.builder.ir_equal_i(elem, target);
                let not_found = self.builder.ir_logical_not(&found);
                let first_match = self.builder.ir_logical_and(&eq, &not_found);
                let idx_const = self.builder.ir_constant_int(i as i64);
                answer = self.builder.ir_select_i(&first_match, &idx_const, &answer);
                found = self.builder.ir_logical_or(&found, &eq);
            }
            // Assert found (the element should be in the list for valid circuits)
            self.builder.ir_assert(&found);
            answer
        } else {
            self.builder.ir_constant_int(-1)
        }
    }

    // ── Chip and external function calls ──────────────────────────────

    fn visit_chip_call(&mut self, name: &str, args: &[Value], _kwargs: &HashMap<String, Value>) -> Value {
        let chip = self.registered_chips.get(name).cloned();
        let chip = match chip {
            Some(c) => c,
            None => return Value::None,
        };

        // Check recursion limit
        if self.recursion_depth >= self.config.recursion_limit {
            // Return a placeholder value
            let return_dt = self.parse_dt_descriptor(&chip.return_dt);
            return match return_dt {
                ZinniaType::Integer | ZinniaType::Boolean => self.builder.ir_constant_int(0),
                ZinniaType::Float => self.builder.ir_constant_float(0.0),
                _ => Value::None,
            };
        }

        // Parse chip AST
        let chip_ast: ASTNode = match serde_json::from_value(chip.chip_ast.clone()) {
            Ok(node) => node,
            Err(_) => return Value::None,
        };
        let chip_node = match &chip_ast {
            ASTNode::ASTChip(c) => c,
            _ => return Value::None,
        };

        // Enter chip scope
        let return_dt = self.parse_dt_descriptor(&chip.return_dt);
        self.ctx.chip_enter(return_dt, None);
        self.recursion_depth += 1;

        // Bind arguments
        for (i, inp) in chip_node.inputs.iter().enumerate() {
            if i < args.len() {
                self.ctx.set(&inp.name, args[i].clone());
            }
        }

        self.register_global_datatypes();

        // Visit chip body
        for stmt in &chip_node.block {
            self.visit(stmt);
        }

        // Check if return is guaranteed for non-None return types
        let return_dt_check = self.parse_dt_descriptor(&chip.return_dt);
        let return_guaranteed = self.ctx.check_return_guaranteed();

        // Collect return value
        // Collect returns BEFORE leaving chip scope
        let returns = self.ctx.get_returns_with_conditions();

        // Check return guarantee: error if chip has non-None return type
        // and no return statement was encountered on any path
        if !matches!(return_dt_check, ZinniaType::None) && returns.is_empty() {
            panic!("Chip control ends without a return statement");
        }

        self.ctx.chip_leave();
        self.recursion_depth -= 1;

        // Merge return values using conditional select
        if returns.is_empty() {
            return Value::None;
        }
        let mut result = returns[0].0.clone();
        for i in 1..returns.len() {
            let (val, cond) = &returns[i];
            result = self.select_value(cond, val, &result);
        }
        result
    }

    fn visit_external_call(&mut self, name: &str, args: &[Value]) -> Value {
        let ext = self.registered_externals.get(name).cloned();
        let ext = match ext {
            Some(e) => e,
            None => return Value::None,
        };

        let return_dt = self.parse_dt_descriptor(&ext.return_dt);

        // Build arg type descriptors for InvokeExternal
        let arg_dts: Vec<serde_json::Value> = args.iter().map(|a| {
            match a.zinnia_type() {
                ZinniaType::Integer | ZinniaType::Boolean => serde_json::json!({"__class__": "IntegerDTDescriptor", "dt_data": {}}),
                ZinniaType::Float => serde_json::json!({"__class__": "FloatDTDescriptor", "dt_data": {}}),
                _ => serde_json::json!({"__class__": "IntegerDTDescriptor", "dt_data": {}}),
            }
        }).collect();

        // Export each argument
        let store_idx = 0u32; // Simple store index
        for (i, arg) in args.iter().enumerate() {
            let flat = self.flatten_composite(arg);
            for (j, v) in flat.iter().enumerate() {
                let key = crate::ir_defs::ExternalKey::Int(j as u32);
                match v {
                    Value::Float(_) => {
                        self.builder.create_ir(
                            &crate::ir_defs::IR::ExportExternalF {
                                for_which: store_idx,
                                key,
                                indices: vec![i as u32],
                            },
                            &[v.clone()],
                        );
                    }
                    _ => {
                        self.builder.create_ir(
                            &crate::ir_defs::IR::ExportExternalI {
                                for_which: store_idx,
                                key,
                                indices: vec![i as u32],
                            },
                            &[v.clone()],
                        );
                    }
                }
            }
        }

        // Invoke the external function
        let invoke_ir = crate::ir_defs::IR::InvokeExternal {
            store_idx,
            func_name: name.to_string(),
            args: arg_dts,
            kwargs: std::collections::HashMap::new(),
        };
        let result = self.builder.create_ir(&invoke_ir, &[]);
        match return_dt {
            ZinniaType::Integer | ZinniaType::Boolean => result,
            ZinniaType::Float => result,
            _ => result,
        }
    }

    /// Cast all elements in a composite to int or float.
    fn cast_composite(&mut self, val: &Value, to_float: bool) -> Value {
        match val {
            Value::List(data) | Value::Tuple(data) => {
                let is_tuple = matches!(val, Value::Tuple(_));
                let new_values: Vec<Value> = data.values.iter()
                    .map(|v| self.cast_composite(v, to_float))
                    .collect();
                let new_types = new_values.iter().map(|v| v.zinnia_type()).collect();
                if is_tuple {
                    Value::Tuple(CompositeData { elements_type: new_types, values: new_values })
                } else {
                    Value::List(CompositeData { elements_type: new_types, values: new_values })
                }
            }
            Value::Integer(_) | Value::Boolean(_) if to_float => {
                self.builder.ir_float_cast(val)
            }
            Value::Float(_) if !to_float => {
                self.builder.ir_int_cast(val)
            }
            _ => val.clone(),
        }
    }

    /// Set a nested value in a composite structure using slice indices.
    /// Cast a value to match the target element dtype if they differ.
    fn cast_value_to_match(&mut self, value: Value, target_type: &ZinniaType) -> Value {
        let vt = value.zinnia_type();
        if vt == *target_type { return value; }
        match (target_type, &value) {
            (ZinniaType::Integer, Value::Float(_)) => self.builder.ir_int_cast(&value),
            (ZinniaType::Float, Value::Integer(_)) => self.builder.ir_float_cast(&value),
            (ZinniaType::Float, Value::Boolean(_)) => self.builder.ir_float_cast(&value),
            (ZinniaType::Integer, Value::Boolean(_)) => self.builder.ir_bool_cast(&value),
            _ => value,
        }
    }

    fn set_nested_value(&mut self, current: Value, indices: &[SliceIndex], value: Value) -> Value {
        if indices.is_empty() {
            // At leaf: cast value to match current's type if they differ
            return self.cast_value_to_match(value, &current.zinnia_type());
        }
        match &current {
            Value::List(data) | Value::Tuple(data) => {
                let is_tuple = matches!(&current, Value::Tuple(_));
                if let SliceIndex::Single(idx_val) = &indices[0] {
                    if let Some(idx) = idx_val.int_val() {
                        // Static index
                        let idx = if idx < 0 { (data.values.len() as i64 + idx) as usize } else { idx as usize };
                        if idx < data.values.len() {
                            let mut new_values = data.values.clone();
                            let mut new_types = data.elements_type.clone();
                            if indices.len() == 1 {
                                // Cast value to match target element type
                                let target_et = &data.elements_type[idx];
                                new_values[idx] = self.cast_value_to_match(value, target_et);
                                new_types[idx] = new_values[idx].zinnia_type();
                            } else {
                                new_values[idx] = self.set_nested_value(new_values[idx].clone(), &indices[1..], value);
                                new_types[idx] = new_values[idx].zinnia_type();
                            }
                            return if is_tuple {
                                Value::Tuple(CompositeData { elements_type: new_types, values: new_values })
                            } else {
                                Value::List(CompositeData { elements_type: new_types, values: new_values })
                            };
                        }
                    } else {
                        // Dynamic index — use mux/memory path
                        if indices.len() == 1 {
                            // Single dynamic index on flat list
                            return self.dynamic_list_set_item(data, idx_val, &value);
                        } else {
                            // Multi-dim dynamic index: compute linear address
                            // For array[x, y] where array is [[...], [...], ...]:
                            // Flatten array, compute addr = x * ncols + y, set at addr
                            let shape = self.get_composite_shape(&current);
                            let flat = self.flatten_composite(&current);
                            if flat.is_empty() { return current; }

                            // Compute linear address from multi-dim indices
                            let mut strides = vec![1usize; shape.len()];
                            for i in (0..shape.len() - 1).rev() {
                                strides[i] = strides[i + 1] * shape[i + 1];
                            }

                            let mut linear_addr = self.builder.ir_constant_int(0);
                            // Process all indices (current + remaining)
                            let all_idx_vals: Vec<&Value> = std::iter::once(idx_val)
                                .chain(indices[1..].iter().filter_map(|si| {
                                    if let SliceIndex::Single(v) = si { Some(v) } else { None }
                                }))
                                .collect();

                            for (dim, &iv) in all_idx_vals.iter().enumerate() {
                                if dim < strides.len() {
                                    let stride_const = self.builder.ir_constant_int(strides[dim] as i64);
                                    let term = self.builder.ir_mul_i(iv, &stride_const);
                                    linear_addr = self.builder.ir_add_i(&linear_addr, &term);
                                }
                            }

                            let flat_data = CompositeData {
                                elements_type: flat.iter().map(|v| v.zinnia_type()).collect(),
                                values: flat,
                            };
                            let updated_flat = self.dynamic_list_set_item(&flat_data, &linear_addr, &value);

                            // Rebuild nested structure from flat
                            if let Value::List(uf) = &updated_flat {
                                let rebuilt = self.build_nested_value(uf.values.clone(), uf.elements_type.clone(), &shape);
                                return rebuilt;
                            }
                            return updated_flat;
                        }
                    }
                }
                // For range slicing assignment
                if let SliceIndex::Range(start, stop, step) = &indices[0] {
                    let len = data.values.len() as i64;
                    let start_idx = start.as_ref().and_then(|v| v.int_val()).unwrap_or(0);
                    let stop_idx = stop.as_ref().and_then(|v| v.int_val()).unwrap_or(len);
                    let step_val = step.as_ref().and_then(|v| v.int_val()).unwrap_or(1);
                    let start_idx = if start_idx < 0 { (len + start_idx).max(0) } else { start_idx.min(len) } as usize;
                    let stop_idx = if stop_idx < 0 { (len + stop_idx).max(0) } else { stop_idx.min(len) } as usize;

                    let mut new_values = data.values.clone();
                    let mut new_types = data.elements_type.clone();
                    if indices.len() > 1 {
                        // Multi-dim range assignment: array[:, col] = values
                        // Apply remaining indices to each selected element
                        if let Value::List(rhs_data) | Value::Tuple(rhs_data) = &value {
                            let mut rhs_idx = 0;
                            let mut i = start_idx;
                            while i < stop_idx && rhs_idx < rhs_data.values.len() {
                                new_values[i] = self.set_nested_value(
                                    new_values[i].clone(),
                                    &indices[1..],
                                    rhs_data.values[rhs_idx].clone(),
                                );
                                new_types[i] = new_values[i].zinnia_type();
                                rhs_idx += 1;
                                i += step_val as usize;
                            }
                        }
                    } else {
                        // Single-dim range assignment: array[start:stop] = values or scalar
                        if let Value::List(rhs_data) | Value::Tuple(rhs_data) = &value {
                            let mut rhs_idx = 0;
                            let mut i = start_idx;
                            while i < stop_idx && rhs_idx < rhs_data.values.len() {
                                let target_et = &data.elements_type[i];
                                new_values[i] = self.cast_value_to_match(rhs_data.values[rhs_idx].clone(), target_et);
                                new_types[i] = new_values[i].zinnia_type();
                                rhs_idx += 1;
                                i += step_val as usize;
                            }
                        } else {
                            // Scalar broadcasting: assign scalar to all positions in range
                            let mut i = start_idx;
                            while i < stop_idx {
                                let target_et = &data.elements_type[i];
                                new_values[i] = self.cast_value_to_match(value.clone(), target_et);
                                new_types[i] = new_values[i].zinnia_type();
                                i += step_val as usize;
                            }
                        }
                    }
                    return if is_tuple {
                        Value::Tuple(CompositeData { elements_type: new_types, values: new_values })
                    } else {
                        Value::List(CompositeData { elements_type: new_types, values: new_values })
                    };
                }
                current
            }
            _ => current,
        }
    }

    fn ndarray_shape(&self, val: &Value) -> Value {
        // Return the shape as a tuple of constants
        match val {
            Value::List(data) => {
                // For a list, shape is (len,)
                let len_val = Value::Integer(crate::types::ScalarValue::new(Some(data.values.len() as i64), None));
                Value::Tuple(CompositeData {
                    elements_type: vec![ZinniaType::Integer],
                    values: vec![len_val],
                })
            }
            Value::NDArray(nd) => {
                let shape_vals: Vec<Value> = nd.shape.iter()
                    .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                    .collect();
                let types = shape_vals.iter().map(|_| ZinniaType::Integer).collect();
                Value::Tuple(CompositeData {
                    elements_type: types,
                    values: shape_vals,
                })
            }
            _ => Value::None,
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────

    fn read_input_value(&mut self, dt: &ZinniaType, indices: Vec<u32>, is_public: bool) -> Value {
        match dt {
            ZinniaType::Integer | ZinniaType::Boolean => {
                self.builder.ir_read_integer(indices, is_public)
            }
            ZinniaType::Float => {
                self.builder.ir_read_float(indices, is_public)
            }
            ZinniaType::PoseidonHashed { .. } => {
                self.builder.ir_read_hash(indices, is_public)
            }
            ZinniaType::NDArray { shape, dtype } => {
                let total: usize = shape.iter().product();
                let inner_dt = match dtype {
                    crate::types::NumberType::Integer => ZinniaType::Integer,
                    crate::types::NumberType::Float => ZinniaType::Float,
                };
                let mut values = Vec::new();
                for flat_idx in 0..total {
                    let mut sub_indices = indices.clone();
                    sub_indices.push(flat_idx as u32);
                    values.push(self.read_input_value(&inner_dt, sub_indices, is_public));
                }
                // Build nested structure from flat values
                let types = values.iter().map(|v| v.zinnia_type()).collect();
                self.build_nested_value(values, types, shape)
            }
            ZinniaType::List { elements } => {
                let mut values = Vec::new();
                let mut types = Vec::new();
                for (j, elem_dt) in elements.iter().enumerate() {
                    let mut sub_indices = indices.clone();
                    sub_indices.push(j as u32);
                    let val = self.read_input_value(elem_dt, sub_indices, is_public);
                    types.push(val.zinnia_type());
                    values.push(val);
                }
                Value::List(CompositeData { elements_type: types, values })
            }
            ZinniaType::Tuple { elements } => {
                let mut values = Vec::new();
                let mut types = Vec::new();
                for (j, elem_dt) in elements.iter().enumerate() {
                    let mut sub_indices = indices.clone();
                    sub_indices.push(j as u32);
                    let val = self.read_input_value(elem_dt, sub_indices, is_public);
                    types.push(val.zinnia_type());
                    values.push(val);
                }
                Value::Tuple(CompositeData { elements_type: types, values })
            }
            ZinniaType::DynamicNDArray { dtype, max_length, max_rank } => {
                let inner_dt = match dtype {
                    crate::types::NumberType::Integer => ZinniaType::Integer,
                    crate::types::NumberType::Float => ZinniaType::Float,
                };
                // Read flat payload elements
                let mut elements = Vec::new();
                for flat_idx in 0..*max_length {
                    let mut sub_indices = indices.clone();
                    sub_indices.push(flat_idx as u32);
                    let val = self.read_input_value(&inner_dt, sub_indices, is_public);
                    elements.push(crate::dyn_ndarray::value_to_scalar_i64(&val));
                }

                // Emit metadata allocation
                let arr_id = self.alloc_array_id();
                let dtype_name = match dtype {
                    crate::types::NumberType::Integer => "int".to_string(),
                    crate::types::NumberType::Float => "float".to_string(),
                };
                self.builder.ir_allocate_dynamic_ndarray_meta(
                    arr_id, dtype_name, *max_length as u32, *max_rank as u32,
                );

                let strides = crate::dyn_ndarray::dyn_row_major_strides(&[*max_length]);
                Value::DynamicNDArray(crate::types::DynamicNDArrayData {
                    max_length: *max_length,
                    max_rank: *max_rank,
                    dtype: *dtype,
                    elements,
                    meta: crate::types::DynArrayMeta {
                        logical_shape: vec![*max_length],
                        logical_offset: 0,
                        logical_strides: strides,
                        runtime_length: crate::types::ScalarValue::new(None, None),
                        runtime_rank: crate::types::ScalarValue::new(None, None),
                        runtime_shape: (0..*max_rank)
                            .map(|_| crate::types::ScalarValue::new(None, None))
                            .collect(),
                        runtime_strides: (0..*max_rank)
                            .map(|_| crate::types::ScalarValue::new(None, None))
                            .collect(),
                        runtime_offset: crate::types::ScalarValue::new(Some(0), None),
                    },
                })
            }
            _ => {
                self.builder.ir_read_integer(indices, is_public)
            }
        }
    }

    fn build_nested_value(&self, flat: Vec<Value>, flat_types: Vec<ZinniaType>, shape: &[usize]) -> Value {
        if shape.len() <= 1 {
            return Value::List(CompositeData { elements_type: flat_types, values: flat });
        }
        let inner_size: usize = shape[1..].iter().product();
        let mut rows = Vec::new();
        for chunk in flat.chunks(inner_size) {
            let chunk_types = chunk.iter().map(|v| v.zinnia_type()).collect();
            rows.push(self.build_nested_value(chunk.to_vec(), chunk_types, &shape[1..]));
        }
        let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: row_types, values: rows })
    }

    fn register_global_datatypes(&mut self) {
        // Register Float and Integer as class values
        let float_class = Value::Class(ZinniaType::Float);
        let int_class = Value::Class(ZinniaType::Integer);
        for name in &["Float", "float"] {
            self.ctx.set(name, float_class.clone());
        }
        for name in &["Integer", "int", "Int", "integer", "Boolean", "bool", "Bool", "boolean"] {
            self.ctx.set(name, int_class.clone());
        }
    }

    fn parse_dt_descriptor(&self, dt_json: &serde_json::Value) -> ZinniaType {
        // Try full DTDescriptorDict format: {"__class__": "...", "dt_data": {...}}
        if let Ok(dict) = serde_json::from_value::<DTDescriptorDict>(dt_json.clone()) {
            return ZinniaType::from_dt_dict(&dict).unwrap_or(ZinniaType::Integer);
        }
        // Fallback: bare dt_data without class wrapper (old format)
        ZinniaType::Integer
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_simple_constants() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                    "value": {"__class__": "ASTConstantInteger", "value": 42}
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        assert!(!graph.is_empty()); // At least the constant + type registrations
    }

    #[test]
    fn test_generate_binary_op() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "z"}],
                    "value": {
                        "__class__": "ASTBinaryOperator",
                        "operator": "add",
                        "lhs": {"__class__": "ASTConstantInteger", "value": 10},
                        "rhs": {"__class__": "ASTConstantInteger", "value": 20}
                    }
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        // Should have: constant(10), constant(20), add_i
        assert!(graph.len() >= 3);
    }

    #[test]
    fn test_generate_if_else() {
        let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "cond"}],
                    "value": {"__class__": "ASTConstantBoolean", "value": true}
                },
                {
                    "__class__": "ASTCondStatement",
                    "cond": {"__class__": "ASTLoad", "name": "cond"},
                    "t_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 1}
                        }
                    ],
                    "f_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 2}
                        }
                    ]
                }
            ],
            "inputs": []
        }"#;

        let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
        assert!(graph.len() >= 3);
    }
}
