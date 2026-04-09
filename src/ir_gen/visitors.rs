use crate::ast::*;
use crate::types::{CompositeData, Value, ZinniaType};

use super::{is_starred_target, IRGenerator, SliceIndex};

impl IRGenerator {
    pub(crate) fn visit_assign(&mut self, n: &ASTAssignStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let val = self.visit(&n.value);
        for target in &n.targets {
            self.do_recursive_assign(target, val.clone(), true);
        }
    }

    pub(crate) fn visit_aug_assign(&mut self, n: &ASTAugAssignStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        // AugAssign on subscript — delegate to builder
        let _val = self.visit(&n.value);
        // TODO: full aug assign implementation
    }

    pub(crate) fn visit_cond(&mut self, n: &ASTCondStatement) {
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
        let true_cond = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &cond_val);
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

    pub(crate) fn visit_for_in(&mut self, n: &ASTForInStatement) {
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

    pub(crate) fn visit_while(&mut self, n: &ASTWhileStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }

        let mut loop_quota = self.config.loop_limit + 1;
        self.ctx.loop_enter();

        loop {
            self.ctx.loop_reiter(&mut self.builder);
            let test = self.visit(&n.test_expr);
            let test_bool = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &test);
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

    pub(crate) fn visit_break(&mut self) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        self.ctx.loop_break(None, &mut self.builder);
        self.ctx.set_terminated_guarantee();
    }

    pub(crate) fn visit_continue(&mut self) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        self.ctx.loop_continue(&mut self.builder);
    }

    pub(crate) fn visit_return(&mut self, n: &ASTReturnStatement) {
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

    pub(crate) fn visit_assert(&mut self, n: &ASTAssertStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let test = self.visit(&n.expr);
        self.assert_value(&test);
    }

    /// Assert a value. For composites, reduces to scalar bool via AND, then asserts.
    /// The assert is conditioned on the current path condition.
    pub(crate) fn assert_value(&mut self, val: &Value) {
        let scalar = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, val);
        let cond = self.ctx.get_condition_value(&mut self.builder);
        let true_val = self.builder.ir_constant_bool(true);
        let conditioned = self.builder.ir_select_i(&cond, &scalar, &true_val);
        self.builder.ir_assert(&conditioned);
    }

    // ── Expressions ───────────────────────────────────────────────────

    pub(crate) fn visit_binary_op(&mut self, n: &ASTBinaryOperator) -> Value {
        let lhs = self.visit(&n.lhs);
        let rhs = self.visit(&n.rhs);
        crate::helpers::value_ops::apply_binary_op(&mut self.builder, n.operator.as_str(), &lhs, &rhs)
    }

    pub(crate) fn visit_unary_op(&mut self, n: &ASTUnaryOperator) -> Value {
        let operand = self.visit(&n.operand);
        self.apply_unary_op(n.operator.as_str(), &operand)
    }

    /// Apply a unary operation, with element-wise support for composite types.
    pub(crate) fn apply_unary_op(&mut self, op: &str, operand: &Value) -> Value {
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
                    if matches!(operand, Value::Float(_)) {
                        let zero = self.builder.ir_constant_float(0.0);
                        self.builder.ir_sub_f(&zero, operand)
                    } else {
                        let zero = self.builder.ir_constant_int(0);
                        self.builder.ir_sub_i(&zero, operand)
                    }
                }
                "uadd" => operand.clone(),
                _ => panic!("Unknown unary operator: {}", op),
            }
        }
    }

    pub(crate) fn visit_load(&mut self, n: &ASTLoad) -> Value {
        self.ctx
            .get(&n.name)
            .unwrap_or_else(|| panic!("Variable `{}` not found", n.name))
    }

    pub(crate) fn visit_subscript(&mut self, n: &ASTSubscriptExp) -> Value {
        let val = self.visit(&n.val);
        // Evaluate slice indices by visiting them as AST nodes
        let slice_values = self.eval_slice_indices(&n.slicing);

        match &val {
            Value::List(data) | Value::Tuple(data) => {
                // Advanced indexing: `array[bool_mask]` or `array[idx_array]`
                // where the index itself is a static-shape composite. Try to
                // resolve at compile time; on shape/typing errors return Err
                // which we surface as a hard error to the user.
                if slice_values.len() == 1 {
                    if let SliceIndex::Single(idx_value) = &slice_values[0] {
                        if matches!(idx_value, Value::List(_) | Value::Tuple(_)) {
                            match crate::helpers::ndarray::try_advanced_index_static(data, idx_value) {
                                Ok(Some(result)) => return result,
                                Ok(Option::None) => {} // not advanced indexing — fall through
                                Err(msg) => panic!("{}", msg),
                            }
                        }
                    }
                }

                // Single Ellipsis (`array[...]`) — selects the whole array.
                // Single NewAxis (`array[None]`) — wraps in a length-1 outer axis.
                // Both fall through to the multi-dim handler so the same code
                // does the work in one place.
                if slice_values.len() == 1
                    && matches!(slice_values[0], SliceIndex::Ellipsis | SliceIndex::NewAxis)
                {
                    return crate::helpers::ndarray::multidim_subscript(&mut self.builder, data, &slice_values);
                }
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
                                crate::helpers::value_ops::dynamic_list_subscript(&mut self.builder, data, idx_value)
                            }
                        }
                        SliceIndex::Range(_, _, _) => {
                            // Range slicing on list/tuple
                            self.list_slice_range(&val, data, idx_val)
                        }
                        SliceIndex::Ellipsis | SliceIndex::NewAxis => {
                            // Already handled by the early-return above; this
                            // arm exists only to satisfy match exhaustiveness.
                            unreachable!()
                        }
                    }
                } else {
                    // Multi-dimensional ndarray-style indexing
                    crate::helpers::ndarray::multidim_subscript(&mut self.builder, data, &slice_values)
                }
            }
            _ => Value::None,
        }
    }

    /// Evaluate slice indices by visiting AST nodes (not just extracting constants).
    pub(crate) fn eval_slice_indices(&mut self, slice: &ASTSlice) -> Vec<SliceIndex> {
        let mut indices = Vec::new();
        for d in &slice.data {
            // Sentinel objects emitted by visit_slice_key for `...` and
            // `np.newaxis` / `None`. We check by `__class__` field before the
            // generic ASTNode parse so they are never confused with a real
            // expression.
            if let Some(cls) = d.get("__class__").and_then(|v| v.as_str()) {
                if cls == "ASTSliceEllipsis" {
                    indices.push(SliceIndex::Ellipsis);
                    continue;
                }
                if cls == "ASTSliceNewAxis" {
                    indices.push(SliceIndex::NewAxis);
                    continue;
                }
            }
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

    /// Handle range slicing on lists/tuples.
    pub(crate) fn list_slice_range(&mut self, _val: &Value, data: &CompositeData, slice: &SliceIndex) -> Value {
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

    pub(crate) fn visit_square_brackets(&mut self, n: &ASTSquareBrackets) -> Value {
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

    pub(crate) fn visit_parenthesis(&mut self, n: &ASTParenthesis) -> Value {
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

    pub(crate) fn visit_generator_exp(&mut self, n: &ASTGeneratorExp) -> Value {
        if n.generators.is_empty() {
            return Value::List(CompositeData { elements_type: vec![], values: vec![] });
        }

        let mut result_values = Vec::new();
        self.expand_generators(&n.generators, 0, &n.elt, &mut result_values);

        let types: Vec<ZinniaType> = result_values.iter().map(|v| v.zinnia_type()).collect();
        if n.kind == "list" {
            Value::List(CompositeData { elements_type: types, values: result_values })
        } else {
            Value::Tuple(CompositeData { elements_type: types, values: result_values })
        }
    }

    fn expand_generators(
        &mut self,
        generators: &[ASTGenerator],
        idx: usize,
        elt: &ASTNode,
        result: &mut Vec<Value>,
    ) {
        if idx >= generators.len() {
            result.push(self.visit(elt));
            return;
        }

        let gen = &generators[idx];
        let iter_val = self.visit(&gen.iter_expr);

        let elements: Vec<Value> = match &iter_val {
            Value::List(data) | Value::Tuple(data) => data.values.clone(),
            _ => return,
        };

        for elem in &elements {
            self.do_recursive_assign(&gen.target, elem.clone(), false);

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
                self.expand_generators(generators, idx + 1, elt, result);
            }
        }
    }

    pub(crate) fn visit_cond_exp(&mut self, n: &ASTCondExp) -> Value {
        let cond = self.visit(&n.cond);
        let true_val = self.visit(&n.t_expr);
        let false_val = self.visit(&n.f_expr);
        let cond_bool = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &cond);
        crate::helpers::value_ops::select_value(&mut self.builder, &cond_bool, &true_val, &false_val)
    }

    pub(crate) fn visit_joined_str(&mut self, n: &ASTJoinedStr) -> Value {
        let values: Vec<Value> = n.values.iter().map(|v| self.visit(v)).collect();
        let mut result = self.builder.ir_constant_str(String::new());
        for val in values {
            result = self.builder.ir_add_str(&result, &val);
        }
        result
    }

    pub(crate) fn visit_formatted_value(&mut self, n: &ASTFormattedValue) -> Value {
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

    pub(crate) fn do_recursive_assign(&mut self, target: &ASTNode, value: Value, conditional_select: bool) {
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
                    let merged = crate::helpers::value_ops::select_value(&mut self.builder, &cond, &value, &orig);
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
}
