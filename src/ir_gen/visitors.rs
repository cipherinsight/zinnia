use std::collections::BTreeSet;

use crate::ast::*;
use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};
use crate::types::{CompositeData, NumberType, Value, ValueId, ZinniaType};

use super::{
    is_starred_target, ForLoopFrame, IRGenerator, LoopIndexBoundContext, LoopIndexExprKey,
    LoopReadReuseContext, ReusableLoopRead, SliceIndex, StaticArrayIdentity,
};

fn dyn_scalar_at(builder: &mut crate::builder::IRBuilder, value: &Value, index: usize) -> Value {
    let Value::DynamicNDArray(data) = value else {
        unreachable!("dyn_scalar_at expects DynamicNDArray")
    };
    if data.envelope.rank() != 1 {
        panic!(
            "TypeInferenceError: {} is not iterable as a fixed scalar tuple",
            value.zinnia_type()
        );
    }
    if index >= data.envelope.total_bound {
        panic!(
            "UnpackingError: cannot unpack DynamicNDArray bound {} into requested slot {}",
            data.envelope.total_bound, index
        );
    }
    let idx = builder.ir_constant_int(index as i64);
    crate::helpers::array_ops::dyn_subscript(builder, data, &[SliceIndex::Single(idx)])
}

fn dynamic_array_to_unpackable_list(
    builder: &mut crate::builder::IRBuilder,
    value: &Value,
    target_count: usize,
) -> Value {
    let Value::DynamicNDArray(data) = value else {
        unreachable!("dynamic_array_to_unpackable_list expects DynamicNDArray")
    };
    if data.envelope.rank() != 1 {
        panic!(
            "TypeInferenceError: {} is not iterable as a fixed scalar tuple",
            value.zinnia_type()
        );
    }
    let static_len = data.meta.runtime_shape.first().and_then(|d| d.static_val);
    let count_ok = static_len
        .map(|n| n as usize == target_count)
        .unwrap_or(data.envelope.total_bound == target_count);
    if !count_ok {
        panic!(
            "UnpackingError: cannot safely unpack DynamicNDArray with runtime length {:?} and bound {} into {} targets",
            static_len, data.envelope.total_bound, target_count
        );
    }
    let values: Vec<Value> = (0..target_count)
        .map(|i| dyn_scalar_at(builder, value, i))
        .collect();
    let elements_type = values.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type,
        values,
        value_id: ValueId::next(),
    })
}

impl IRGenerator {
    fn subscript_output_can_relay_forall_eq_const(out: &Value) -> bool {
        matches!(
            out,
            Value::NDArray(_)
                | Value::DynamicNDArray(_)
                | Value::StaticArray { .. }
                | Value::List(_)
                | Value::Tuple(_)
        )
    }

    fn relay_forall_eq_const_to_shaped_subscript_output(&mut self, input: &Value, output: &Value) {
        if !Self::subscript_output_can_relay_forall_eq_const(output) {
            return;
        }
        if let (Some(in_vid), Some(out_vid)) = (input.value_id(), output.value_id()) {
            crate::optim::resolver::relay_forall_eq_const_from_input(
                &mut self.builder,
                in_vid,
                out_vid,
            );
        }
    }

    fn simple_name_target(target: &ASTNode) -> Option<&str> {
        match target {
            ASTNode::ASTNameAssignTarget(t) => Some(t.name.as_str()),
            _ => None,
        }
    }

    fn assigned_name(target: &ASTNode) -> Option<&str> {
        match target {
            ASTNode::ASTNameAssignTarget(t) => Some(t.name.as_str()),
            _ => None,
        }
    }

    fn alias_from_active_for_var(&self, value: &ASTNode) -> Option<String> {
        let ASTNode::ASTLoad(load) = value else {
            return None;
        };
        let frame = self.for_loop_frames.last()?;
        if load.name == frame.var_name {
            Some(frame.var_name.clone())
        } else {
            None
        }
    }

    fn maybe_record_induction_alias(&mut self, target: &ASTNode, value: &ASTNode) {
        let Some(name) = Self::assigned_name(target).map(str::to_string) else {
            return;
        };
        if let Some(origin) = self.alias_from_active_for_var(value) {
            self.induction_origins.insert(name, origin);
        } else {
            self.induction_origins.remove(&name);
        }
    }

    fn for_frame_from_elements(target: &ASTNode, elements: &[Value]) -> Option<ForLoopFrame> {
        let var_name = Self::simple_name_target(target)?.to_string();
        if elements.is_empty() {
            return None;
        }
        let mut max_seen: Option<i64> = None;
        for elem in elements {
            let n = elem.int_val()?;
            max_seen = Some(max_seen.map_or(n, |cur| cur.max(n)));
        }
        let max_seen = max_seen?;
        if max_seen < 0 {
            return None;
        }
        let max_bound = (max_seen as u32).saturating_add(1);
        if max_bound == 0 {
            return None;
        }
        Some(ForLoopFrame {
            var_name,
            max_bound,
            current_bound: None,
        })
    }

    fn clear_induction_origins_from(&mut self, origin: &str) {
        self.induction_origins.retain(|_, value| value != origin);
    }

    fn guard_positive_var(node: &ASTNode) -> Option<&str> {
        match node {
            ASTNode::ASTBinaryOperator(bin) if bin.operator == "and" => {
                Self::guard_positive_var(&bin.lhs).or_else(|| Self::guard_positive_var(&bin.rhs))
            }
            ASTNode::ASTBinaryOperator(bin) if bin.operator == "gt" => {
                match (&*bin.lhs, &*bin.rhs) {
                    (ASTNode::ASTLoad(load), ASTNode::ASTConstantInteger(z)) if z.value == 0 => {
                        Some(load.name.as_str())
                    }
                    _ => None,
                }
            }
            ASTNode::ASTBinaryOperator(bin) if bin.operator == "lt" => {
                match (&*bin.lhs, &*bin.rhs) {
                    (ASTNode::ASTConstantInteger(z), ASTNode::ASTLoad(load)) if z.value == 0 => {
                        Some(load.name.as_str())
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn is_unit_decrement_assignment(var_name: &str, stmt: &ASTNode) -> bool {
        match stmt {
            ASTNode::ASTAssignStatement(assign) => {
                if assign.targets.len() != 1
                    || Self::assigned_name(&assign.targets[0]) != Some(var_name)
                {
                    return false;
                }
                let ASTNode::ASTBinaryOperator(bin) = &*assign.value else {
                    return false;
                };
                if bin.operator != "sub" {
                    return false;
                }
                matches!(
                    (&*bin.lhs, &*bin.rhs),
                    (ASTNode::ASTLoad(load), ASTNode::ASTConstantInteger(one))
                        if load.name == var_name && one.value == 1
                )
            }
            _ => false,
        }
    }

    fn stmt_writes_name(stmt: &ASTNode, var_name: &str) -> bool {
        let mut names = BTreeSet::new();
        Self::collect_assigned_names_from_stmt(stmt, &mut names);
        names.contains(var_name)
    }

    /// Whether the only write to `var_name` in `block` is one top-level
    /// `var_name = var_name - 1`. Then every executed iteration lowers it by
    /// exactly one (inactive iterations leave it unchanged), which is what
    /// makes the structural `while var > 0` cap a bound on the trip count.
    fn body_has_only_unit_decrement(block: &[ASTNode], var_name: &str) -> bool {
        let mut decrements = 0;
        for stmt in block {
            if Self::is_unit_decrement_assignment(var_name, stmt) {
                decrements += 1;
            } else if Self::stmt_writes_name(stmt, var_name) {
                return false;
            }
        }
        decrements == 1
    }

    fn structural_while_unroll_cap(&self, n: &ASTWhileStatement) -> Option<u32> {
        let var_name = Self::guard_positive_var(&n.test_expr)?;
        let origin = self.induction_origins.get(var_name)?;
        let frame = self.for_loop_frames.last()?;
        if origin != &frame.var_name || frame.max_bound == 0 {
            return None;
        }
        if !Self::body_has_only_unit_decrement(&n.block, var_name) {
            return None;
        }
        frame
            .current_bound
            .filter(|bound| *bound > 0)
            .or(Some(frame.max_bound))
    }

    fn structural_loop_index_bound_context(
        &self,
        n: &ASTWhileStatement,
    ) -> Option<LoopIndexBoundContext> {
        let var_name = Self::guard_positive_var(&n.test_expr)?;
        let origin = self.induction_origins.get(var_name)?;
        let frame = self.for_loop_frames.last()?;
        if origin != &frame.var_name || frame.max_bound == 0 {
            return None;
        }
        let current_bound = frame.current_bound?;
        if current_bound == 0 {
            return None;
        }
        if !Self::body_has_only_unit_decrement(&n.block, var_name) {
            return None;
        }
        Some(LoopIndexBoundContext {
            var_name: var_name.to_string(),
            max_bound: frame.max_bound,
            current_bound,
            positive_guard: true,
            guard_condition: None,
            record_guard_read: false,
        })
    }

    fn collect_assigned_names_from_target(target: &ASTNode, names: &mut BTreeSet<String>) {
        match target {
            ASTNode::ASTNameAssignTarget(t) => {
                names.insert(t.name.clone());
            }
            ASTNode::ASTTupleAssignTarget(t) => {
                for target in &t.targets {
                    Self::collect_assigned_names_from_target(target, names);
                }
            }
            ASTNode::ASTListAssignTarget(t) => {
                for target in &t.targets {
                    Self::collect_assigned_names_from_target(target, names);
                }
            }
            ASTNode::ASTSubscriptAssignTarget(_) => {}
            _ => {}
        }
    }

    pub(crate) fn collect_assigned_names_from_stmt(stmt: &ASTNode, names: &mut BTreeSet<String>) {
        match stmt {
            ASTNode::ASTAssignStatement(assign) => {
                for target in &assign.targets {
                    Self::collect_assigned_names_from_target(target, names);
                }
            }
            ASTNode::ASTAugAssignStatement(assign) => {
                for target in &assign.targets {
                    Self::collect_assigned_names_from_target(target, names);
                }
            }
            ASTNode::ASTForInStatement(for_in) => {
                Self::collect_assigned_names_from_target(&for_in.target, names);
                for stmt in &for_in.block {
                    Self::collect_assigned_names_from_stmt(stmt, names);
                }
                for stmt in &for_in.orelse {
                    Self::collect_assigned_names_from_stmt(stmt, names);
                }
            }
            ASTNode::ASTWhileStatement(while_stmt) => {
                for stmt in &while_stmt.block {
                    Self::collect_assigned_names_from_stmt(stmt, names);
                }
                for stmt in &while_stmt.orelse {
                    Self::collect_assigned_names_from_stmt(stmt, names);
                }
            }
            ASTNode::ASTCondStatement(cond) => {
                for stmt in &cond.t_block {
                    Self::collect_assigned_names_from_stmt(stmt, names);
                }
                for stmt in &cond.f_block {
                    Self::collect_assigned_names_from_stmt(stmt, names);
                }
            }
            _ => {}
        }
    }

    fn mark_unreachable_loop_body_bindings(&mut self, block: &[ASTNode], reason: &str) {
        let mut names = BTreeSet::new();
        for stmt in block {
            Self::collect_assigned_names_from_stmt(stmt, &mut names);
        }
        for name in names {
            self.ctx
                .mark_unreachable_loop_binding(&name, reason.to_string());
        }
    }

    pub(crate) fn bool_value_fact(val: &Value, expected: bool) -> Option<ContractTerm> {
        val.value_id().map(|vid| ContractTerm::Cmp {
            op: CmpOp::Eq,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitInt(if expected { 1 } else { 0 })),
        })
    }

    fn assert_cmp_op(op: &str) -> Option<CmpOp> {
        match op {
            "eq" => Some(CmpOp::Eq),
            "ne" => Some(CmpOp::Ne),
            "lt" => Some(CmpOp::Lt),
            "lte" => Some(CmpOp::Le),
            "gt" => Some(CmpOp::Gt),
            "gte" => Some(CmpOp::Ge),
            _ => None,
        }
    }

    fn assert_term_leaf(&mut self, node: &ASTNode) -> Option<ContractTerm> {
        match node {
            ASTNode::ASTLoad(load) => match self.ctx.get(&load.name)? {
                Value::Integer(v) => Some(ContractTerm::Var(ContractVar::Value(v.value_id))),
                Value::Float(v) => Some(ContractTerm::Var(ContractVar::Value(v.value_id))),
                Value::Boolean(v) => Some(ContractTerm::Var(ContractVar::Value(v.value_id))),
                _ => None,
            },
            ASTNode::ASTConstantInteger(n) => Some(ContractTerm::LitInt(n.value)),
            ASTNode::ASTConstantFloat(n) => Some(ContractTerm::LitFloat(ContractFloat(n.value))),
            ASTNode::ASTConstantBoolean(n) => {
                Some(ContractTerm::LitInt(if n.value { 1 } else { 0 }))
            }
            _ => None,
        }
    }

    fn assert_comparison_fact_term(&mut self, node: &ASTNode) -> Option<ContractTerm> {
        let ASTNode::ASTBinaryOperator(bin) = node else {
            return None;
        };
        let op = Self::assert_cmp_op(bin.operator.as_str())?;
        let lhs = self.assert_term_leaf(&bin.lhs)?;
        let rhs = self.assert_term_leaf(&bin.rhs)?;
        let term = ContractTerm::Cmp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
        if crate::optim::predicates::collect_value_ids(&term).is_empty() {
            None
        } else {
            Some(term)
        }
    }

    fn collect_assert_fact_terms(&mut self, node: &ASTNode, out: &mut Vec<ContractTerm>) {
        if let ASTNode::ASTBinaryOperator(bin) = node {
            if bin.operator == "and" {
                self.collect_assert_fact_terms(&bin.lhs, out);
                self.collect_assert_fact_terms(&bin.rhs, out);
                return;
            }
        }
        if let Some(term) = self.assert_comparison_fact_term(node) {
            out.push(term);
        }
    }

    fn assert_fact_terms(&mut self, node: &ASTNode) -> Vec<ContractTerm> {
        let mut terms = Vec::new();
        self.collect_assert_fact_terms(node, &mut terms);
        terms
    }

    fn publish_assert_fact(&mut self, term: ContractTerm) {
        let anchors = crate::optim::predicates::collect_value_ids(&term);
        for anchor in anchors {
            self.builder.facts.insert_for(anchor, term.clone());
        }
    }

    fn load_name(node: &ASTNode) -> Option<&str> {
        match node {
            ASTNode::ASTLoad(load) => Some(load.name.as_str()),
            _ => None,
        }
    }

    fn is_active_loop_index_name(&self, name: &str) -> bool {
        let Some(ctx) = self.loop_index_bound_contexts.last() else {
            return false;
        };
        ctx.positive_guard && name == ctx.var_name
    }

    fn loop_index_bound_context_allows_array_len(&self, len: usize) -> bool {
        let Some(ctx) = self.loop_index_bound_contexts.last() else {
            return false;
        };
        ctx.positive_guard && (ctx.current_bound as usize) <= len && (ctx.max_bound as usize) <= len
    }

    fn slice_has_preproved_loop_index_bounds_for_array(
        &self,
        slice: &ASTSlice,
        array_len: usize,
    ) -> bool {
        if slice.data.len() != 1 || !self.loop_index_bound_context_allows_array_len(array_len) {
            return false;
        }
        let Ok(node) = serde_json::from_value::<ASTNode>(slice.data[0].clone()) else {
            return false;
        };
        if let Some(name) = Self::load_name(&node) {
            return self.is_active_loop_index_name(name);
        }
        let ASTNode::ASTBinaryOperator(bin) = node else {
            return false;
        };
        if bin.operator != "sub" {
            return false;
        }
        let Some(name) = Self::load_name(&bin.lhs) else {
            return false;
        };
        matches!(&*bin.rhs, ASTNode::ASTConstantInteger(one) if one.value == 1)
            && self.is_active_loop_index_name(name)
    }

    fn path_conditioned_loop_index_bound_context(&self) -> Option<&LoopIndexBoundContext> {
        let ctx = self.loop_index_bound_contexts.last()?;
        (ctx.positive_guard && ctx.guard_condition.is_some()).then_some(ctx)
    }

    fn static_array_identity(value: &Value) -> Option<StaticArrayIdentity> {
        match value {
            Value::StaticArray {
                dtype,
                shape,
                segment_id,
                strides,
                offset,
                ..
            } if *dtype != NumberType::Complex && shape.len() == 1 => Some(StaticArrayIdentity {
                dtype: *dtype,
                segment_id: *segment_id,
                offset: *offset,
                shape: shape.clone(),
                strides: strides.clone(),
            }),
            _ => None,
        }
    }

    fn loop_index_expr_key(&self, slice: &ASTSlice) -> Option<LoopIndexExprKey> {
        if slice.data.len() != 1 {
            return None;
        }
        let Ok(node) = serde_json::from_value::<ASTNode>(slice.data[0].clone()) else {
            return None;
        };
        if let Some(name) = Self::load_name(&node) {
            return self
                .is_active_loop_index_name(name)
                .then_some(LoopIndexExprKey::Var);
        }
        let ASTNode::ASTBinaryOperator(bin) = node else {
            return None;
        };
        if bin.operator != "sub" {
            return None;
        }
        let Some(name) = Self::load_name(&bin.lhs) else {
            return None;
        };
        (matches!(&*bin.rhs, ASTNode::ASTConstantInteger(one) if one.value == 1)
            && self.is_active_loop_index_name(name))
        .then_some(LoopIndexExprKey::MinusOne)
    }

    fn matching_loop_read_reuse(
        &mut self,
        array: &StaticArrayIdentity,
        slice: &ASTSlice,
    ) -> Option<Value> {
        let index = self.loop_index_expr_key(slice)?;
        let ctx = self.loop_read_reuse_contexts.last_mut()?;
        if !ctx.body_prefix_active {
            return None;
        }
        let candidate = ctx.candidate.as_ref()?;
        if &candidate.array == array && candidate.index == index {
            let value = candidate.value.clone();
            ctx.candidate = None;
            ctx.body_prefix_active = false;
            Some(value)
        } else {
            None
        }
    }

    fn record_loop_guard_read_reuse(
        &mut self,
        array: Option<StaticArrayIdentity>,
        index: Option<LoopIndexExprKey>,
        value: &Value,
    ) {
        let Some(ctx) = self.loop_index_bound_contexts.last() else {
            return;
        };
        if !ctx.record_guard_read {
            return;
        }
        let Some(array) = array else {
            return;
        };
        let Some(index) = index else {
            return;
        };
        let Some(reuse_ctx) = self.loop_read_reuse_contexts.last_mut() else {
            return;
        };
        reuse_ctx.candidate = Some(ReusableLoopRead {
            array,
            index,
            value: value.clone(),
        });
    }

    fn clear_loop_read_reuse(&mut self) {
        if let Some(ctx) = self.loop_read_reuse_contexts.last_mut() {
            ctx.candidate = None;
            ctx.body_prefix_active = false;
        }
    }

    fn body_stmt_can_preserve_read_reuse_prefix(stmt: &ASTNode) -> bool {
        match stmt {
            ASTNode::ASTAssignStatement(assign) if assign.targets.len() == 1 => {
                matches!(
                    &assign.targets[0],
                    ASTNode::ASTSubscriptAssignTarget(target)
                        if matches!(&*target.target, ASTNode::ASTLoad(_))
                )
            }
            _ => false,
        }
    }

    fn target_writes_static_array_identity(
        &mut self,
        target: &ASTNode,
        array: &StaticArrayIdentity,
    ) -> bool {
        let ASTNode::ASTSubscriptAssignTarget(t) = target else {
            return false;
        };
        let current = match &*t.target {
            ASTNode::ASTLoad(load) => self.ctx.get(&load.name),
            _ => None,
        };
        current
            .as_ref()
            .and_then(Self::static_array_identity)
            .is_some_and(|id| id == *array)
    }

    fn positive_guard_matches_var(node: &ASTNode, var_name: &str) -> bool {
        match node {
            ASTNode::ASTBinaryOperator(bin) if bin.operator == "gt" => {
                matches!(
                    (&*bin.lhs, &*bin.rhs),
                    (ASTNode::ASTLoad(load), ASTNode::ASTConstantInteger(z))
                        if load.name == var_name && z.value == 0
                )
            }
            ASTNode::ASTBinaryOperator(bin) if bin.operator == "lt" => {
                matches!(
                    (&*bin.lhs, &*bin.rhs),
                    (ASTNode::ASTConstantInteger(z), ASTNode::ASTLoad(load))
                        if z.value == 0 && load.name == var_name
                )
            }
            _ => false,
        }
    }

    fn visit_while_guard_expr(
        &mut self,
        node: &ASTNode,
        loop_index_bound_context: Option<&LoopIndexBoundContext>,
    ) -> Value {
        let Some(base_ctx) = loop_index_bound_context else {
            return self.visit(node);
        };
        let ASTNode::ASTBinaryOperator(bin) = node else {
            return self.visit(node);
        };
        if bin.operator != "and" || !Self::positive_guard_matches_var(&bin.lhs, &base_ctx.var_name)
        {
            return self.visit(node);
        }

        let lhs = self.visit(&bin.lhs);
        let lhs_bool = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &lhs);
        let mut rhs_ctx = base_ctx.clone();
        rhs_ctx.guard_condition = Some(lhs_bool.clone());
        rhs_ctx.record_guard_read = true;
        self.loop_index_bound_contexts.push(rhs_ctx);
        let rhs = self.visit(&bin.rhs);
        self.loop_index_bound_contexts.pop();
        let rhs_bool = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &rhs);
        self.builder.ir_logical_and(&lhs_bool, &rhs_bool)
    }

    pub(crate) fn visit_assign(&mut self, n: &ASTAssignStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let val = self.visit(&n.value);
        if n.augmented {
            if let [ASTNode::ASTNameAssignTarget(target)] = n.targets.as_slice() {
                if self.update_in_place(&target.name, &val) {
                    return;
                }
            }
        }
        for target in &n.targets {
            self.maybe_record_induction_alias(target, &n.value);
            self.do_recursive_assign(target, val.clone(), true);
        }
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
                if data
                    .values
                    .iter()
                    .any(|v| matches!(v, Value::List(_) | Value::Tuple(_)))
                {
                    panic!("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                }
            }
        }
        // P4a: same guard for the segment-backed `Value::StaticArray`. After
        // the constructor migration, `np.ones((2, 2))`-style values land here
        // as `StaticArray`s instead of nested `List`s; a multi-D condition
        // must still raise rather than silently reducing.
        if let Value::StaticArray { shape, .. } = &cond_val {
            if shape.iter().product::<usize>() > 1 && shape.len() > 1 {
                panic!(
                    "The truth value of an array with more than one element is ambiguous. \
                     Use a.any() or a.all()"
                );
            }
        }
        let true_cond = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &cond_val);
        let false_cond = self.builder.ir_logical_not(&true_cond);

        let true_static = true_cond
            .bool_val()
            .or_else(|| true_cond.int_val().map(|v| v != 0));
        let false_static = false_cond
            .bool_val()
            .or_else(|| false_cond.int_val().map(|v| v != 0));

        // True branch
        let mut scope_true_ret_guaranteed = false;
        let mut scope_true_term_guaranteed = false;
        let mut then_scope: Option<crate::scope::Scope> = None;
        let mut then_fact_scope: Option<crate::optim::predicates::FactScope> = None;
        if true_static.is_none() || true_static == Some(true) {
            self.ctx.if_enter(true_cond.clone(), &mut self.builder);
            let path_predicate = Self::bool_value_fact(&true_cond, true);
            self.builder.facts.enter(
                crate::optim::predicates::ScopeKind::Conditional,
                path_predicate,
            );
            // Each statement in the arm sets the write predicate to its own
            // full execution condition, which includes this arm's condition.
            for stmt in &n.t_block {
                self.visit(stmt);
            }
            scope_true_ret_guaranteed = self.ctx.check_return_guaranteed();
            scope_true_term_guaranteed = self.ctx.check_loop_terminated_guaranteed();
            let scope_true = self.ctx.if_leave();
            then_fact_scope = Some(self.builder.facts.leave());
            then_scope = Some(scope_true);
        }

        // False branch
        let mut scope_false_ret_guaranteed = false;
        let mut scope_false_term_guaranteed = false;
        let mut else_scope: Option<crate::scope::Scope> = None;
        let mut else_fact_scope: Option<crate::optim::predicates::FactScope> = None;
        if false_static.is_none() || false_static == Some(true) {
            self.ctx.if_enter(false_cond.clone(), &mut self.builder);
            let path_predicate = Self::bool_value_fact(&true_cond, false);
            self.builder.facts.enter(
                crate::optim::predicates::ScopeKind::Conditional,
                path_predicate,
            );
            for stmt in &n.f_block {
                self.visit(stmt);
            }
            scope_false_ret_guaranteed = self.ctx.check_return_guaranteed();
            scope_false_term_guaranteed = self.ctx.check_loop_terminated_guaranteed();
            let scope_false = self.ctx.if_leave();
            else_fact_scope = Some(self.builder.facts.leave());
            else_scope = Some(scope_false);
        }

        // Branch join: dynamic branches use guarded implications for
        // arm-only facts. If either branch lacks a representable guard, or
        // only one branch executed because the condition was static, keep the
        // fallback conservative.
        match (then_fact_scope, else_fact_scope) {
            (Some(t), Some(e)) => {
                if t.path_conditions.is_empty() || e.path_conditions.is_empty() {
                    self.builder.facts.merge_branches_intersect_only(t, e);
                } else {
                    self.builder.facts.merge_branches(t, e);
                }
            }
            (Some(t), None) if true_static == Some(true) => {
                self.builder.facts.absorb_unconditional(t);
            }
            (None, Some(e)) if false_static == Some(true) => {
                self.builder.facts.absorb_unconditional(e);
            }
            _ => {}
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

        match (then_scope, else_scope) {
            (Some(then_scope), Some(else_scope)) => {
                self.merge_conditional_branch_bindings(then_scope, else_scope, &true_cond);
            }
            // Statically-resolved condition: exactly one arm ran, so its
            // bindings propagate unconditionally (Python semantics).
            (Some(scope), None) | (None, Some(scope)) => {
                self.propagate_single_branch_bindings(scope);
            }
            (None, None) => {}
        }
    }

    /// Join variable bindings after a dynamic two-arm conditional.
    ///
    /// Names assigned in BOTH arms merge via `select(cond, then, else)`.
    /// Names assigned in ONE arm propagate as-is IF they already existed
    /// outside the branch: `do_recursive_assign` pre-merged such values
    /// against the pre-branch binding using the full path condition, so
    /// the stored value is already correct for both paths. (Before
    /// 2026-06-11 one-armed assignments were silently DROPPED — `r = 7;
    /// if cond: r = 3` kept `r == 7` on every path; see the binding-audit
    /// notes for case 0374.) A name born inside one arm of a dynamic
    /// branch is conditionally defined and stays branch-local.
    fn merge_conditional_branch_bindings(
        &mut self,
        then_scope: crate::scope::Scope,
        else_scope: crate::scope::Scope,
        condition: &Value,
    ) {
        let Some(then_vars) = then_scope.get_local_var_table() else {
            return;
        };
        let Some(else_vars) = else_scope.get_local_var_table() else {
            return;
        };

        let mut names: Vec<String> = then_vars.keys().cloned().collect();
        for name in else_vars.keys() {
            if !then_vars.contains_key(name) {
                names.push(name.clone());
            }
        }

        for name in names {
            match (then_vars.get(&name), else_vars.get(&name)) {
                (Some(t), Some(e)) => {
                    let then_val = t.value.clone();
                    let else_val = e.value.clone();
                    let merged = self.merge_binding(condition, &then_val, &else_val);
                    self.ctx.set(&name, merged);
                }
                (Some(one), None) | (None, Some(one)) => {
                    if self.ctx.exists(&name) {
                        let val = one.value.clone();
                        self.ctx.set(&name, val);
                    }
                }
                (None, None) => unreachable!(),
            }
        }
    }

    /// Propagate bindings out of the single arm of a statically-resolved
    /// conditional: the arm definitely ran, so existing names update and
    /// new names become defined, both unconditionally.
    fn propagate_single_branch_bindings(&mut self, scope: crate::scope::Scope) {
        let Some(vars) = scope.get_local_var_table() else {
            return;
        };
        for (name, store) in vars {
            let val = store.value.clone();
            self.ctx.set(name, val);
        }
    }

    /// Rebind a subscript-assignment target after a setitem. In-place
    /// (same-segment) updates already carry branch predication at the
    /// memory level (`IRBuilder::write_predicate`); rebinds that produce
    /// a NEW value (materialised List, fresh dyn array, ...) must merge
    /// against the previous binding under the current path condition, or
    /// a dynamic branch's setitem would apply unconditionally.
    fn rebind_subscript_target(&mut self, var_name: &str, current: &Value, updated: Value) {
        let same_segment = match (current, &updated) {
            (
                Value::StaticArray { segment_id: a, .. },
                Value::StaticArray { segment_id: b, .. },
            ) => a == b,
            (Value::DynamicNDArray(da), Value::DynamicNDArray(db)) => {
                da.segment_id == db.segment_id
            }
            _ => false,
        };
        if same_segment {
            self.ctx.set(var_name, updated);
            return;
        }
        // Another binding (or the caller) still holds the old array.
        if let Some(written) = self.write_back_if_shared(var_name, current, &updated) {
            self.ctx.set(var_name, written);
            return;
        }
        if !self.ctx.has_nontrivial_condition() {
            self.ctx.set(var_name, updated);
            return;
        }
        let cond = self.ctx.get_condition_value(&mut self.builder);
        let merged =
            crate::helpers::value_ops::select_value(&mut self.builder, &cond, &updated, current);
        self.ctx.set(var_name, merged);
    }

    pub(crate) fn visit_for_in(&mut self, n: &ASTForInStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let items = self.iteration_items(&n.iter_expr);
        self.unroll_loop(&n.target, &items, &n.block, &n.orelse);
    }

    /// Unroll a `for` loop over `items` (see [`super::iteration`]). The
    /// target is bound to each item's value unconditionally, so indices
    /// built from it stay compile-time constants inside the body; an
    /// inactive item skips its body like a `continue`. After the loop a
    /// simple-name target holds its value at the end of the last item that
    /// actually ran, as in Python.
    pub(crate) fn unroll_loop(
        &mut self,
        target: &ASTNode,
        items: &[super::iteration::LoopItem],
        block: &[ASTNode],
        orelse: &[ASTNode],
    ) {
        let elements: Vec<Value> = items.iter().map(|item| item.value.clone()).collect();
        // Loop targets are bound unconditionally during symbolic unrolling so
        // indices remain static. Preserve their old values when an enclosing
        // branch does not execute; branch joins expect pre-merged bindings.
        let entry_condition = self.ctx.get_condition_value(&mut self.builder);
        let mut target_names = BTreeSet::new();
        Self::collect_assigned_names_from_target(target, &mut target_names);
        let previous_targets: Vec<_> = target_names.into_iter()
            .filter_map(|name| self.ctx.get(&name).map(|value| (name, value)))
            .collect();
        let for_frame = Self::for_frame_from_elements(target, &elements);

        if elements.is_empty() {
            let mut names = BTreeSet::new();
            Self::collect_assigned_names_from_target(target, &mut names);
            for stmt in block {
                Self::collect_assigned_names_from_stmt(stmt, &mut names);
            }
            for name in names {
                self.ctx.mark_unreachable_loop_binding(
                    &name,
                    "loop body is unreachable because the for-loop has zero iterations".to_string(),
                );
            }
        }
        self.ctx.loop_enter();
        self.builder
            .facts
            .enter(crate::optim::predicates::ScopeKind::Loop, None);
        let pushed_frame = for_frame.is_some();
        if let Some(frame) = for_frame {
            self.clear_induction_origins_from(&frame.var_name);
            self.for_loop_frames.push(frame);
        }
        let target_name = Self::simple_name_target(target);
        let guarded = items.iter().any(|item| item.active.is_some());
        // The target's value at the end of the last item that ran.
        let mut exit_target: Option<Value> =
            target_name.and_then(|name| self.ctx.get(name));
        for item in items {
            self.ctx.loop_reiter(&mut self.builder);
            // An inactive item skips its body like a `continue`, so a later
            // `else` block still sees the loop as exhausted, not broken.
            if let Some(active) = &item.active {
                let inactive = self.builder.ir_logical_not(active);
                self.ctx.loop_skip_iteration_if(inactive, &mut self.builder);
            }
            if pushed_frame {
                if let Some(frame) = self.for_loop_frames.last_mut() {
                    frame.current_bound = item
                        .value
                        .int_val()
                        .and_then(|n| (n >= 0).then(|| (n as u32).saturating_add(1)));
                }
            }

            self.do_recursive_assign(target, item.value.clone(), false);

            let mut terminated = false;
            for stmt in block {
                self.visit(stmt);
                if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed()
                {
                    terminated = true;
                    break;
                }
            }
            if guarded {
                if let Some(end) = target_name.and_then(|name| self.ctx.get(name)) {
                    exit_target = Some(match (&item.active, exit_target.take()) {
                        (Some(active), Some(previous)) => self.merge_binding(active, &end, &previous),
                        _ => end,
                    });
                }
            }
            if terminated {
                break;
            }
        }

        // The `else` block runs when the loop was not broken. A `continue`
        // (or an inactive item) in the last iteration does not affect it, so
        // reset the per-iteration skip condition first.
        if !orelse.is_empty() {
            self.ctx.loop_reiter(&mut self.builder);
        }
        for stmt in orelse {
            self.visit(stmt);
        }

        // For-loop trips are statically enumerated, so a return guarantee
        // established inside the body (or the for-else block, which always
        // runs when the loop wasn't broken) holds unconditionally outside.
        // The flag lives on the Loop scope and would vanish at loop_leave.
        let returned = self.ctx.check_return_guaranteed();
        self.ctx.loop_leave();
        if pushed_frame {
            if let Some(frame) = self.for_loop_frames.pop() {
                self.clear_induction_origins_from(&frame.var_name);
            }
        }
        let _loop_facts = self.builder.facts.leave();
        if let (true, Some(name), Some(exit)) = (guarded, target_name, exit_target) {
            self.ctx.set(name, exit);
        }
        if entry_condition.bool_val() != Some(true) {
            for (name, previous) in previous_targets {
                if let Some(current) = self.ctx.get(&name) {
                    let merged = self.merge_binding(&entry_condition, &current, &previous);
                    self.ctx.set(&name, merged);
                }
            }
        }
        if returned {
            self.ctx.set_return_guarantee();
        }
    }

    pub(crate) fn visit_while(&mut self, n: &ASTWhileStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }

        if !n.orelse.is_empty() && Self::block_has_loop_break(&n.block) {
            panic!(
                "`while ... else` with `break` in the loop body is not supported: \
                 the `else` block must run only when the loop ends without `break`"
            );
        }
        if let Some((counter, items)) = self.counted_while_items(n) {
            let target = ASTNode::ASTNameAssignTarget(ASTNameAssignTarget {
                name: counter,
                star: false,
                dbg: None,
            });
            self.unroll_loop(&target, &items, &n.block, &n.orelse);
            return;
        }
        let structural_cap = self.structural_while_unroll_cap(n);
        let loop_limit = structural_cap.unwrap_or(self.config.loop_limit);
        let mut loop_quota = loop_limit + 1;
        let loop_index_bound_context = self.structural_loop_index_bound_context(n);
        self.ctx.loop_enter();
        self.builder
            .facts
            .enter(crate::optim::predicates::ScopeKind::Loop, None);

        // P4 round 1.5: per-iteration `resolve_bool` calls in tight while
        // loops are the dominant overhead in round-1's +41% slowdown. Guard
        // the layered resolver behind two cheap checks:
        //   (1) `bool_val()` / `int_val()` — O(1) static-val cache lookup.
        //   (2) Guard `ptr()` memoization — if the guard's StmtId is the
        //       same as the previous iteration's, the IR DAG underneath it
        //       is identical (statements are append-only, never mutated),
        //       so the resolver would return the same answer. Reuse the
        //       previously cached resolution instead of re-running the
        //       layered resolver.
        let mut prev_guard_ptr: Option<crate::types::StmtId> = None;
        let mut prev_resolver_result: Option<bool> = None;
        let mut returned_guaranteed_outside = false;
        let mut unresolved_guard_queries = 0usize;
        const WHILE_GUARD_UNRESOLVED_QUERY_BUDGET: usize = 8;

        loop {
            self.ctx.loop_reiter(&mut self.builder);
            let pushed_reuse_context = loop_index_bound_context.is_some();
            if pushed_reuse_context {
                self.loop_read_reuse_contexts
                    .push(LoopReadReuseContext::default());
            }
            let test = self.visit_while_guard_expr(&n.test_expr, loop_index_bound_context.as_ref());
            let test_bool = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &test);
            loop_quota -= 1;

            // P4 round 1: route the guard resolution through the active
            // Resolver (instead of only the value's static_val cache). Static
            // bools / ints stay on the cheap `bool_val` / `int_val` path; for
            // a symbolic guard the LayeredResolver gets a shot at proving it
            // const via the range layer or SMT. This is the paper's
            // While-Step / While-Exit rules: if the guard provably evaluates
            // to false against the post-iteration environment, exit the
            // unroll loop immediately so we don't pay symbolic-execution
            // cost on the unreached iterations.
            //
            // P4 round 1.5: only consult the resolver when the guard is
            // (a) not statically known via `bool_val`/`int_val`, AND
            // (b) the guard's StmtId changed since the last iteration.
            // Track the resolver's result across iterations so a cached
            // answer can still drive the early-exit on subsequent iters.
            let static_fast = test_bool
                .bool_val()
                .or_else(|| test_bool.int_val().map(|v| v != 0));

            let test_static = if let Some(b) = static_fast {
                Some(b)
            } else {
                let cur_ptr = test_bool.stmt_id();
                if cur_ptr.is_some() && cur_ptr == prev_guard_ptr {
                    // Same guard StmtId as last iter — IR is monotonic,
                    // so the resolver would return the same answer.
                    prev_resolver_result
                } else if unresolved_guard_queries >= WHILE_GUARD_UNRESOLVED_QUERY_BUDGET {
                    // Fail open to symbolic unroll once this loop has shown
                    // that guard-resolution queries are unproductive. This
                    // preserves semantics: `None` is the same answer the
                    // resolver has been returning, just without another
                    // expensive SMT attempt.
                    if let Some(t) = self.builder.resolver_telemetry() {
                        t.queries_skipped_while_throttle
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    prev_guard_ptr = cur_ptr;
                    prev_resolver_result = None;
                    None
                } else {
                    let (resolver, stmts) = self.builder.split_resolver_and_stmts();
                    let resolved = resolver.resolve_bool_with_stmts(&test_bool, stmts);
                    if resolved.is_none() {
                        unresolved_guard_queries += 1;
                    } else {
                        unresolved_guard_queries = 0;
                    }
                    prev_guard_ptr = cur_ptr;
                    prev_resolver_result = resolved;
                    resolved
                }
            };

            // Bound-based exit: the guard can no longer hold for any input
            // the compiler admits (e.g. `i < n` once `i` reaches the proven
            // maximum of `n`).
            let test_static = match test_static {
                None if self.provably_false(&test_bool) => Some(false),
                other => other,
            };

            if test_static != Some(false) && loop_quota == 0 {
                if pushed_reuse_context {
                    self.loop_read_reuse_contexts.pop();
                }
                self.finish_unroll_limit(&test_bool, structural_cap.is_some(), loop_limit);
                break;
            }

            if test_static == Some(false) {
                if loop_quota == loop_limit {
                    self.mark_unreachable_loop_body_bindings(
                        &n.block,
                        "loop body is unreachable because the while condition is provably false before the first iteration",
                    );
                }
                if pushed_reuse_context {
                    self.loop_read_reuse_contexts.pop();
                }
                break;
            }

            if test_static.is_none() {
                let not_test = self.builder.ir_logical_not(&test_bool);
                self.ctx.loop_break(Some(not_test), &mut self.builder);
            }

            let mut terminated = false;
            let pushed_loop_index_context = if let Some(ctx) = loop_index_bound_context.clone() {
                self.loop_index_bound_contexts.push(ctx);
                true
            } else {
                false
            };
            for stmt in &n.block {
                if !Self::body_stmt_can_preserve_read_reuse_prefix(stmt) {
                    self.clear_loop_read_reuse();
                }
                self.visit(stmt);
                if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed()
                {
                    // A return reached during an iteration whose guard was
                    // statically TRUE is unconditional from outside the
                    // loop: `while True: ... return x` guarantees a
                    // return. (The guarantee flag lives on the Loop scope
                    // and would otherwise vanish at loop_leave.)
                    if self.ctx.check_return_guaranteed() && test_static == Some(true) {
                        returned_guaranteed_outside = true;
                    }
                    terminated = true;
                    break;
                }
            }
            if pushed_loop_index_context {
                self.loop_index_bound_contexts.pop();
            }
            if pushed_reuse_context {
                self.loop_read_reuse_contexts.pop();
            }
            if terminated {
                break;
            }
        }

        self.ctx.loop_leave();
        let _while_facts = self.builder.facts.leave();
        if returned_guaranteed_outside {
            self.ctx.set_return_guarantee();
        }
        // Without a `break`, Python runs the `else` block whenever the loop
        // ends, which (after the termination check above) it always does.
        for stmt in &n.orelse {
            self.visit(stmt);
        }
    }

    /// The unroll of a `while` loop used up its iteration quota and the guard
    /// is not known to be false. If no path can still be iterating, the loop
    /// has ended. A structural cap is a bound by construction: the guard
    /// requires `j > 0`, `j` starts at most at the enclosing for-loop's
    /// current item, and the body's only write to `j` lowers it by one per
    /// executed iteration (`body_has_only_unit_decrement`), so after `cap`
    /// iterations no path can still be iterating. Otherwise the trip count
    /// has no provable maximum and the program is rejected: truncating the
    /// loop would prove a different result.
    fn finish_unroll_limit(&mut self, guard: &Value, structural_cap: bool, limit: u32) {
        if structural_cap {
            return;
        }
        let condition = self.ctx.get_condition_value(&mut self.builder);
        let still_running = self.builder.ir_logical_and(&condition, guard);
        if still_running.bool_val() == Some(false) || self.provably_false(&still_running) {
            return;
        }
        panic!(
            "{}: the `while` condition was not proven false within loop_limit={} unrolled \
             iterations. Bound the values the condition depends on (a bounded array \
             annotation or a `requires` precondition), or raise ZinniaConfig loop_limit if a \
             larger bound exists",
            super::iteration::UNBOUNDED_LOOP_ERROR,
            limit
        );
    }

    /// Whether `block` contains a `break` that exits the loop owning
    /// `block` (breaks inside nested loops belong to those loops).
    fn block_has_loop_break(block: &[ASTNode]) -> bool {
        block.iter().any(|stmt| match stmt {
            ASTNode::ASTBreakStatement(_) => true,
            ASTNode::ASTCondStatement(cond) => {
                Self::block_has_loop_break(&cond.t_block) || Self::block_has_loop_break(&cond.f_block)
            }
            _ => false,
        })
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
        let is_bare_return = n.expr.is_none();
        let val = match &n.expr {
            Some(expr) => self.visit(expr),
            None => Value::None,
        };
        self.ctx
            .register_return(val, is_bare_return, &mut self.builder);
        self.ctx.set_return_guarantee();
    }

    pub(crate) fn visit_assert(&mut self, n: &ASTAssertStatement) {
        if self.ctx.check_return_guaranteed() || self.ctx.check_loop_terminated_guaranteed() {
            return;
        }
        let facts = self.assert_fact_terms(&n.expr);
        let test = self.visit(&n.expr);
        self.assert_value(&test);
        for fact in facts {
            self.publish_assert_fact(fact);
        }
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
        crate::helpers::value_ops::apply_binary_op(
            &mut self.builder,
            n.operator.as_str(),
            &lhs,
            &rhs,
        )
    }

    pub(crate) fn visit_unary_op(&mut self, n: &ASTUnaryOperator) -> Value {
        let operand = self.visit(&n.operand);
        self.apply_unary_op(n.operator.as_str(), &operand)
    }

    /// Apply a unary operation, with element-wise support for composite types.
    pub(crate) fn apply_unary_op(&mut self, op: &str, operand: &Value) -> Value {
        // P4a: native StaticArray unary dispatch. Falls back to the legacy
        // materialised path for unsupported ops (we only migrated
        // `usub`, `uadd`, `not`, `invert`).
        if matches!(operand, Value::StaticArray { .. }) {
            if let Some(out) = crate::helpers::static_array::elementwise::try_apply_unary_op(
                &mut self.builder,
                op,
                operand,
            ) {
                return out;
            }
            let lst = crate::helpers::static_array::base::to_value_list(&mut self.builder, operand);
            return self.apply_unary_op(op, &lst);
        }
        match operand {
            Value::List(data) | Value::Tuple(data) => {
                let results: Vec<Value> = data
                    .values
                    .iter()
                    .map(|v| self.apply_unary_op(op, v))
                    .collect();
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: types,
                    values: results,
                    value_id: ValueId::next(),
                })
            }
            Value::DynamicNDArray(d) => match op {
                "usub" => {
                    let out_dtype = d.dtype;
                    crate::ops::dyn_ndarray::binary::dyn_unary_op(
                        &mut self.builder,
                        d,
                        out_dtype,
                        |b, v| {
                            if matches!(v, Value::Float(_)) {
                                let zero = b.ir_constant_float(0.0);
                                b.ir_sub_f(&zero, v)
                            } else {
                                let zero = b.ir_constant_int(0);
                                b.ir_sub_i(&zero, v)
                            }
                        },
                    )
                }
                "not" => crate::ops::dyn_ndarray::binary::dyn_unary_op(
                    &mut self.builder,
                    d,
                    crate::types::NumberType::Integer,
                    |b, v| b.ir_logical_not(v),
                ),
                "invert" => crate::ops::dyn_ndarray::binary::dyn_unary_op(
                    &mut self.builder,
                    d,
                    crate::types::NumberType::Integer,
                    |b, v| b.ir_bit_not_i(v),
                ),
                "uadd" => operand.clone(),
                _ => panic!("Unknown unary operator for DynamicNDArray: {}", op),
            },
            Value::Complex { real, imag } => match op {
                "usub" => {
                    let zero = self.builder.ir_constant_float(0.0);
                    let neg_real = self.builder.ir_sub_f(&zero, &Value::Float(real.clone()));
                    let neg_imag = self.builder.ir_sub_f(&zero, &Value::Float(imag.clone()));
                    let nr = match neg_real {
                        Value::Float(s) => s,
                        _ => unreachable!(),
                    };
                    let ni = match neg_imag {
                        Value::Float(s) => s,
                        _ => unreachable!(),
                    };
                    Value::Complex { real: nr, imag: ni }
                }
                "uadd" => operand.clone(),
                _ => panic!("Unsupported unary operator on Complex: {}", op),
            },
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
                "invert" => {
                    if matches!(operand, Value::Float(_)) {
                        panic!("Bitwise NOT (~) requires an integer/boolean operand, got float");
                    }
                    // numpy semantics: `~bool` is logical NOT (single field op).
                    // `ir_bit_not_i` is a 64-bit decompose+recompose (~320 halo2
                    // rows). For Boolean operands route to `ir_logical_not` (one
                    // row). Mirrors `static_array::elementwise::unary_array`,
                    // which already handles this for StaticArray cells.
                    if matches!(operand, Value::Boolean(_)) {
                        self.builder.ir_logical_not(operand)
                    } else {
                        self.builder.ir_bit_not_i(operand)
                    }
                }
                _ => panic!("Unknown unary operator: {}", op),
            },
        }
    }

    pub(crate) fn visit_load(&mut self, n: &ASTLoad) -> Value {
        let value = self.ctx.get(&n.name).unwrap_or_else(|| {
            if let Some(reason) = self.ctx.unreachable_loop_binding_reason(&n.name) {
                panic!(
                    "Variable `{}` is not definitely bound after loop: {}",
                    n.name, reason
                );
            }
            panic!("Variable `{}` not found", n.name)
        });
        self.check_not_stale(&n.name, &value);
        value
    }

    pub(crate) fn visit_subscript(&mut self, n: &ASTSubscriptExp) -> Value {
        let val = self.visit(&n.val);
        // Evaluate slice indices by visiting them as AST nodes
        let slice_values = self.eval_slice_indices(&n.slicing);

        // P2 segarr-read-paths: native StaticArray dispatch BEFORE the
        // legacy boundary shim. Element / view / slice / multi-dim reads
        // operate directly on the segment without materialising a List.
        //
        // P5b: boolean-mask read on StaticArray dispatches to the native
        // mask path BEFORE the materialise-index shim below, so the cached
        // Boolean cells stay live.
        if matches!(val, Value::StaticArray { .. }) {
            if slice_values.len() == 1 {
                if let SliceIndex::Single(idx_val) = &slice_values[0] {
                    if matches!(idx_val, Value::StaticArray { .. })
                        && crate::helpers::static_array::mask::is_boolean_mask_static_array(
                            &mut self.builder,
                            idx_val,
                        )
                    {
                        if let Some(out) =
                            crate::helpers::static_array::mask::try_apply_boolean_mask_read(
                                &mut self.builder,
                                &val,
                                idx_val,
                            )
                        {
                            return out;
                        }
                    }
                }
            }
            // Fall back to converting StaticArray indices to lists for
            // the existing fancy-index path (see P4a notes).
            let slice_values: Vec<SliceIndex> = slice_values
                .into_iter()
                .map(|si| match si {
                    SliceIndex::Single(iv) if matches!(iv, Value::StaticArray { .. }) => {
                        SliceIndex::Single(crate::helpers::static_array::base::to_value_list(
                            &mut self.builder,
                            &iv,
                        ))
                    }
                    other => other,
                })
                .collect();

            // Fancy indexing (`arr[idx_array]` where `idx_array` is an
            // integer composite) is not handled by `static_array_subscript`
            // — its single-Single path only understands an integer scalar
            // or a dynamic integer. Detect the composite-index case here
            // and route through `try_advanced_index_static` on the
            // materialised List representation, matching the `Value::List`
            // arm below. Without this, a `List`-shaped index value falls
            // into the "dynamic 1-D" arm and produces garbage addresses.
            if slice_values.len() == 1 {
                if let SliceIndex::Single(idx_value) = &slice_values[0] {
                    if matches!(idx_value, Value::List(_) | Value::Tuple(_)) {
                        let val_list = crate::helpers::static_array::base::to_value_list(
                            &mut self.builder,
                            &val,
                        );
                        if let Value::List(data) | Value::Tuple(data) = &val_list {
                            match crate::helpers::ndarray::try_advanced_index_static(
                                data, idx_value,
                            ) {
                                Ok(Some(result)) => {
                                    self.relay_forall_eq_const_to_shaped_subscript_output(
                                        &val, &result,
                                    );
                                    return result;
                                }
                                Ok(Option::None) => {}
                                Err(msg) => {
                                    if let Some(result) =
                                        crate::helpers::ndarray::try_dynamic_boolean_mask_from_static(
                                            &mut self.builder,
                                            data,
                                            idx_value,
                                        )
                                    {
                                        return result;
                                    }
                                    if crate::helpers::static_array::mask::is_boolean_mask_static_array(
                                        &mut self.builder,
                                        idx_value,
                                    ) {
                                        if let Some(out) =
                                            crate::helpers::static_array::mask::try_apply_boolean_mask_read(
                                                &mut self.builder,
                                                &val,
                                                idx_value,
                                            )
                                        {
                                            return out;
                                        }
                                    }
                                    panic!("{}", msg);
                                }
                            }
                        }
                    }
                }
            }

            let array_identity = Self::static_array_identity(&val);
            if let Some(array_identity) = array_identity.as_ref() {
                if let Some(reused) = self.matching_loop_read_reuse(array_identity, &n.slicing) {
                    return reused;
                }
            }

            let use_preproved_index_bounds = match &val {
                Value::StaticArray { dtype, shape, .. }
                    if shape.len() == 1 && *dtype != crate::types::NumberType::Complex =>
                {
                    self.slice_has_preproved_loop_index_bounds_for_array(&n.slicing, shape[0])
                }
                _ => false,
            };
            let guard_condition = self
                .path_conditioned_loop_index_bound_context()
                .and_then(|ctx| ctx.guard_condition.clone());
            let out = if use_preproved_index_bounds {
                if let Some(guard_condition) = guard_condition {
                    crate::helpers::static_array::read::static_array_subscript_with_path_conditioned_preproved_index_bounds(
                        &mut self.builder,
                        &val,
                        &slice_values,
                        &guard_condition,
                    )
                } else {
                    crate::helpers::static_array::read::static_array_subscript_with_preproved_index_bounds(
                        &mut self.builder,
                        &val,
                        &slice_values,
                    )
                }
            } else {
                crate::helpers::static_array::read::static_array_subscript(
                    &mut self.builder,
                    &val,
                    &slice_values,
                )
            };
            // Group 5c (E-axis): content fact relay for gather of a
            // `forall_eq_const(arr, k)` input. Idempotent w.r.t. any inner
            // relays already fired by slice helpers. Scalar element reads
            // deliberately skip the relay: `forall_eq_const` is an array
            // content fact, and probing it for every dynamic scalar read is
            // expensive pre-optimizer work with no shaped output to inform.
            self.relay_forall_eq_const_to_shaped_subscript_output(&val, &out);
            self.record_subscript_copy(&val, &slice_values, &out);
            let index_key = self.loop_index_expr_key(&n.slicing);
            self.record_loop_guard_read_reuse(array_identity, index_key, &out);
            return out;
        }

        match &val {
            Value::List(data) | Value::Tuple(data) => {
                // Advanced indexing: `array[bool_mask]` or `array[idx_array]`
                // where the index itself is a static-shape composite. Try to
                // resolve at compile time; on shape/typing errors return Err
                // which we surface as a hard error to the user.
                if slice_values.len() == 1 {
                    if let SliceIndex::Single(idx_value) = &slice_values[0] {
                        if matches!(idx_value, Value::List(_) | Value::Tuple(_)) {
                            match crate::helpers::ndarray::try_advanced_index_static(
                                data, idx_value,
                            ) {
                                Ok(Some(result)) => {
                                    // Group 5c (E-axis): content fact relay for
                                    // gather (boolean mask or fancy index) of a
                                    // `forall_eq_const(arr, k)` input.
                                    if let (Some(in_vid), Some(out_vid)) =
                                        (val.value_id(), result.value_id())
                                    {
                                        crate::optim::resolver::relay_forall_eq_const_from_input(
                                            &mut self.builder,
                                            in_vid,
                                            out_vid,
                                        );
                                    }
                                    return result;
                                }
                                Ok(Option::None) => {} // not advanced indexing — fall through
                                Err(msg) => {
                                    if let Some(result) =
                                        crate::helpers::ndarray::try_dynamic_boolean_mask_from_static(
                                            &mut self.builder,
                                            data,
                                            idx_value,
                                        )
                                    {
                                        return result;
                                    }
                                    panic!("{}", msg);
                                }
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
                    return crate::helpers::ndarray::multidim_subscript(
                        &mut self.builder,
                        data,
                        &slice_values,
                    );
                }
                if slice_values.len() == 1 {
                    let idx_val = &slice_values[0];
                    match idx_val {
                        SliceIndex::Single(idx_value) => {
                            // Try to resolve statically
                            if let Some(idx) = idx_value.int_val() {
                                let idx = if idx < 0 {
                                    (data.values.len() as i64 + idx) as usize
                                } else {
                                    idx as usize
                                };
                                if idx < data.values.len() {
                                    data.values[idx].clone()
                                } else {
                                    Value::None
                                }
                            } else {
                                // Dynamic index: generate a chain of SelectI to pick the right element
                                crate::helpers::value_ops::dynamic_list_subscript(
                                    &mut self.builder,
                                    data,
                                    idx_value,
                                )
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
                    crate::helpers::ndarray::multidim_subscript(
                        &mut self.builder,
                        data,
                        &slice_values,
                    )
                }
            }
            Value::DynamicNDArray(d) => {
                let out = crate::helpers::array_ops::dyn_subscript(&mut self.builder, d, &slice_values);
                self.record_subscript_copy(&val, &slice_values, &out);
                out
            }
            Value::String(text) => self.string_subscript(&text.val, &slice_values),
            other => panic!("'{}' object is not subscriptable", other.zinnia_type()),
        }
    }

    /// `"text"[i]` with a compile-time index (strings are compile-time
    /// constants in circuits).
    fn string_subscript(&mut self, text: &str, indices: &[SliceIndex]) -> Value {
        let [SliceIndex::Single(index)] = indices else {
            panic!("string subscripts support a single integer index");
        };
        let chars: Vec<char> = text.chars().collect();
        let raw = index
            .int_val()
            .unwrap_or_else(|| panic!("string index must be a compile-time integer"));
        let position = if raw < 0 { raw + chars.len() as i64 } else { raw };
        let Some(ch) = usize::try_from(position).ok().and_then(|p| chars.get(p)) else {
            panic!("IndexError: string index out of range");
        };
        self.builder.ir_constant_str(ch.to_string())
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
    pub(crate) fn list_slice_range(
        &mut self,
        _val: &Value,
        data: &CompositeData,
        slice: &SliceIndex,
    ) -> Value {
        if let SliceIndex::Range(start, stop, step) = slice {
            let coords = crate::helpers::shape_arith::resolve_slice(
                data.values.len(),
                start.as_ref().and_then(|v| v.int_val()),
                stop.as_ref().and_then(|v| v.int_val()),
                step.as_ref().and_then(|v| v.int_val()),
            );
            let result_values: Vec<Value> = coords
                .iter()
                .map(|&i| data.values[i].clone())
                .collect();
            let types = result_values.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: result_values,
                value_id: ValueId::next(),
            })
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
                let inner = if matches!(inner, Value::StaticArray { .. }) {
                    crate::helpers::static_array::base::to_value_list(&mut self.builder, &inner)
                } else {
                    inner
                };
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
        // List literals stay `Value::List`: Python list semantics (`*` is
        // repetition, `+` is concatenation, `==` is sequence equality,
        // `.reverse()` mutates) are NOT numpy elementwise semantics, and a
        // segment-backed StaticArray can't tell those consumers apart —
        // the former eager conversion silently gave `[1, 2] * 4 == [4, 8]`.
        // Numpy constructors (`np.asarray`, `np.zeros`, ...) still build
        // StaticArrays directly; array-consuming ops materialise Lists on
        // demand through the existing `to_static_array` boundary shim.
        Value::List(CompositeData {
            elements_type: types,
            values,
            value_id: ValueId::next(),
        })
    }

    pub(crate) fn visit_parenthesis(&mut self, n: &ASTParenthesis) -> Value {
        let mut values = Vec::new();
        for v in &n.values {
            if let ASTNode::ASTStarredExpr(se) = v {
                // Unpack starred: (1, 2, *a) → flatten a's elements into the tuple
                let inner = self.visit(&se.value);
                let inner = if matches!(inner, Value::StaticArray { .. }) {
                    crate::helpers::static_array::base::to_value_list(&mut self.builder, &inner)
                } else {
                    inner
                };
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

            value_id: ValueId::next(),
        })
    }

    pub(crate) fn visit_generator_exp(&mut self, n: &ASTGeneratorExp) -> Value {
        if n.generators.is_empty() {
            return Value::List(CompositeData {
                elements_type: vec![],
                values: vec![],
                value_id: ValueId::next(),
            });
        }

        let mut result_values = Vec::new();
        let mut result_guards = Vec::new();
        let active = self.builder.ir_constant_bool(true);
        self.expand_generators(&n.generators, 0, &n.elt, &mut result_values, &mut result_guards, &active);

        // Runtime comprehension filters select a prefix of a bounded array;
        // they must not be discarded simply because their truth is unknown.
        if result_guards.iter().any(|g| g.bool_val() != Some(true)) {
            let types = result_values.iter().map(Value::zinnia_type).collect();
            let raw = Value::List(CompositeData::new(types, result_values));
            let promoted = crate::helpers::promote::promote_static_to_dynamic(&mut self.builder, &raw);
            let Value::DynamicNDArray(data) = promoted else { unreachable!() };
            assert_eq!(data.envelope.dims.len(), 1, "filtered comprehensions currently require scalar elements");
            let mask = Value::List(CompositeData::new(
                result_guards.iter().map(Value::zinnia_type).collect(), result_guards));
            return crate::ops::dyn_ndarray::memory_ops::dyn_filter(&mut self.builder, &data, &[mask]);
        }

        let types: Vec<ZinniaType> = result_values.iter().map(|v| v.zinnia_type()).collect();
        if n.kind == "list" {
            Value::List(CompositeData {
                elements_type: types,
                values: result_values,
                value_id: ValueId::next(),
            })
        } else {
            Value::Tuple(CompositeData {
                elements_type: types,
                values: result_values,
                value_id: ValueId::next(),
            })
        }
    }

    /// Expand comprehension generators `idx..` for every item, collecting
    /// the element values and, for each, the runtime condition under which
    /// Python produces it (`active`: the conjunction of the enclosing items'
    /// activity and filters). Filters, nested iterables and the element are
    /// evaluated under that condition, as Python evaluates them only for
    /// produced candidates.
    fn expand_generators(
        &mut self,
        generators: &[ASTGenerator],
        idx: usize,
        elt: &ASTNode,
        result: &mut Vec<Value>,
        guards: &mut Vec<Value>,
        active: &Value,
    ) {
        if idx >= generators.len() {
            let value = self.with_guard(active, |gen| gen.visit(elt));
            result.push(value);
            guards.push(active.clone());
            return;
        }

        let gen = &generators[idx];
        let items = self.with_guard(active, |this| this.iteration_items(&gen.iter_expr));

        for item in &items {
            self.do_recursive_assign(&gen.target, item.value.clone(), false);

            let mut guard = match &item.active {
                Some(item_active) => self.builder.ir_logical_and(active, item_active),
                None => active.clone(),
            };
            for if_expr in &gen.ifs {
                if guard.bool_val() == Some(false) {
                    break;
                }
                let cond = self.with_guard(&guard, |this| this.visit(if_expr));
                let cond = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &cond);
                guard = self.builder.ir_logical_and(&guard, &cond);
            }

            if guard.bool_val() != Some(false) {
                self.expand_generators(generators, idx + 1, elt, result, guards, &guard);
            }
        }
    }

    pub(crate) fn visit_cond_exp(&mut self, n: &ASTCondExp) -> Value {
        let cond = self.visit(&n.cond);
        let cond_bool = crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &cond);
        // Python evaluates only the selected arm. When the condition is
        // compile-time known, do not visit an unreachable arm (which may
        // legitimately contain an out-of-bounds neighbor access).
        if let Some(take_true) = cond_bool.bool_val() {
            return self.visit(if take_true { &n.t_expr } else { &n.f_expr });
        }
        // Each arm runs only on its side of the condition: writes, chip calls
        // and assertions inside an arm are predicated on it.
        let true_val = self.with_guard(&cond_bool, |this| this.visit(&n.t_expr));
        let not_cond = self.builder.ir_logical_not(&cond_bool);
        let false_val = self.with_guard(&not_cond, |this| this.visit(&n.f_expr));
        // `a if c else b` is the object `a` or `b` itself in Python.
        self.merge_binding(&cond_bool, &true_val, &false_val)
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

    pub(crate) fn do_recursive_assign(
        &mut self,
        target: &ASTNode,
        value: Value,
        conditional_select: bool,
    ) {
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
                    let merged = self.merge_binding(&cond, &value, &orig);
                    self.ctx.set(&t.name, merged);
                } else {
                    self.ctx.set(&t.name, value);
                }
            }
            ASTNode::ASTTupleAssignTarget(t) => {
                // P2 segarr: a StaticArray bound to an unpacking target
                // (e.g. `for a, b in arr2d:` where `arr2d` is a 2-D
                // segment-backed array) must be materialised to a List so
                // the unpacking machinery can iterate its elements.
                let value = if matches!(&value, Value::StaticArray { .. }) {
                    crate::helpers::static_array::base::to_value_list(&mut self.builder, &value)
                } else if matches!(&value, Value::DynamicNDArray(_)) {
                    dynamic_array_to_unpackable_list(&mut self.builder, &value, t.targets.len())
                } else {
                    value
                };
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
                            self.do_recursive_assign(
                                &t.targets[i],
                                data.values[i].clone(),
                                conditional_select,
                            );
                        }
                        // Assign starred target (collect middle into list)
                        let star_count = n_values - n_before - n_after;
                        let star_values: Vec<Value> =
                            data.values[n_before..n_before + star_count].to_vec();
                        let star_types = star_values.iter().map(|v| v.zinnia_type()).collect();
                        let star_list = Value::List(CompositeData {
                            elements_type: star_types,
                            values: star_values,
                            value_id: ValueId::next(),
                        });
                        // Assign to the starred target directly (star flag is on the target itself)
                        self.do_recursive_assign(&t.targets[si], star_list, conditional_select);
                        // Assign after-star targets
                        for i in 0..n_after {
                            self.do_recursive_assign(
                                &t.targets[si + 1 + i],
                                data.values[n_before + star_count + i].clone(),
                                conditional_select,
                            );
                        }
                    } else {
                        // No starred target — exact match required
                        if t.targets.len() != data.values.len() {
                            panic!(
                                "UnpackingError: cannot unpack {} values into {} targets",
                                data.values.len(),
                                t.targets.len()
                            );
                        }
                        for (i, tgt) in t.targets.iter().enumerate() {
                            if matches!(
                                tgt,
                                ASTNode::ASTTupleAssignTarget(_) | ASTNode::ASTListAssignTarget(_)
                            ) {
                                if !matches!(
                                    &data.values[i],
                                    Value::List(_) | Value::Tuple(_) | Value::DynamicNDArray(_)
                                ) {
                                    let tp = data.values[i].zinnia_type();
                                    panic!("TypeInferenceError: {} is not iterable", tp);
                                }
                            }
                            if Self::simple_name_target(tgt).is_some() {
                                if let Some(n) = data.values[i].int_val() {
                                    if let Some(vid) = data.values[i].value_id() {
                                        let fact = ContractTerm::eq(
                                            ContractTerm::Var(ContractVar::Value(vid)),
                                            ContractTerm::lit(n),
                                        );
                                        self.builder.facts.insert_for(vid, fact);
                                    }
                                    // Fall through to do_recursive_assign:
                                    // a direct ctx.set here would bypass
                                    // the conditional-branch merge for
                                    // names that exist outside the branch.
                                }
                            }
                            self.do_recursive_assign(
                                tgt,
                                data.values[i].clone(),
                                conditional_select,
                            );
                        }
                    }
                } else {
                    panic!("UnpackingError: cannot unpack non-iterable value");
                }
            }
            ASTNode::ASTListAssignTarget(t) => {
                // P2 segarr: same materialisation as the tuple case above.
                let value = if matches!(&value, Value::StaticArray { .. }) {
                    crate::helpers::static_array::base::to_value_list(&mut self.builder, &value)
                } else {
                    value
                };
                if let Value::Tuple(data) | Value::List(data) = &value {
                    for (i, tgt) in t.targets.iter().enumerate() {
                        if i < data.values.len() {
                            self.do_recursive_assign(
                                tgt,
                                data.values[i].clone(),
                                conditional_select,
                            );
                        }
                    }
                }
            }
            ASTNode::ASTSubscriptAssignTarget(t) => {
                if let Some(candidate) = self
                    .loop_read_reuse_contexts
                    .last()
                    .and_then(|ctx| ctx.candidate.clone())
                {
                    if self.target_writes_static_array_identity(target, &candidate.array) {
                        self.clear_loop_read_reuse();
                    }
                }
                match &*t.target {
                    ASTNode::ASTLoad(load) => {
                        let indices = self.eval_slice_indices(&t.slicing);
                        self.assign_to_subscript_of_name(&load.name, indices, value);
                    }
                    ASTNode::ASTSubscriptExp(inner) => {
                        self.assign_chained_subscript(inner, &t.slicing, value, conditional_select);
                    }
                    _ => panic!(
                        "assignment to a subscript of this expression is not supported; \
                         assign to a subscript of a variable (name[...] = value)"
                    ),
                }
            }
            _ => panic!("unsupported assignment target"),
        }
    }

    /// `a[i1, ...][j1, ...] = v` (any depth). When every index of the inner
    /// subscripts is a scalar integer, NumPy writes `a[i1, ..., j1, ...] = v`,
    /// so the assignment is flattened onto the root array, which handles
    /// private indices. Otherwise the inner subscript must be a view that
    /// shares the root's memory (a constant slice); a materialised
    /// temporary would silently drop the write, so it is rejected.
    fn assign_chained_subscript(
        &mut self,
        inner: &ASTSubscriptExp,
        outer: &ASTSlice,
        value: Value,
        conditional_select: bool,
    ) {
        // Bounded-dynamic roots are not flattened: their single-step writes
        // are handed off to the core (a write into `np.zeros((3, n))` is
        // lost), so a chained write into one stays rejected below.
        if let Some((root, mut indices)) = self.scalar_index_chain(inner) {
            if matches!(self.ctx.get(&root), Some(Value::StaticArray { .. })) {
                indices.extend(self.eval_slice_indices(outer));
                self.assign_to_subscript_of_name(&root, indices, value);
                return;
            }
        }
        let inner_val = self.visit_subscript(inner);
        let indices = self.eval_slice_indices(outer);
        let root_segment = chain_root(inner)
            .and_then(|root| self.ctx.get(root))
            .and_then(|root| super::detached_copies::array_segment(&root));
        let shares_root = super::detached_copies::array_segment(&inner_val)
            .is_some_and(|segment| Some(segment) == root_segment);
        let has_new_axis = indices.iter().any(|i| matches!(i, SliceIndex::NewAxis));
        if shares_root && matches!(inner_val, Value::StaticArray { .. }) && !has_new_axis {
            crate::helpers::static_array::write::static_array_setitem(
                &mut self.builder,
                &inner_val,
                &indices,
                &value,
            );
            self.note_array_write(&inner_val);
            return;
        }
        if matches!(inner_val, Value::List(_)) {
            if let Some(root) = chain_root(inner) {
                self.ensure_list_not_shared(root, &inner_val);
            }
            let updated = self.set_nested_value(inner_val, &indices, value);
            let target = ASTNode::ASTSubscriptAssignTarget(ASTSubscriptAssignTarget {
                target: inner.val.clone(),
                slicing: inner.slicing.clone(),
                star: false,
                dbg: inner.dbg.clone(),
            });
            self.do_recursive_assign(&target, updated, conditional_select);
            return;
        }
        panic!(
            "chained subscript assignment (a[i][j] = v) is unsupported here: the inner \
             subscript produces a temporary copy (a slice with a runtime bound or a negative \
             step, a new axis, or a dynamic array), so the write would be silently dropped. \
             Index in a single step (a[i, j] = v)."
        );
    }

    /// For `root[i...][j...]...` where every index is a scalar integer,
    /// return the root name and the concatenated indices. Indices are
    /// evaluated left to right, as Python does.
    fn scalar_index_chain(&mut self, node: &ASTSubscriptExp) -> Option<(String, Vec<SliceIndex>)> {
        let (root, mut indices) = match &*node.val {
            ASTNode::ASTLoad(load) => (load.name.clone(), Vec::new()),
            ASTNode::ASTSubscriptExp(inner) => self.scalar_index_chain(inner)?,
            _ => return None,
        };
        if node.slicing.data.iter().any(|d| d.is_array() || d.get("__class__").is_some_and(|c| {
            c == "ASTSliceEllipsis" || c == "ASTSliceNewAxis"
        })) {
            return None;
        }
        let evaluated = self.eval_slice_indices(&node.slicing);
        if !evaluated.iter().all(|i| matches!(i, SliceIndex::Single(Value::Integer(_)))) {
            return None;
        }
        indices.extend(evaluated);
        Some((root, indices))
    }

    /// `name[indices] = value` for any subscriptable value bound to `name`.
    fn assign_to_subscript_of_name(&mut self, var_name: &str, indices: Vec<SliceIndex>, value: Value) {
        let Some(current) = self.ctx.get(var_name) else {
            panic!("Variable `{}` not found", var_name);
        };
        let updated = match &current {
            Value::Tuple(_) => panic!("'tuple' object does not support item assignment"),
            Value::StaticArray { .. } => self.static_array_setitem(&current, &indices, value),
            Value::DynamicNDArray(data) => self.dynamic_array_setitem(data, &indices, &value),
            _ => {
                self.ensure_list_not_shared(var_name, &current);
                self.set_nested_value(current.clone(), &indices, value)
            }
        };
        self.note_array_write(&current);
        self.note_array_write(&updated);
        self.rebind_subscript_target(var_name, &current, updated);
    }

    /// Write into a segment-backed static array (element, multi-dim, slice,
    /// boolean-mask). Returns the array value to rebind.
    fn static_array_setitem(&mut self, current: &Value, indices: &[SliceIndex], value: Value) -> Value {
        // Boolean-mask write (`arr[mask] = v`): dispatch before the generic
        // setitem so the cached Boolean cells are seen, not materialised to
        // a List that collapses the bool/int distinction. Snapshot the
        // pre-write element bounds and deposit their widened union on the
        // post-write value so downstream reductions keep bounds.
        if let [SliceIndex::Single(mask)] = indices {
            if matches!(mask, Value::StaticArray { .. })
                && crate::helpers::static_array::mask::is_boolean_mask_static_array(
                    &mut self.builder,
                    mask,
                )
            {
                let pre = crate::helpers::static_array::mask::capture_mask_write_pre_bounds(
                    &mut self.builder,
                    current,
                    &value,
                );
                let updated = crate::helpers::static_array::mask::try_apply_boolean_mask_write(
                    &mut self.builder,
                    current,
                    mask,
                    &value,
                )
                .unwrap_or_else(|| {
                    crate::helpers::static_array::write::static_array_setitem(
                        &mut self.builder,
                        current,
                        indices,
                        &value,
                    )
                });
                crate::helpers::static_array::mask::deposit_mask_write_post(
                    &mut self.builder,
                    pre,
                    &updated,
                );
                return updated;
            }
        }
        if indices.iter().any(|i| matches!(i, SliceIndex::NewAxis)) {
            // A new axis on a setitem target is rare; materialise and use
            // the generic nested-value writer.
            let current_list =
                crate::helpers::static_array::base::to_value_list(&mut self.builder, current);
            return self.set_nested_value(current_list, indices, value);
        }
        crate::helpers::static_array::write::static_array_setitem(
            &mut self.builder,
            current,
            indices,
            &value,
        )
    }

    /// Write into a bounded-dynamic array (boolean mask, slice or element).
    fn dynamic_array_setitem(
        &mut self,
        data: &crate::types::DynamicNDArrayData,
        indices: &[SliceIndex],
        value: &Value,
    ) -> Value {
        if let [SliceIndex::Single(mask)] = indices {
            if crate::helpers::array_ops::is_boolean_mask_with(&mut self.builder, mask) {
                return crate::helpers::array_ops::dyn_setitem_mask(&mut self.builder, data, mask, value);
            }
        }
        if indices.iter().any(|i| matches!(i, SliceIndex::Range(..))) {
            return crate::helpers::array_ops::dyn_setitem_slice(&mut self.builder, data, indices, value);
        }
        crate::helpers::array_ops::dyn_setitem(&mut self.builder, data, indices, value)
    }

    // ── Built-in function helpers ────────────────────────────────────
}

/// The variable at the bottom of a subscript chain `name[...][...]`.
fn chain_root(node: &ASTSubscriptExp) -> Option<&str> {
    match &*node.val {
        ASTNode::ASTLoad(load) => Some(load.name.as_str()),
        ASTNode::ASTSubscriptExp(inner) => chain_root(inner),
        _ => None,
    }
}
