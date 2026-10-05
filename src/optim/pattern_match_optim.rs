use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct PatternMatchOptim;

impl PatternMatchOptim {
    fn optimize_ir(&self, builder: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if let Some(min_arity) = min_required_arity(ir) {
            if args.len() < min_arity {
                return builder.create_ir(ir, args);
            }
        }

        match ir {
            IR::LogicalAnd => self.optimize_logical_and(builder, ir, args),
            IR::LogicalOr => self.optimize_logical_or(builder, ir, args),
            IR::SelectB => self.optimize_select_b(builder, ir, args),
            IR::SelectI => self.optimize_select_i(builder, ir, args),
            IR::SelectF => self.optimize_select_f(builder, ir, args),
            IR::AddI => self.optimize_add_i(builder, ir, args),
            IR::AddF => self.optimize_add_f(builder, ir, args),
            IR::SubI => self.optimize_sub_i(builder, ir, args),
            IR::SubF => self.optimize_sub_f(builder, ir, args),
            IR::MulI => self.optimize_mul_i(builder, ir, args),
            IR::MulF => self.optimize_mul_f(builder, ir, args),
            IR::DivI => self.optimize_div_i(builder, ir, args),
            IR::DivF => self.optimize_div_f(builder, ir, args),
            IR::LtI | IR::LtF | IR::LtIBounded => {
                self.optimize_same_ptr_cmp(builder, ir, args, false)
            }
            IR::GtI | IR::GtF | IR::GtIBounded | IR::LteI | IR::LteF | IR::GteI | IR::GteF => {
                if is_same_ptr(args) {
                    let reflexive = matches!(ir, IR::LteI | IR::LteF | IR::GteI | IR::GteF);
                    return builder.ir_constant_bool(reflexive);
                }
                self.canonical_comparison(builder, ir, args)
            }
            IR::EqI | IR::EqF => self.optimize_same_ptr_cmp(builder, ir, args, true),
            IR::NeI | IR::NeF => self.optimize_same_ptr_cmp(builder, ir, args, false),
            _ => builder.create_ir(ir, args),
        }
    }

    fn optimize_logical_and(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].int_val();
        let rv = args[1].int_val();
        if lv == Some(0) {
            return b.ir_constant_bool(false);
        }
        if rv == Some(0) {
            return b.ir_constant_bool(false);
        }
        if lv.is_some() && lv != Some(0) {
            return args[1].clone();
        }
        if rv.is_some() && rv != Some(0) {
            return args[0].clone();
        }
        if let (Some(a), Some(c)) = (lv, rv) {
            return b.ir_constant_bool(a != 0 && c != 0);
        }
        b.create_ir(ir, args)
    }

    fn optimize_logical_or(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].int_val();
        let rv = args[1].int_val();
        if lv.is_some() && lv != Some(0) {
            return b.ir_constant_bool(true);
        }
        if rv.is_some() && rv != Some(0) {
            return b.ir_constant_bool(true);
        }
        if lv == Some(0) {
            return args[1].clone();
        }
        if rv == Some(0) {
            return args[0].clone();
        }
        if let (Some(a), Some(c)) = (lv, rv) {
            return b.ir_constant_bool(a != 0 || c != 0);
        }
        b.create_ir(ir, args)
    }

    fn optimize_select_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let cond = &args[0];
        let tv = &args[1];
        let fv = &args[2];
        // If both branches are boolean, delegate to select_b logic
        if (tv.bool_val().is_some() || fv.bool_val().is_some())
            && matches!(tv, Value::Boolean(_))
            && matches!(fv, Value::Boolean(_))
        {
            return self.optimize_select_b(b, ir, args);
        }
        if tv.int_val() == fv.int_val() && fv.int_val().is_some() {
            return b.ir_constant_int(tv.int_val().unwrap());
        }
        if tv.stmt_id() == fv.stmt_id() && tv.stmt_id().is_some() {
            return tv.clone();
        }
        if tv.int_val() == Some(1) && fv.int_val() == Some(0) {
            return cond.clone();
        }
        let cv = cond.bool_val().or_else(|| cond.int_val().map(|v| v != 0));
        if let Some(true) = cv {
            if tv.int_val().is_some() {
                return b.ir_constant_int(tv.int_val().unwrap());
            }
            return tv.clone();
        }
        if let Some(false) = cv {
            if fv.int_val().is_some() {
                return b.ir_constant_int(fv.int_val().unwrap());
            }
            return fv.clone();
        }
        b.create_ir(ir, args)
    }

    fn optimize_select_b(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let cond = &args[0];
        let tv = &args[1];
        let fv = &args[2];
        if tv.bool_val() == fv.bool_val() && fv.bool_val().is_some() {
            return b.ir_constant_bool(tv.bool_val().unwrap());
        }
        if tv.stmt_id() == fv.stmt_id() && tv.stmt_id().is_some() {
            return tv.clone();
        }
        if tv.bool_val() == Some(true) && fv.bool_val() == Some(false) {
            return cond.clone();
        }
        let cv = cond.bool_val().or_else(|| cond.int_val().map(|v| v != 0));
        if let Some(true) = cv {
            if tv.bool_val().is_some() {
                return b.ir_constant_bool(tv.bool_val().unwrap());
            }
            return tv.clone();
        }
        if let Some(false) = cv {
            if fv.bool_val().is_some() {
                return b.ir_constant_bool(fv.bool_val().unwrap());
            }
            return fv.clone();
        }
        if fv.bool_val() == Some(false) {
            return b.ir_logical_and(cond, tv);
        }
        if tv.bool_val() == Some(true) {
            return b.ir_logical_or(cond, fv);
        }
        b.create_ir(ir, args)
    }

    fn optimize_select_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let cond = &args[0];
        let tv = &args[1];
        let fv = &args[2];
        if tv.float_val() == fv.float_val() && fv.float_val().is_some() {
            return b.ir_constant_float(tv.float_val().unwrap());
        }
        if tv.stmt_id() == fv.stmt_id() && tv.stmt_id().is_some() {
            return tv.clone();
        }
        let cv = cond.bool_val().or_else(|| cond.int_val().map(|v| v != 0));
        if let Some(true) = cv {
            if tv.float_val().is_some() {
                return b.ir_constant_float(tv.float_val().unwrap());
            }
            return tv.clone();
        }
        if let Some(false) = cv {
            if fv.float_val().is_some() {
                return b.ir_constant_float(fv.float_val().unwrap());
            }
            return fv.clone();
        }
        b.create_ir(ir, args)
    }

    fn optimize_add_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].int_val();
        let rv = args[1].int_val();
        if lv == Some(0) {
            return args[1].clone();
        }
        if rv == Some(0) {
            return args[0].clone();
        }
        if let (Some(a), Some(c)) = (lv, rv) {
            return b.ir_constant_int(a + c);
        }
        b.create_ir(ir, args)
    }

    fn optimize_add_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].float_val();
        let rv = args[1].float_val();
        if lv == Some(0.0) {
            return args[1].clone();
        }
        if rv == Some(0.0) {
            return args[0].clone();
        }
        if let (Some(a), Some(c)) = (lv, rv) {
            return b.ir_constant_float(a + c);
        }
        b.create_ir(ir, args)
    }

    fn optimize_sub_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[1].int_val() == Some(0) {
            return args[0].clone();
        }
        if let (Some(a), Some(c)) = (args[0].int_val(), args[1].int_val()) {
            return b.ir_constant_int(a - c);
        }
        b.create_ir(ir, args)
    }

    fn optimize_sub_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[1].float_val() == Some(0.0) {
            return args[0].clone();
        }
        if let (Some(a), Some(c)) = (args[0].float_val(), args[1].float_val()) {
            return b.ir_constant_float(a - c);
        }
        b.create_ir(ir, args)
    }

    fn optimize_mul_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].int_val();
        let rv = args[1].int_val();
        if lv == Some(1) {
            return args[1].clone();
        }
        if rv == Some(1) {
            return args[0].clone();
        }
        if let (Some(a), Some(c)) = (lv, rv) {
            return b.ir_constant_int(a * c);
        }
        b.create_ir(ir, args)
    }

    fn optimize_mul_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let lv = args[0].float_val();
        let rv = args[1].float_val();
        if lv == Some(1.0) {
            return args[1].clone();
        }
        if rv == Some(1.0) {
            return args[0].clone();
        }
        if let (Some(a), Some(c)) = (lv, rv) {
            return b.ir_constant_float(a * c);
        }
        b.create_ir(ir, args)
    }

    fn optimize_div_i(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[1].int_val() == Some(1) {
            return args[0].clone();
        }
        b.create_ir(ir, args)
    }

    fn optimize_div_f(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        if args[1].float_val() == Some(1.0) {
            return args[0].clone();
        }
        b.create_ir(ir, args)
    }

    /// Every ordering comparison as a strict less-than, negated where needed:
    /// `a > b` is `b < a`, `a >= b` is `not (a < b)` and `a <= b` is
    /// `not (b < a)` (total orders on the signed window and on the
    /// non-negative bounded operands alike). Mirrored and complementary
    /// comparisons of the same operands (`x < t` next to `x >= t`) then
    /// become the same `Lt` statement, which `DuplicateCodeElimination`
    /// computes once; every backend lowers `>=`/`<=` as a negated `<`
    /// anyway.
    fn canonical_comparison(&self, b: &mut IRBuilder, ir: &IR, args: &[Value]) -> Value {
        let (lhs, rhs) = (&args[0], &args[1]);
        let less = |b: &mut IRBuilder, strict: &IR, x: &Value, y: &Value| {
            b.create_ir(strict, &[x.clone(), y.clone()])
        };
        match ir {
            IR::GtI => less(b, &IR::LtI, rhs, lhs),
            IR::GtF => less(b, &IR::LtF, rhs, lhs),
            IR::GtIBounded => less(b, &IR::LtIBounded, rhs, lhs),
            IR::GteI | IR::GteF => {
                let strict = if matches!(ir, IR::GteI) { IR::LtI } else { IR::LtF };
                let lt = less(b, &strict, lhs, rhs);
                b.create_ir(&IR::LogicalNot, &[lt])
            }
            IR::LteI | IR::LteF => {
                let strict = if matches!(ir, IR::LteI) { IR::LtI } else { IR::LtF };
                let gt = less(b, &strict, rhs, lhs);
                b.create_ir(&IR::LogicalNot, &[gt])
            }
            _ => unreachable!("not an ordering comparison: {:?}", ir),
        }
    }

    /// Optimize comparisons where both operands are the same pointer.
    /// For `a == a` → true, `a != a` → false, `a < a` → false, `a <= a` → true.
    fn optimize_same_ptr_cmp(
        &self,
        b: &mut IRBuilder,
        ir: &IR,
        args: &[Value],
        result_if_same: bool,
    ) -> Value {
        if args[0].stmt_id() == args[1].stmt_id() && args[0].stmt_id().is_some() {
            return b.ir_constant_bool(result_if_same);
        }
        b.create_ir(ir, args)
    }
}

fn is_same_ptr(args: &[Value]) -> bool {
    args[0].stmt_id() == args[1].stmt_id() && args[0].stmt_id().is_some()
}

fn min_required_arity(ir: &IR) -> Option<usize> {
    Some(match ir {
        IR::LogicalNot => 1,
        IR::LogicalAnd
        | IR::LogicalOr
        | IR::AddI
        | IR::AddF
        | IR::SumF { .. }
        | IR::SubI
        | IR::SubF
        | IR::MulI
        | IR::MulF
        | IR::DivI
        | IR::DivF
        | IR::LtI
        | IR::GtI
        | IR::LtIBounded
        | IR::GtIBounded
        | IR::LtF
        | IR::GtF
        | IR::LteI
        | IR::GteI
        | IR::LteF
        | IR::GteF
        | IR::EqI
        | IR::EqF
        | IR::NeI
        | IR::NeF => 2,
        IR::SelectB | IR::SelectI | IR::SelectF => 3,
        _ => 0,
    })
}

impl IRPass for PatternMatchOptim {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();

        for stmt in ir_graph.get_topological_order(false) {
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();
            let val = self.optimize_ir(&mut builder, &stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, val);
        }

        let mut new_graph = builder.export_ir_graph();
        // P0 seam: signal mutation to the resolver. No-op today via
        // StaticOnlyResolver; P1 uses this for cache invalidation.
        new_graph.resolver_mut().on_ir_mutated(&[]);
        new_graph
    }
}
