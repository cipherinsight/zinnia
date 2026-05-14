//! Tests for the strategy-selection framework (`optim::strategy`).
//!
//! Two flavours of coverage:
//!
//! 1. Direct unit tests for `dispatch_strategy`'s wiring: gated-on-Proved,
//!    fallthrough-on-Unknown, first-declared-wins.
//! 2. A regression test for the `dyn_aggregate_all` refactor: the
//!    is_sorted short-circuit still fires when the fact is planted.

#[cfg(test)]
mod tests {
    use crate::builder::IRBuilder;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::optim::{dispatch_strategy, CostHint, OpStrategy, OpStrategySet};
    use crate::types::ValueId;

    /// Build a `ContractTerm` that asks: "is the value at `vid` >= 0?"
    /// Trivially Proved when a `vid >= 0` fact is planted on `vid`.
    fn ge_zero(vid: ValueId) -> ContractTerm {
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        }
    }

    fn ge_one(vid: ValueId) -> ContractTerm {
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitInt(1)),
        }
    }

    fn fast_lowering(_b: &mut IRBuilder, _inputs: &ValueId) -> &'static str {
        "fast"
    }

    fn fast2_lowering(_b: &mut IRBuilder, _inputs: &ValueId) -> &'static str {
        "fast2"
    }

    fn slow_lowering(_b: &mut IRBuilder, _inputs: &ValueId) -> &'static str {
        "slow"
    }

    #[test]
    fn gated_strategy_fires_when_precondition_proved() {
        // Plant `vid >= 0`. Gated strategy's precondition is `vid >= 0` →
        // Proved → fast path wins.
        let mut b = IRBuilder::new();
        let vid = ValueId::next();
        b.facts.insert_for(vid, ge_zero(vid));

        let set: OpStrategySet<ValueId, &'static str> = OpStrategySet {
            strategies: vec![OpStrategy {
                name: "gated",
                precondition: ge_zero(vid),
                cost_hint: CostHint::O1,
                lower: fast_lowering,
            }],
            default: slow_lowering,
        };

        let out = dispatch_strategy(&mut b, "test_op", &vid, &set);
        assert_eq!(out, "fast", "gated strategy should fire on Proved");
    }

    #[test]
    fn default_fires_when_precondition_unknown() {
        // No facts planted. `prove(vid >= 0)` is Unknown → gated skipped,
        // default wins.
        let mut b = IRBuilder::new();
        let vid = ValueId::next();

        let set: OpStrategySet<ValueId, &'static str> = OpStrategySet {
            strategies: vec![OpStrategy {
                name: "gated",
                precondition: ge_zero(vid),
                cost_hint: CostHint::O1,
                lower: fast_lowering,
            }],
            default: slow_lowering,
        };

        let out = dispatch_strategy(&mut b, "test_op", &vid, &set);
        assert_eq!(out, "slow", "default must fire when precondition is Unknown");
    }

    #[test]
    fn first_declared_wins_when_multiple_strategies_proved() {
        // Plant `vid >= 1` — both `vid >= 0` and `vid >= 1` become Proved.
        // Declared order: fast first, fast2 second. fast wins.
        let mut b = IRBuilder::new();
        let vid = ValueId::next();
        b.facts.insert_for(vid, ge_one(vid));

        let set: OpStrategySet<ValueId, &'static str> = OpStrategySet {
            strategies: vec![
                OpStrategy {
                    name: "gated-1",
                    precondition: ge_zero(vid),
                    cost_hint: CostHint::O1,
                    lower: fast_lowering,
                },
                OpStrategy {
                    name: "gated-2",
                    precondition: ge_one(vid),
                    cost_hint: CostHint::O1,
                    lower: fast2_lowering,
                },
            ],
            default: slow_lowering,
        };

        let out = dispatch_strategy(&mut b, "test_op", &vid, &set);
        assert_eq!(
            out, "fast",
            "first-declared strategy must win when multiple preconditions are Proved"
        );
    }

    /// Regression test for the `dyn_aggregate_all` refactor: with
    /// `is_sorted(arr_vid)` planted, calling `dyn_aggregate_all(Max)` must
    /// still take the boundary-read fast path, producing a single
    /// `IR::ReadMemory` as the result (no select-chain).
    #[test]
    fn refactored_max_on_sorted_still_short_circuits() {
        use crate::ir_defs::IR;
        use crate::ops::dyn_ndarray::aggregation::dyn_aggregate_all;
        use crate::ops::dyn_ndarray::constructors::dyn_from_values_with_active_nd;
        use crate::ops::dyn_ndarray::DynAggKind;
        use crate::types::{NumberType, ScalarValue, Value};

        let mut b = IRBuilder::new();
        let values: Vec<ScalarValue<i64>> = (1..=5)
            .map(|v| ScalarValue::new(Some(v), None))
            .collect();
        let runtime_shape = vec![ScalarValue::new(Some(5), None)];
        let runtime_length = b.ir_constant_int(5);
        let arr_val = dyn_from_values_with_active_nd(
            &mut b,
            values,
            vec![5],
            runtime_shape,
            runtime_length,
            NumberType::Integer,
        );
        let data = match arr_val {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        };

        // Plant is_sorted(arr_vid) — same anchor path that user @requires
        // would deposit it on.
        let is_sorted_fact = ContractTerm::PredicateApp {
            kind: "is_sorted".to_string(),
            args: vec![ContractTerm::Var(ContractVar::Value(data.value_id))],
        };
        b.facts.insert_for(data.value_id, is_sorted_fact);

        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Max);
        let target = result.stmt_id().expect("max result has stmt_id");
        let final_stmt = b
            .stmts
            .iter()
            .find(|s| s.stmt_id == target)
            .expect("target stmt present");
        assert!(
            matches!(final_stmt.ir, IR::ReadMemory { .. }),
            "refactored max-on-sorted should still emit a single ReadMemory, got {:?}",
            final_stmt.ir
        );

        let read_count = b
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::ReadMemory { .. }))
            .count();
        assert_eq!(
            read_count, 1,
            "refactored max-on-sorted should emit exactly one ReadMemory"
        );

        let select_count = b
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::SelectI))
            .count();
        assert_eq!(
            select_count, 0,
            "refactored max-on-sorted must not emit any SelectI (no select-chain)"
        );
    }

    // ── Group 3d: builtin_reduce sum/prod and np_mean strategy dispatch ──
    //
    // These tests plant a `forall_eq_const(arr_vid, k)` fact on a
    // `Value::List` composite (which carries a `value_id` per
    // `compiler.value-list-tuple-value-id`) and verify the strategy
    // selection logic in `builtin_reduce` / `np_mean` short-circuits to
    // the constant lowering rather than emitting an N-way sweep.

    use crate::helpers::ndarray::builtin_reduce;
    use crate::ir_defs::IR;
    use crate::ops::static_ndarray_ops::np_mean;
    use crate::types::{CompositeData, Value};
    use std::collections::HashMap;

    fn make_int_list(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let leaves: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        let types = leaves.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData::new(types, leaves))
    }

    fn plant_forall_eq_const(b: &mut IRBuilder, vid: ValueId, k: i64) {
        let fact = ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(vid)),
                ContractTerm::LitInt(k),
            ],
        };
        b.facts.insert_for(vid, fact);
    }

    fn count_ir_kind<F: Fn(&IR) -> bool>(b: &IRBuilder, pred: F) -> usize {
        b.stmts.iter().filter(|s| pred(&s.ir)).count()
    }

    #[test]
    fn sum_strategy_zero_fires_on_forall_eq_zero() {
        // Plant `forall_eq_const(list_vid, 0)`. builtin_reduce("sum") must
        // resolve to a constant 0 instead of an N-1 chain of AddI ops.
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[7, 8, 9, 10]); // values irrelevant
        let vid = lst.value_id().expect("list has value_id");
        plant_forall_eq_const(&mut b, vid, 0);

        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let result = builtin_reduce(&mut b, "sum", &lst);
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(result.int_val(), Some(0), "sum on forall_eq_const(0) must be 0");
        assert_eq!(
            post_adds, pre_adds,
            "sum-on-zeros strategy must not emit any AddI ops"
        );
    }

    #[test]
    fn sum_strategy_one_fires_on_forall_eq_one() {
        // Plant `forall_eq_const(list_vid, 1)`. builtin_reduce("sum") must
        // resolve to a constant N (here, 4) — no AddI emitted.
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[3, 3, 3, 3]);
        let vid = lst.value_id().expect("list has value_id");
        plant_forall_eq_const(&mut b, vid, 1);

        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let result = builtin_reduce(&mut b, "sum", &lst);
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(result.int_val(), Some(4), "sum on forall_eq_const(1) must be N");
        assert_eq!(
            post_adds, pre_adds,
            "sum-on-ones strategy must not emit any AddI ops"
        );
    }

    #[test]
    fn sum_strategy_default_fires_when_no_fact() {
        // No fact planted. builtin_reduce("sum") must fall through to the
        // generic sweep, emitting N-1 (= 3) AddI statements.
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[1, 2, 3, 4]);

        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let _ = builtin_reduce(&mut b, "sum", &lst);
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(
            post_adds - pre_adds,
            3,
            "default sum sweep must emit N-1 AddI ops (got {})",
            post_adds - pre_adds
        );
    }

    #[test]
    fn prod_strategy_zero_fires_on_forall_eq_zero() {
        // Plant `forall_eq_const(list_vid, 0)`. builtin_reduce("prod") must
        // resolve to a constant 0 — no MulI emitted.
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[7, 8, 9, 10]);
        let vid = lst.value_id().expect("list has value_id");
        plant_forall_eq_const(&mut b, vid, 0);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let result = builtin_reduce(&mut b, "prod", &lst);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));

        assert_eq!(result.int_val(), Some(0), "prod on forall_eq_const(0) must be 0");
        assert_eq!(
            post_muls, pre_muls,
            "prod-on-zeros strategy must not emit any MulI ops"
        );
    }

    #[test]
    fn prod_strategy_one_fires_on_forall_eq_one() {
        // Plant `forall_eq_const(list_vid, 1)`. builtin_reduce("prod") must
        // resolve to constant 1 — no MulI emitted.
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[7, 8, 9, 10]);
        let vid = lst.value_id().expect("list has value_id");
        plant_forall_eq_const(&mut b, vid, 1);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let result = builtin_reduce(&mut b, "prod", &lst);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));

        assert_eq!(result.int_val(), Some(1), "prod on forall_eq_const(1) must be 1");
        assert_eq!(
            post_muls, pre_muls,
            "prod-on-ones strategy must not emit any MulI ops"
        );
    }

    #[test]
    fn prod_strategy_default_fires_when_no_fact() {
        // Default sweep emits N-1 (= 3) MulI ops.
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[1, 2, 3, 4]);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let _ = builtin_reduce(&mut b, "prod", &lst);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));

        assert_eq!(
            post_muls - pre_muls,
            3,
            "default prod sweep must emit N-1 MulI ops (got {})",
            post_muls - pre_muls
        );
    }

    #[test]
    fn mean_strategy_zero_fires_on_forall_eq_zero() {
        // Plant `forall_eq_const(list_vid, 0)`. np_mean (no-axis) must
        // resolve to constant 0.0 — no AddI / DivF emitted.
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[7, 8, 9, 10]);
        let vid = lst.value_id().expect("list has value_id");
        plant_forall_eq_const(&mut b, vid, 0);

        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let pre_divs = count_ir_kind(&b, |ir| matches!(ir, IR::DivF));
        let result = np_mean(&mut b, &[lst.clone()], &HashMap::new());
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let post_divs = count_ir_kind(&b, |ir| matches!(ir, IR::DivF));

        assert_eq!(
            result.float_val(),
            Some(0.0),
            "mean on forall_eq_const(0) must be 0.0"
        );
        assert_eq!(
            post_adds, pre_adds,
            "mean-on-zeros strategy must not emit any AddI ops"
        );
        assert_eq!(
            post_divs, pre_divs,
            "mean-on-zeros strategy must not emit any DivF ops"
        );
    }

    #[test]
    fn mean_strategy_one_fires_on_forall_eq_one() {
        // Plant `forall_eq_const(list_vid, 1)`. np_mean (no-axis) must
        // resolve to constant 1.0 — no AddI / DivF emitted.
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[7, 8, 9, 10]);
        let vid = lst.value_id().expect("list has value_id");
        plant_forall_eq_const(&mut b, vid, 1);

        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let pre_divs = count_ir_kind(&b, |ir| matches!(ir, IR::DivF));
        let result = np_mean(&mut b, &[lst.clone()], &HashMap::new());
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let post_divs = count_ir_kind(&b, |ir| matches!(ir, IR::DivF));

        assert_eq!(
            result.float_val(),
            Some(1.0),
            "mean on forall_eq_const(1) must be 1.0"
        );
        assert_eq!(
            post_adds, pre_adds,
            "mean-on-ones strategy must not emit any AddI ops"
        );
        assert_eq!(
            post_divs, pre_divs,
            "mean-on-ones strategy must not emit any DivF ops"
        );
    }

    #[test]
    fn mean_strategy_default_fires_when_no_fact() {
        // No fact planted. np_mean falls through to the generic lowering,
        // which goes through builtin_reduce("sum") then DivF — at least
        // one DivF must appear (the final divide-by-N).
        let mut b = IRBuilder::new();
        let lst = make_int_list(&mut b, &[1, 2, 3, 4]);

        let pre_divs = count_ir_kind(&b, |ir| matches!(ir, IR::DivF));
        let _ = np_mean(&mut b, &[lst.clone()], &HashMap::new());
        let post_divs = count_ir_kind(&b, |ir| matches!(ir, IR::DivF));

        assert!(
            post_divs > pre_divs,
            "default mean lowering must emit at least one DivF (got delta {})",
            post_divs - pre_divs
        );
    }

    // ── Group 5e: np_where arm-elision strategy dispatch ────────────────
    //
    // Plant `forall_eq_const(cond_vid, k)` on a boolean cond list and
    // verify np_where short-circuits to x (k=1) or y (k=0). Without a
    // fact (or with a non-{0,1} constant), the default per-element
    // select fires, emitting SelectI ops.

    use crate::ops::static_ndarray_ops::np_where;

    fn make_bool_list(b: &mut IRBuilder, vals: &[bool]) -> Value {
        let leaves: Vec<Value> = vals.iter().map(|v| b.ir_constant_bool(*v)).collect();
        let types = leaves.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData::new(types, leaves))
    }

    #[test]
    fn where_strategy_true_returns_x() {
        // Plant `forall_eq_const(cond_vid, 1)`. np_where(cond, x, y) must
        // return `x_b` (the broadcast x), not run a per-element select —
        // no SelectI ops emitted.
        let mut b = IRBuilder::new();
        let cond = make_bool_list(&mut b, &[true, true, true, true]);
        let x = make_int_list(&mut b, &[10, 20, 30, 40]);
        let y = make_int_list(&mut b, &[1, 2, 3, 4]);
        let cond_vid = cond.value_id().expect("bool list has value_id");
        plant_forall_eq_const(&mut b, cond_vid, 1);

        let pre_selects = count_ir_kind(&b, |ir| matches!(ir, IR::SelectI));
        let result = np_where(&mut b, &[cond.clone(), x.clone(), y.clone()]);
        let post_selects = count_ir_kind(&b, |ir| matches!(ir, IR::SelectI));

        assert_eq!(
            post_selects, pre_selects,
            "where-on-all-true strategy must not emit any SelectI ops"
        );
        // Result content should match x (broadcast). Compare flattened ints.
        let result_flat = crate::helpers::composite::flatten_composite(&result);
        let x_flat = crate::helpers::composite::flatten_composite(&x);
        let result_ints: Vec<Option<i64>> = result_flat.iter().map(|v| v.int_val()).collect();
        let x_ints: Vec<Option<i64>> = x_flat.iter().map(|v| v.int_val()).collect();
        assert_eq!(result_ints, x_ints, "result must equal x when cond all-true");
    }

    #[test]
    fn where_strategy_false_returns_y() {
        // Plant `forall_eq_const(cond_vid, 0)`. np_where(cond, x, y) must
        // return `y_b` (the broadcast y), no SelectI ops emitted.
        let mut b = IRBuilder::new();
        let cond = make_bool_list(&mut b, &[false, false, false, false]);
        let x = make_int_list(&mut b, &[10, 20, 30, 40]);
        let y = make_int_list(&mut b, &[1, 2, 3, 4]);
        let cond_vid = cond.value_id().expect("bool list has value_id");
        plant_forall_eq_const(&mut b, cond_vid, 0);

        let pre_selects = count_ir_kind(&b, |ir| matches!(ir, IR::SelectI));
        let result = np_where(&mut b, &[cond.clone(), x.clone(), y.clone()]);
        let post_selects = count_ir_kind(&b, |ir| matches!(ir, IR::SelectI));

        assert_eq!(
            post_selects, pre_selects,
            "where-on-all-false strategy must not emit any SelectI ops"
        );
        let result_flat = crate::helpers::composite::flatten_composite(&result);
        let y_flat = crate::helpers::composite::flatten_composite(&y);
        let result_ints: Vec<Option<i64>> = result_flat.iter().map(|v| v.int_val()).collect();
        let y_ints: Vec<Option<i64>> = y_flat.iter().map(|v| v.int_val()).collect();
        assert_eq!(result_ints, y_ints, "result must equal y when cond all-false");
    }

    #[test]
    fn where_strategy_default_fires_when_no_fact() {
        // No fact planted. np_where must fall through to the generic
        // per-element select, emitting N (= 4) SelectI statements.
        let mut b = IRBuilder::new();
        let cond = make_bool_list(&mut b, &[true, false, true, false]);
        let x = make_int_list(&mut b, &[10, 20, 30, 40]);
        let y = make_int_list(&mut b, &[1, 2, 3, 4]);

        let pre_selects = count_ir_kind(&b, |ir| matches!(ir, IR::SelectI));
        let _ = np_where(&mut b, &[cond, x, y]);
        let post_selects = count_ir_kind(&b, |ir| matches!(ir, IR::SelectI));

        assert_eq!(
            post_selects - pre_selects,
            4,
            "default where lowering must emit N SelectI ops (got {})",
            post_selects - pre_selects
        );
    }

    #[test]
    fn where_strategy_mixed_fact_falls_through() {
        // Plant `forall_eq_const(cond_vid, 2)` — a constant that is
        // neither 0 nor 1. Neither gated strategy's precondition is
        // Proved, so the default per-element select must run.
        let mut b = IRBuilder::new();
        let cond = make_bool_list(&mut b, &[true, false, true, false]);
        let x = make_int_list(&mut b, &[10, 20, 30, 40]);
        let y = make_int_list(&mut b, &[1, 2, 3, 4]);
        let cond_vid = cond.value_id().expect("bool list has value_id");
        plant_forall_eq_const(&mut b, cond_vid, 2);

        let pre_selects = count_ir_kind(&b, |ir| matches!(ir, IR::SelectI));
        let _ = np_where(&mut b, &[cond, x, y]);
        let post_selects = count_ir_kind(&b, |ir| matches!(ir, IR::SelectI));

        assert_eq!(
            post_selects - pre_selects,
            4,
            "non-{{0,1}} forall_eq_const must fall through to default select-chain (got {} SelectI)",
            post_selects - pre_selects
        );
    }

    // ── Group 8a: matmul zero short-circuit strategy dispatch ───────────
    //
    // Plant `forall_eq_const(lhs_vid, 0)` (or rhs) on a 2D or 1D composite
    // operand and verify `matmul` returns a zeros composite without
    // emitting any MulI/AddI ops. Without a fact, the default generic
    // matmul body runs and emits the usual O(N^3) MulI/AddI ops.

    use crate::ops::static_ndarray_ops::matmul;

    fn make_int_matrix(b: &mut IRBuilder, rows: usize, cols: usize) -> Value {
        let row_vals: Vec<Value> = (0..rows)
            .map(|i| {
                let leaves: Vec<Value> = (0..cols)
                    .map(|j| b.ir_constant_int((i * cols + j) as i64 + 1))
                    .collect();
                let types = leaves.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData::new(types, leaves))
            })
            .collect();
        let types = row_vals.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData::new(types, row_vals))
    }

    #[test]
    fn matmul_zero_lhs_returns_zeros() {
        // Plant `forall_eq_const(lhs_vid, 0)` on a 2x3 matrix. matmul with
        // a 3x4 rhs must short-circuit to a 2x4 zeros composite — no
        // MulI / AddI emitted, and the result must flatten to all zeros.
        let mut b = IRBuilder::new();
        let lhs = make_int_matrix(&mut b, 2, 3);
        let rhs = make_int_matrix(&mut b, 3, 4);
        let lhs_vid = lhs.value_id().expect("matrix has value_id");
        plant_forall_eq_const(&mut b, lhs_vid, 0);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let result = matmul(&mut b, &lhs, &rhs);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(
            post_muls, pre_muls,
            "matmul on forall_eq_const(lhs, 0) must not emit any MulI ops"
        );
        assert_eq!(
            post_adds, pre_adds,
            "matmul on forall_eq_const(lhs, 0) must not emit any AddI ops"
        );
        let flat = crate::helpers::composite::flatten_composite(&result);
        assert_eq!(flat.len(), 2 * 4, "output shape must be (2, 4) → 8 leaves");
        for v in &flat {
            assert_eq!(v.int_val(), Some(0), "every output leaf must be 0");
        }
    }

    #[test]
    fn matmul_zero_rhs_returns_zeros() {
        // Plant `forall_eq_const(rhs_vid, 0)` on a 3x4 matrix. matmul with
        // a 2x3 lhs must short-circuit to a 2x4 zeros composite — no
        // MulI / AddI emitted.
        let mut b = IRBuilder::new();
        let lhs = make_int_matrix(&mut b, 2, 3);
        let rhs = make_int_matrix(&mut b, 3, 4);
        let rhs_vid = rhs.value_id().expect("matrix has value_id");
        plant_forall_eq_const(&mut b, rhs_vid, 0);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let result = matmul(&mut b, &lhs, &rhs);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(
            post_muls, pre_muls,
            "matmul on forall_eq_const(rhs, 0) must not emit any MulI ops"
        );
        assert_eq!(
            post_adds, pre_adds,
            "matmul on forall_eq_const(rhs, 0) must not emit any AddI ops"
        );
        let flat = crate::helpers::composite::flatten_composite(&result);
        assert_eq!(flat.len(), 2 * 4);
        for v in &flat {
            assert_eq!(v.int_val(), Some(0), "every output leaf must be 0");
        }
    }

    #[test]
    fn matmul_default_when_no_fact() {
        // No fact planted. matmul falls through to the generic body —
        // 2x3 @ 3x4 emits 2*4*3 = 24 MulI and 2*4*3 = 24 AddI ops (the
        // accumulator starts from a constant zero and adds each product).
        let mut b = IRBuilder::new();
        let lhs = make_int_matrix(&mut b, 2, 3);
        let rhs = make_int_matrix(&mut b, 3, 4);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let _ = matmul(&mut b, &lhs, &rhs);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(
            post_muls - pre_muls,
            24,
            "default matmul must emit M*N*K MulI ops (got {})",
            post_muls - pre_muls
        );
        assert_eq!(
            post_adds - pre_adds,
            24,
            "default matmul must emit M*N*K AddI ops (got {})",
            post_adds - pre_adds
        );
    }

    #[test]
    fn matmul_zero_1d_1d_returns_scalar_zero() {
        // 1D@1D dot product short-circuits to a scalar 0 when either
        // operand is provably all-zeros. No MulI / AddI emitted.
        let mut b = IRBuilder::new();
        let lhs = make_int_list(&mut b, &[1, 2, 3, 4]);
        let rhs = make_int_list(&mut b, &[5, 6, 7, 8]);
        let lhs_vid = lhs.value_id().expect("list has value_id");
        plant_forall_eq_const(&mut b, lhs_vid, 0);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let result = matmul(&mut b, &lhs, &rhs);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(
            result.int_val(),
            Some(0),
            "1D@1D matmul on forall_eq_const(lhs, 0) must be scalar 0"
        );
        assert_eq!(
            post_muls, pre_muls,
            "1D@1D matmul-on-zeros must not emit any MulI ops"
        );
        assert_eq!(
            post_adds, pre_adds,
            "1D@1D matmul-on-zeros must not emit any AddI ops"
        );
    }

    // ── Group 8b: matmul is_identity short-circuit strategy dispatch ────

    fn plant_is_identity(b: &mut IRBuilder, vid: ValueId) {
        let fact = ContractTerm::PredicateApp {
            kind: "is_identity".to_string(),
            args: vec![ContractTerm::Var(ContractVar::Value(vid))],
        };
        b.facts.insert_for(vid, fact);
    }

    #[test]
    fn matmul_lhs_identity_returns_rhs() {
        // Plant `is_identity(lhs_vid)` on a 3x3 matrix. matmul with a
        // 3x4 rhs must short-circuit by returning rhs unchanged — no
        // MulI / AddI emitted, and the result must be the rhs Value.
        let mut b = IRBuilder::new();
        let lhs = make_int_matrix(&mut b, 3, 3);
        let rhs = make_int_matrix(&mut b, 3, 4);
        let lhs_vid = lhs.value_id().expect("matrix has value_id");
        let rhs_vid = rhs.value_id().expect("matrix has value_id");
        plant_is_identity(&mut b, lhs_vid);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let result = matmul(&mut b, &lhs, &rhs);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(
            post_muls, pre_muls,
            "matmul on is_identity(lhs) must not emit any MulI ops"
        );
        assert_eq!(
            post_adds, pre_adds,
            "matmul on is_identity(lhs) must not emit any AddI ops"
        );
        assert_eq!(
            result.value_id(),
            Some(rhs_vid),
            "lhs_is_identity strategy must return rhs unchanged (same value_id)"
        );
    }

    #[test]
    fn matmul_rhs_identity_returns_lhs() {
        // Plant `is_identity(rhs_vid)` on a 4x4 matrix. matmul with a
        // 3x4 lhs must short-circuit by returning lhs unchanged.
        let mut b = IRBuilder::new();
        let lhs = make_int_matrix(&mut b, 3, 4);
        let rhs = make_int_matrix(&mut b, 4, 4);
        let lhs_vid = lhs.value_id().expect("matrix has value_id");
        let rhs_vid = rhs.value_id().expect("matrix has value_id");
        plant_is_identity(&mut b, rhs_vid);

        let pre_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let pre_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));
        let result = matmul(&mut b, &lhs, &rhs);
        let post_muls = count_ir_kind(&b, |ir| matches!(ir, IR::MulI));
        let post_adds = count_ir_kind(&b, |ir| matches!(ir, IR::AddI));

        assert_eq!(
            post_muls, pre_muls,
            "matmul on is_identity(rhs) must not emit any MulI ops"
        );
        assert_eq!(
            post_adds, pre_adds,
            "matmul on is_identity(rhs) must not emit any AddI ops"
        );
        assert_eq!(
            result.value_id(),
            Some(lhs_vid),
            "rhs_is_identity strategy must return lhs unchanged (same value_id)"
        );
    }
}
