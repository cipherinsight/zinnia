//! Witness-free layout regression tests. No prover, witness, or key generation.
use super::*;
use halo2_base::gates::circuit::CircuitBuilderStage;

fn layout(op: impl FnOnce(&mut Halo2BaseSynthesizer<'_>)) -> (usize, usize) {
    // This stage selects shape-only cell accounting; no keys are generated.
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Keygen);
    builder.set_k(12);
    builder.set_lookup_bits(8);
    builder.set_instance_columns(1);
    let params = ProvingParams { k: 12, precision_bits: 32, lookup_bits: 8 };
    op(&mut Halo2BaseSynthesizer::new(&mut builder, None, params));
    let stats = builder.statistics();
    (stats.gate.total_advice_per_phase[0], stats.total_lookup_advice_per_phase[0])
}

#[test]
fn neutral_arithmetic_preserves_cells_and_nonconstant_zero() {
    layout(|s| {
        let x = s.cell(Fp::zero()); // Unknown input placeholder, not a constant.
        let zero = s.constant_int(0).unwrap();
        let one = s.constant_int(1).unwrap();
        let before = s.builder.main(0).advice.len();
        assert_eq!(s.add_i(&x, &zero).unwrap().id, x.id);
        assert_eq!(s.add_i(&zero, &x).unwrap().id, x.id);
        assert_eq!(s.sub_i(&x, &zero).unwrap().id, x.id);
        assert_eq!(s.sub_i(&x, &x).unwrap().id, zero.id);
        assert_eq!(s.mul_i(&x, &one).unwrap().id, x.id);
        assert_eq!(s.mul_i(&one, &x).unwrap().id, x.id);
        assert_eq!(s.mul_i(&x, &zero).unwrap().id, zero.id);
        assert_eq!(s.mul_i(&zero, &x).unwrap().id, zero.id);
        let float_zero = s.constant_float(0.0).unwrap();
        let float_one = s.constant_float(1.0).unwrap();
        assert_eq!(s.mul_f(&x, &float_zero).unwrap().id, zero.id);
        assert_eq!(s.mul_f(&x, &float_one).unwrap().id, x.id);
        assert_eq!(s.builder.main(0).advice.len(), before);
        let y = s.cell(Fp::zero());
        let before = s.builder.main(0).advice.len();
        let product = s.mul_i(&x, &y).unwrap();
        assert!(!s.is_const_cell(&product));
        assert_eq!(s.builder.main(0).advice.len(), before + 4);
    });
}

#[test]
fn neutral_add_materializes_a_deferred_product() {
    layout(|s| {
        let x = s.cell(Fp::zero());
        let y = s.cell(Fp::zero());
        let deferred = s.defer_product(&x, &y);
        let zero = s.constant_int(0).unwrap();
        let before = s.builder.main(0).advice.len();
        let output = s.add_i(&zero, &deferred).unwrap();
        assert!(output.assigned.unwrap().cell.is_some());
        assert_eq!(s.builder.main(0).advice.len(), before + 4);
    });
}

#[test]
fn boolean_idempotence_retains_truth_conversion() {
    layout(|s| {
        let bit = s.cell(Fp::zero());
        s.mark_bit(&bit);
        let before = s.builder.main(0).advice.len();
        assert_eq!(s.logical_and(&bit, &bit).unwrap().id, bit.id);
        assert_eq!(s.logical_or(&bit, &bit).unwrap().id, bit.id);
        assert_eq!(s.builder.main(0).advice.len(), before);
        let x = s.cell(Fp::zero());
        let before = s.builder.main(0).advice.len();
        let truth = s.logical_and(&x, &x).unwrap();
        assert!(s.is_bit(&truth));
        assert_ne!(truth.id, x.id);
        assert!(s.builder.main(0).advice.len() > before);
    });
}

#[test]
fn quarter_multiply_matches_direct_two_bit_rescale_layout() {
    let multiply = layout(|s| {
        let x = s.cell(Fp::zero());
        let quarter = s.constant_float(0.25).unwrap();
        s.mul_f(&quarter, &x).unwrap();
    });
    let direct = layout(|s| {
        let x = s.cell(Fp::zero());
        s.rescale_by(&x, 2).unwrap();
    });
    assert_eq!(multiply, direct);
}

#[test]
fn tanh_shares_exp_pair_layout() {
    let shared = layout(|s| {
        let x = s.cell(Fp::zero());
        s.tanh_f(&x).unwrap();
    });
    let duplicated = layout(|s| {
        let x = s.cell(Fp::zero());
        let sinh = s.sinh_f(&x).unwrap();
        let cosh = s.cosh_f(&x).unwrap();
        s.fixed_div(&sinh, &cosh).unwrap();
    });
    assert!(shared.0 < duplicated.0, "advice: {shared:?} vs {duplicated:?}");
    assert!(shared.1 < duplicated.1, "lookup: {shared:?} vs {duplicated:?}");
}

fn graph_layout(ops: Vec<(crate::ir_defs::IR, Vec<u32>)>, affine: bool) -> (usize, usize) {
    use crate::ir::{IRGraph, IRStatement};
    use crate::types::ValueId;
    let mut statements: Vec<IRStatement> = Vec::new();
    for (ir, arguments) in ops {
        let values = arguments.iter().map(|id| statements[*id as usize].value_id).collect();
        statements.push(IRStatement::new(statements.len() as u32, ValueId::next(), ir, arguments, values, None));
    }
    let graph = IRGraph::new(statements);
    layout(|s| {
        s.plan_ahead(&graph);
        if !affine { s.plan.affine_roots.clear(); s.plan.affine_links.clear(); s.plan.clips.clear(); s.plan.clip_links.clear(); }
        crate::prove::interpreter::interpret_ir(&graph, s).unwrap();
    })
}

#[test]
fn affine_sum_tree_removes_only_accumulator_copies() {
    use crate::ir_defs::IR;
    let mut ops: Vec<_> = (0..16).map(|i| (IR::ReadFloat {
        path: InputPath::new(format!("x{i}"), vec![]), is_public: false }, vec![])).collect();
    for i in 0..4 { ops.push((IR::SumF { len: 4 }, (i * 4..i * 4 + 4).collect())); }
    ops.push((IR::SumF { len: 4 }, vec![16, 17, 18, 19]));
    let original = graph_layout(ops.clone(), false);
    let packed = graph_layout(ops, true);
    assert_eq!(original.0 - packed.0, 4);
    assert_eq!(original.1, packed.1);
}

#[test]
fn affine_accumulation_keeps_rounded_products() {
    use crate::ir_defs::IR;
    let mut ops: Vec<_> = (0..8).map(|i| (IR::ReadFloat {
        path: InputPath::new(format!("x{i}"), vec![]), is_public: false }, vec![])).collect();
    ops.push((IR::MulF, vec![0, 1]));
    let mut acc = 8;
    for i in 1..4 {
        let product = ops.len() as u32;
        ops.push((IR::MulF, vec![i * 2, i * 2 + 1]));
        let next = ops.len() as u32;
        ops.push((IR::AddF, vec![acc, product]));
        acc = next;
    }
    let original = graph_layout(ops.clone(), false);
    let packed = graph_layout(ops, true);
    assert_eq!(original.0 - packed.0, 2);
    assert_eq!(original.1, packed.1, "all four product remainder checks remain");
}

#[test]
fn dyadic_chain_shares_signed_floor_decomposition() {
    layout(|s| {
        let a = s.cell(Fp::zero()); let b = s.cell(Fp::zero());
        s.ensure_signed64(&a).unwrap(); s.ensure_signed64(&b).unwrap();
        let sum = s.add_i(&a, &b).unwrap();
        let quarter = s.constant_float(0.25).unwrap();
        let minus_quarter = s.constant_float(-0.25).unwrap();
        let half = s.constant_float(0.5).unwrap();
        s.plan.dyadic_links.insert(1, 1); s.plan.dyadic_links.insert(3, 1);
        s.on_stmt_begin(1);
        let quarter_sum = s.mul_f(&quarter, &sum).unwrap();
        assert!(quarter_sum.assigned.unwrap().cell.is_none());
        s.on_stmt_begin(2);
        let positive = s.mul_f(&half, &quarter_sum).unwrap();
        s.on_stmt_begin(3);
        let minus_quarter_sum = s.mul_f(&minus_quarter, &sum).unwrap();
        s.on_stmt_begin(4);
        let negative = s.mul_f(&half, &minus_quarter_sum).unwrap();
        assert!(positive.assigned.unwrap().cell.is_some());
        assert!(negative.assigned.unwrap().cell.is_some());
        assert_eq!(s.dyadic_splits.len(), 1);
        assert!(s.dyadic_splits.contains_key(&(sum.id, 3)));
    });
}

#[test]
fn dyadic_deferral_requires_a_proven_intermediate_window() {
    layout(|s| {
        let x = s.cell(Fp::zero()); // No input bound has been constrained.
        let quarter = s.constant_float(0.25).unwrap();
        s.plan.dyadic_links.insert(1, 1); s.on_stmt_begin(1);
        let out = s.mul_f(&quarter, &x).unwrap();
        assert!(out.assigned.unwrap().cell.is_some());
    });
}

#[test]
fn constant_clip_uses_fewer_cells_than_two_comparisons() {
    use crate::ir_defs::IR;
    let ops = vec![
        (IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false }, vec![]),
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::ConstantInt { value: 10 }, vec![]),
        (IR::GtI, vec![0, 1]), (IR::SelectI, vec![3, 0, 1]),
        (IR::LtI, vec![4, 2]), (IR::SelectI, vec![5, 4, 2]),
    ];
    // The graph helper's control switch also disables the specialized clip.
    let packed = graph_layout(ops.clone(), true);
    let generic = graph_layout(ops, false);
    assert!(packed.0 + packed.1 < generic.0 + generic.1);
}

#[test]
fn relational_fraction_retains_remainders_and_narrows_quotient() {
    let measure = |retain_relation: bool| layout(|s| {
        let a = s.cell(Fp::zero()); let m = s.cell(Fp::zero());
        s.range_check(&a, 63);
        s.assert_relation(&a, &m, false).unwrap();
        let one = s.constant_float(1.0).unwrap();
        let cond = s.lt_f(&one, &m).unwrap();
        let maximum = s.select_bit(&cond, &m, &one).unwrap();
        let epsilon = s.constant_float(1e-9).unwrap();
        let denominator = s.add_i(&maximum, &epsilon).unwrap();
        assert!(s.strictly_below(&a, &denominator));
        s.record_nonzero(&denominator);
        if !retain_relation { s.lower_bounds.clear(); }
        s.div_f(&a, &denominator).unwrap();
    });
    let generic = measure(false); let fraction = measure(true);
    assert!(fraction.0 < generic.0);
    assert_eq!(generic.1 - fraction.1, 4, "only the quotient loses four lookup limbs");
}

#[test]
fn comparison_gap_is_reused_by_exact_subtraction() {
    layout(|s| {
        let a = s.cell(Fp::zero()); let b = s.cell(Fp::zero());
        s.ensure_signed64(&a).unwrap(); s.ensure_signed64(&b).unwrap();
        s.assert_relation(&a, &b, false).unwrap();
        let before = s.builder.main(0).advice.len();
        let difference = s.sub_i(&b, &a).unwrap();
        assert_eq!(s.differences[&(b.id, a.id)].id, difference.id);
        assert_eq!(before, s.builder.main(0).advice.len());
    });
}

#[test]
fn bitvector_shift_is_wiring_and_keeps_arithmetic_sign() {
    layout(|s| {
        let input = s.cell(Fp::zero()); s.ensure_signed64(&input).unwrap();
        let bits = s.low_bits(&input, 64).unwrap();
        let one = s.constant_int(1).unwrap();
        s.plan.vector_ops.insert(0); s.plan.vector_links.insert(0);
        let before = s.builder.main(0).advice.len();
        let shifted = s.shift_right(&input, &one).unwrap();
        assert_eq!(s.builder.main(0).advice.len(), before);
        let shifted_bits = s.low_bits(&shifted, 64).unwrap();
        assert_eq!(shifted_bits[0].id, bits[1].id);
        assert_eq!(shifted_bits[63].id, bits[63].id);
        assert_eq!(shifted_bits[62].id, bits[63].id);
        let scalar = s.materialize(&shifted);
        assert!(scalar.assigned.unwrap().cell.is_some());
    });
}

#[test]
fn bitvector_xor_select_retains_cached_lanes() {
    layout(|s| {
        let a = s.cell(Fp::zero()); let b = s.cell(Fp::zero());
        s.ensure_signed64(&a).unwrap(); s.ensure_signed64(&b).unwrap();
        let condition = s.hint_bit(false);
        s.plan.vector_ops.insert(0); s.plan.vector_links.insert(0);
        let xor = s.bit_xor(&a, &b).unwrap();
        assert!(s.vector_deferred.contains(&xor.id));
        let selected = s.select(&condition, &xor, &a).unwrap();
        assert!(s.vector_deferred.contains(&selected.id));
        let before = s.builder.main(0).advice.len();
        let bits = s.low_bits(&selected, 64).unwrap();
        assert_eq!(bits.len(), 64);
        assert_eq!(s.builder.main(0).advice.len(), before);
        let scalar = s.materialize(&selected);
        assert!(scalar.assigned.unwrap().cell.is_some());
    });
}

#[test]
fn conjunction_contradictions_remove_redundant_masks() {
    layout(|s| {
        let a = s.hint_bit(false); let b = s.hint_bit(false);
        let not_a = s.logical_not(&a).unwrap(); let not_b = s.logical_not(&b).unwrap();
        let h0 = s.logical_and(&not_a, &not_b).unwrap();
        let h1 = s.logical_and(&a, &not_b).unwrap();
        let not_h0 = s.logical_not(&h0).unwrap();
        let before = s.builder.main(0).advice.len();
        assert_eq!(s.logical_and(&h1, &not_h0).unwrap().id, h1.id);
        let impossible = s.logical_and(&h0, &not_h0).unwrap();
        assert_eq!(s.constant_truth(&impossible), Some(false));
        let x = s.cell(Fp::zero()); let y = s.cell(Fp::zero());
        assert_eq!(s.select_bit(&impossible, &x, &y).unwrap().id, y.id);
        assert_eq!(s.builder.main(0).advice.len(), before + 2);
    });
}

#[test]
fn indicator_sum_guard_is_removed_only_for_disjoint_terms() {
    layout(|s| {
        let a = s.hint_bit(false); let not_a = s.logical_not(&a).unwrap();
        let x = s.cell(Fp::zero()); let y = s.cell(Fp::zero());
        let xv = crate::types::ValueId::next(); let yv = crate::types::ValueId::next();
        let av = crate::types::ValueId::next(); let nv = crate::types::ValueId::next();
        for (vid, cell) in [(xv, x), (yv, y), (av, a), (nv, not_a)] { s.cell_of_vid.insert(vid, cell); }
        s.plan.affine_roots.insert(0, plan::AffineTree { terms: vec![plan::AffineTerm::Product(av, xv), plan::AffineTerm::Product(nv, yv)] });
        let sum = s.planned_affine(Fp::zero()).unwrap().unwrap();
        let before = s.builder.main(0).advice.len();
        assert_eq!(s.select_bit(&a, &x, &sum).unwrap().id, sum.id);
        assert_eq!(s.builder.main(0).advice.len(), before);
        let independent = s.hint_bit(false);
        s.indicator_sums.insert(sum.id, vec![(a, x), (independent, y)]);
        assert!(s.redundant_indicator_select(&a, &x, &sum).is_none());
    });
}
