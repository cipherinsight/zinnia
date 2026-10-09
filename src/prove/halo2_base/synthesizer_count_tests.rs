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
