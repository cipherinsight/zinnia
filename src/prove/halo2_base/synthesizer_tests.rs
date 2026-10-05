//! Soundness tests for the halo2-base synthesizer, run on the real lowering.
//!
//! Each test synthesizes one primitive on free input cells, exposes its
//! outputs as public instances and checks the circuit with halo2's
//! `MockProver`:
//!
//! * the honest synthesis is accepted and its outputs equal the shared
//!   kernel's value (the mock backend's semantics), and
//! * playing a malicious prover, every prover hint (`Halo2BaseSynthesizer::
//!   hint`) is perturbed in turn by each of a set of deltas, and the
//!   circuit must reject every perturbed witness: each hint has exactly one
//!   admissible value, so the outputs are a function of the inputs. The
//!   deltas include the classic division attacks: `±1` on a quotient (the
//!   remainder absorbs a multiple of the divisor) and `±1/d` in the field
//!   (the remainder moves by one, the quotient becomes a non-integer).

use halo2_base::gates::circuit::CircuitBuilderStage;
use halo2_base::halo2_proofs::dev::MockProver;

use super::*;
use crate::circuit_input::CircuitInputs;
use crate::provenance::ProvenanceTable;

const K: u32 = 12;
const PRECISION_BITS: u32 = 32;
const LOOKUP_BITS: u32 = 8;

type Op = dyn Fn(&mut Halo2BaseSynthesizer<'_>, &[Halo2BaseCell]) -> Result<Vec<Halo2BaseCell>, ProvingError>;

struct Outcome {
    outputs: Vec<Fp>,
    hints: usize,
    accepted: bool,
}

/// Synthesize `op` on free input cells holding `inputs`, adding each
/// `(index, delta)` of `tamper` to the hint with that index, and run the
/// MockProver.
fn synthesize(op: &Op, inputs: &[Fp], tamper: &[(usize, Fp)]) -> Outcome {
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Mock);
    builder.set_k(K as usize);
    builder.set_lookup_bits(LOOKUP_BITS as usize);
    builder.set_instance_columns(1);
    let provenance = ProvenanceTable::new();
    let params = ProvingParams { k: K, precision_bits: PRECISION_BITS, lookup_bits: LOOKUP_BITS };
    let witness = ResolvedWitness::new(CircuitInputs::new(), PRECISION_BITS);
    let (outputs, hints) = {
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, Some(witness), params, &provenance);
        synth.tamper = tamper::HintTamper::targeting(tamper);
        let cells: Vec<Halo2BaseCell> = inputs.iter().map(|v| synth.cell(*v)).collect();
        let outs = op(&mut synth, &cells).expect("synthesis failed");
        for out in &outs {
            synth.expose_public(out, "out").unwrap();
        }
        (synth.public_values().to_vec(), synth.tamper.hints_seen())
    };
    builder.calculate_params(Some(20));
    let accepted = MockProver::run(K, &builder, vec![outputs.clone()])
        .expect("MockProver::run")
        .verify()
        .is_ok();
    Outcome { outputs, hints, accepted }
}

fn inv(n: u128) -> Fp {
    kernel::i128_to_fp(n as i128).invert().unwrap()
}

/// Perturbations applied to each hint: small shifts, Q-unit shifts, and
/// field inverses of typical divisors (the "remainder moves by one,
/// quotient becomes a field fraction" attack).
fn deltas(extra_divisors: &[u128]) -> Vec<Fp> {
    let mut out = vec![Fp::one(), -Fp::one(), Fp::from(2u64)];
    let scale = 1u128 << PRECISION_BITS;
    out.push(kernel::i128_to_fp(scale as i128));
    out.push(-kernel::i128_to_fp(scale as i128));
    for d in [2u128, 3, scale].iter().chain(extra_divisors) {
        out.push(inv(*d));
        out.push(-inv(*d));
    }
    out
}

/// Honest synthesis is accepted with outputs `expected`; every perturbed
/// hint is rejected.
fn assert_sound(name: &str, op: &Op, inputs: &[Fp], expected: &[Fp], extra_divisors: &[u128]) {
    let honest = synthesize(op, inputs, &[]);
    assert!(honest.accepted, "{name}: honest witness rejected");
    assert_eq!(honest.outputs, expected, "{name}: honest outputs differ from the kernel");
    for index in 0..honest.hints {
        for delta in deltas(extra_divisors) {
            let tampered = synthesize(op, inputs, &[(index, delta)]);
            assert!(
                !tampered.accepted,
                "{name}: hint #{index} perturbed by {delta:?} was accepted (outputs {:?}, honest {:?})",
                tampered.outputs, honest.outputs,
            );
        }
    }
}

fn int(v: i64) -> Fp {
    kernel::i64_to_fp(v)
}

fn q(v: f64) -> Fp {
    kernel::quantize_to_fp(v, PRECISION_BITS)
}

fn binary(f: fn(&mut Halo2BaseSynthesizer<'_>, &Halo2BaseCell, &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError>) -> Box<Op> {
    Box::new(move |s, c| Ok(vec![f(s, &c[0], &c[1])?]))
}

fn unary(f: fn(&mut Halo2BaseSynthesizer<'_>, &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError>) -> Box<Op> {
    Box::new(move |s, c| Ok(vec![f(s, &c[0])?]))
}

/// `f(input, constant)`, with the right operand a fixed-column constant.
fn with_constant_rhs(
    f: fn(&mut Halo2BaseSynthesizer<'_>, &Halo2BaseCell, &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError>,
    rhs: Fp,
) -> Box<Op> {
    Box::new(move |s, c| {
        let constant = s.const_cell(rhs);
        Ok(vec![f(s, &c[0], &constant)?])
    })
}

// ── Integer and fixed-point division (B1) ──────────────────────────────

const INT_PAIRS: [(i64, i64); 8] = [(7, 2), (-7, 2), (7, -2), (-7, -2), (6, 3), (0, 5), (5, 0), (-9, 1)];

#[test]
fn floor_div_and_mod_follow_python_and_are_pinned() {
    let floor_div = binary(|s, a, b| s.floor_div_i(a, b));
    let modulo = binary(|s, a, b| s.mod_i(a, b));
    for (a, b) in INT_PAIRS {
        let (eq, er) = kernel::fp_floor_div(int(a), int(b));
        assert_sound(&format!("{a} // {b}"), &*floor_div, &[int(a), int(b)], &[eq], &[b.unsigned_abs().max(1) as u128]);
        assert_sound(&format!("{a} % {b}"), &*modulo, &[int(a), int(b)], &[er], &[b.unsigned_abs().max(1) as u128]);
    }
    // Python semantics, spelled out.
    assert_eq!(synthesize(&*floor_div, &[int(-7), int(2)], &[]).outputs, vec![int(-4)]);
    assert_eq!(synthesize(&*floor_div, &[int(7), int(-2)], &[]).outputs, vec![int(-4)]);
    assert_eq!(synthesize(&*modulo, &[int(-7), int(2)], &[]).outputs, vec![int(1)]);
    assert_eq!(synthesize(&*modulo, &[int(7), int(-2)], &[]).outputs, vec![int(-1)]);
}

#[test]
fn floor_div_and_mod_by_constant_are_pinned() {
    for (a, c) in [(23i64, 5i64), (-23, 5), (23, -5), (17, 8), (-1, 3), (40, 1)] {
        let (eq, er) = kernel::fp_floor_div(int(a), int(c));
        let floor_div = with_constant_rhs(|s, a, b| s.floor_div_i(a, b), int(c));
        let modulo = with_constant_rhs(|s, a, b| s.mod_i(a, b), int(c));
        assert_sound(&format!("{a} // const {c}"), &*floor_div, &[int(a)], &[eq], &[c.unsigned_abs() as u128]);
        assert_sound(&format!("{a} % const {c}"), &*modulo, &[int(a)], &[er], &[c.unsigned_abs() as u128]);
    }
}

#[test]
fn fixed_point_division_is_pinned() {
    let div = binary(|s, a, b| s.div_f(a, b));
    for (a, b) in [(1.0, 3.0), (-1.5, 2.5), (10.0, -4.0), (7.25, 0.0), (0.0, -2.0), (1e6, 1e-3)] {
        let expected = kernel::fp_div_prescale(q(a), q(b), PRECISION_BITS);
        let divisor = kernel::fp_to_i128(q(b)).unwrap().unsigned_abs();
        assert_sound(&format!("{a} / {b}"), &*div, &[q(a), q(b)], &[expected], &[divisor.max(1)]);
    }
    for c in [3.0, -0.75, 2.0] {
        let by_constant = with_constant_rhs(|s, a, b| s.div_f(a, b), q(c));
        let expected = kernel::fp_div_prescale(q(5.5), q(c), PRECISION_BITS);
        let divisor = kernel::fp_to_i128(q(c)).unwrap().unsigned_abs();
        assert_sound(&format!("5.5 / const {c}"), &*by_constant, &[q(5.5)], &[expected], &[divisor]);
    }
}

#[test]
fn float_floor_div_and_mod_are_pinned() {
    let floor_div = binary(|s, a, b| s.floor_div_f(a, b));
    let modulo = binary(|s, a, b| s.mod_f(a, b));
    for (a, b) in [(7.5, 2.0), (-7.5, 2.0), (7.5, -2.0), (1.0, 0.0)] {
        let divisor = kernel::fp_to_i128(q(b)).unwrap().unsigned_abs().max(1);
        assert_sound(
            &format!("{a} // {b}"),
            &*floor_div,
            &[q(a), q(b)],
            &[kernel::fp_float_floor_div(q(a), q(b), PRECISION_BITS)],
            &[divisor],
        );
        assert_sound(&format!("{a} % {b}"), &*modulo, &[q(a), q(b)], &[kernel::fp_float_mod(q(a), q(b))], &[divisor]);
    }
}

#[test]
fn int_cast_truncates_and_is_pinned() {
    let cast = unary(|s, a| s.int_cast(a));
    for (x, expected) in [(-2.5, -2), (-0.5, 0), (-3.9, -3), (2.5, 2), (7.0, 7), (0.0, 0)] {
        assert_sound(&format!("int({x})"), &*cast, &[q(x)], &[int(expected)], &[]);
    }
}

// ── Fixed-point multiplication (X-1: unbounded quotient) ───────────────

#[test]
fn fixed_point_multiplication_is_pinned() {
    let mul = binary(|s, a, b| s.mul_f(a, b));
    for (a, b) in [(1.5, -2.25), (0.1, 0.1), (-3.0, -7.5), (1234.5, 0.001)] {
        let (expected, _) = kernel::fp_mul_rescale(q(a), q(b), PRECISION_BITS);
        assert_sound(&format!("{a} * {b}"), &*mul, &[q(a), q(b)], &[expected], &[]);
    }
    let dot: Box<Op> = Box::new(|s, c| Ok(vec![s.product_sum_f(&[(c[0], c[1]), (c[2], c[3])])?]));
    let inputs = [q(0.5), q(-1.25), q(3.0), q(0.125)];
    let pairs: Vec<(Fp, Fp)> = vec![(inputs[0], inputs[1]), (inputs[2], inputs[3])];
    let (expected, _) = kernel::fp_product_sum_rescale(&pairs, PRECISION_BITS);
    assert_sound("product_sum", &*dot, &inputs, &[expected], &[]);
}

// ── Transcendentals (B2) ────────────────────────────────────────────────

#[test]
fn sqrt_is_exact_floor_and_pinned() {
    let sqrt = unary(|s, a| s.sqrt_f(a));
    for x in [2.0, 0.25, 0.0, 1e6, 3.0e-9, -1.0] {
        let expected = kernel::fp_sqrt(q(x), PRECISION_BITS);
        assert_sound(&format!("sqrt({x})"), &*sqrt, &[q(x)], &[expected], &[]);
    }
    // The negative root of 4 is not an admissible output.
    assert_eq!(synthesize(&*sqrt, &[q(4.0)], &[]).outputs, vec![q(2.0)]);
}

#[test]
fn sin_and_cos_are_pinned() {
    let sin = unary(|s, a| s.sin_f(a));
    let cos = unary(|s, a| s.cos_f(a));
    let two_pi = 2 * kernel::pi_q(PRECISION_BITS) as u128;
    for x in [0.0, 0.5, -1.0, 3.0, 4.0, -7.5, 100.0] {
        assert_sound(&format!("sin({x})"), &*sin, &[q(x)], &[kernel::fp_sin(q(x), PRECISION_BITS)], &[two_pi]);
    }
    for x in [0.0, 2.0, -4.0] {
        assert_sound(&format!("cos({x})"), &*cos, &[q(x)], &[kernel::fp_cos(q(x), PRECISION_BITS)], &[two_pi]);
    }
}

#[test]
fn arcsin_is_pinned_to_the_principal_branch() {
    let arcsin = unary(|s, a| s.arcsin_f(a));
    for x in [0.0, 0.5, -0.5, 1.0, -1.0, 0.99, 1.5] {
        let expected = kernel::fp_arcsin(q(x), PRECISION_BITS);
        assert_sound(&format!("arcsin({x})"), &*arcsin, &[q(x)], &[expected], &[(1u128 << 33) + 7]);
    }
}

// ── Other arithmetic primitives ────────────────────────────────────────

#[test]
fn abs_and_sign_are_pinned() {
    let abs = unary(|s, a| s.abs_i(a));
    let sign = unary(|s, a| s.sign_i(a));
    for (x, a, sg) in [(-5i64, 5i64, -1i64), (0, 0, 0), (5, 5, 1), (i64::MIN + 1, i64::MAX, -1)] {
        assert_sound(&format!("abs({x})"), &*abs, &[int(x)], &[int(a)], &[]);
        assert_sound(&format!("sign({x})"), &*sign, &[int(x)], &[int(sg)], &[]);
    }
}

#[test]
fn integer_power_is_pinned() {
    let pow = binary(|s, a, b| s.pow_i(a, b));
    for (base, e, expected) in [(3i64, 0i64, 1i64), (3, 1, 3), (-2, 5, -32), (7, 13, 96_889_010_407)] {
        assert_sound(&format!("{base} ** {e}"), &*pow, &[int(base), int(e)], &[int(expected)], &[]);
        let by_constant = with_constant_rhs(|s, a, b| s.pow_i(a, b), int(e));
        assert_sound(&format!("{base} ** const {e}"), &*by_constant, &[int(base)], &[int(expected)], &[]);
    }
    // An exponent outside [0, 64) has no witness.
    assert!(!synthesize(&*pow, &[int(2), int(64)], &[]).accepted);
    assert!(!synthesize(&*pow, &[int(2), int(-1)], &[]).accepted);
}

#[test]
fn field_division_is_total_and_pinned() {
    let div = binary(|s, a, b| s.div_i(a, b));
    let inverse = unary(|s, a| s.inv_i(a));
    assert_sound("6 /f 3", &*div, &[int(6), int(3)], &[int(2)], &[]);
    assert_sound("0 /f 0", &*div, &[int(0), int(0)], &[int(0)], &[]);
    assert_sound("5 /f 0", &*div, &[int(5), int(0)], &[int(0)], &[]);
    assert_sound("1/7", &*inverse, &[int(7)], &[int(7).invert().unwrap()], &[]);
    assert_sound("1/0", &*inverse, &[int(0)], &[int(0)], &[]);
}

#[test]
fn comparisons_and_casts_have_no_free_hints() {
    let ops: Vec<(&str, Box<Op>)> = vec![
        ("lt", binary(|s, a, b| s.lt_i(a, b))),
        ("lte", binary(|s, a, b| s.lte_i(a, b))),
        ("gt", binary(|s, a, b| s.gt_i(a, b))),
        ("gte", binary(|s, a, b| s.gte_i(a, b))),
        ("eq", binary(|s, a, b| s.eq_i(a, b))),
        ("ne", binary(|s, a, b| s.ne_i(a, b))),
        ("lt_bounded", binary(|s, a, b| s.lt_i_bounded(a, b))),
        ("and", binary(|s, a, b| s.logical_and(a, b))),
        ("or", binary(|s, a, b| s.logical_or(a, b))),
    ];
    for (name, op) in &ops {
        for (a, b) in [(3i64, 5i64), (5, 3), (4, 4), (0, 7)] {
            let outcome = synthesize(&**op, &[int(a), int(b)], &[]);
            assert!(outcome.accepted, "{name}({a}, {b}) rejected");
            assert_eq!(outcome.hints, 0, "{name} loads a prover hint");
        }
    }
    for (name, op) in [
        ("not", unary(|s, a| s.logical_not(a))),
        ("bool", unary(|s, a| s.bool_cast(a))),
        ("float", unary(|s, a| s.float_cast(a))),
    ] {
        let outcome = synthesize(&*op, &[int(-3)], &[]);
        assert!(outcome.accepted, "{name} rejected");
        assert_eq!(outcome.hints, 0, "{name} loads a prover hint");
    }
}

#[test]
fn signed_comparison_rejects_inputs_outside_the_64_bit_window() {
    let lt = binary(|s, a, b| s.lt_i(a, b));
    let huge = kernel::i128_to_fp(1i128 << 70);
    assert!(!synthesize(&*lt, &[huge, int(1)], &[]).accepted);
}

// ── Dynamic-array metadata (C15, previously a constraint-shape replica) ─

/// The trait-default `assert_dynamic_ndarray_meta`: every runtime axis is
/// in `[0, bound]` and the product is at most the total bound.
#[test]
fn dynamic_ndarray_meta_rejects_shapes_outside_the_envelope() {
    let op: Box<Op> = Box::new(|s, c| {
        s.allocate_dynamic_ndarray_meta(0, "Integer", 20, 2, &[4, 5])?;
        s.assert_dynamic_ndarray_meta(0, 2, 20, c)?;
        Ok(vec![])
    });
    assert!(synthesize(&*op, &[int(3), int(5)], &[]).accepted, "in-envelope shape rejected");
    assert!(synthesize(&*op, &[int(0), int(5)], &[]).accepted, "empty axis rejected");
    assert!(!synthesize(&*op, &[int(5), int(4)], &[]).accepted, "axis over its bound accepted");
    assert!(!synthesize(&*op, &[int(-1), int(5)], &[]).accepted, "negative axis accepted");
    let small_total: Box<Op> = Box::new(|s, c| {
        s.allocate_dynamic_ndarray_meta(0, "Integer", 10, 2, &[4, 5])?;
        s.assert_dynamic_ndarray_meta(0, 2, 10, c)?;
        Ok(vec![])
    });
    assert!(!synthesize(&*small_total, &[int(4), int(5)], &[]).accepted, "product over the total accepted");
}

// ── Witness independence of the new lowerings ──────────────────────────

/// Keygen without a witness (all inputs zero, the verifier's re-synthesis)
/// must produce the same circuit shape as a real witness.
#[test]
fn new_lowerings_are_witness_independent() {
    let ops: Vec<(&str, Box<Op>, Vec<Fp>)> = vec![
        ("floor_div", binary(|s, a, b| s.floor_div_i(a, b)), vec![int(-7), int(2)]),
        ("div_f", binary(|s, a, b| s.div_f(a, b)), vec![q(1.0), q(-3.0)]),
        ("sin", unary(|s, a| s.sin_f(a)), vec![q(-4.0)]),
        ("sqrt", unary(|s, a| s.sqrt_f(a)), vec![q(2.0)]),
        ("arcsin", unary(|s, a| s.arcsin_f(a)), vec![q(0.3)]),
        ("pow", binary(|s, a, b| s.pow_i(a, b)), vec![int(3), int(5)]),
        ("int_cast", unary(|s, a| s.int_cast(a)), vec![q(-2.5)]),
    ];
    for (name, op, inputs) in ops {
        let shape = |witness: Option<ResolvedWitness>| {
            let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Keygen);
            builder.set_k(K as usize);
            builder.set_lookup_bits(LOOKUP_BITS as usize);
            builder.set_instance_columns(1);
            let provenance = ProvenanceTable::new();
            let params = ProvingParams { k: K, precision_bits: PRECISION_BITS, lookup_bits: LOOKUP_BITS };
            let zeros = vec![Fp::zero(); inputs.len()];
            let values = if witness.is_some() { &inputs } else { &zeros };
            {
                let mut synth = Halo2BaseSynthesizer::new(&mut builder, witness, params, &provenance);
                let cells: Vec<Halo2BaseCell> = values.iter().map(|v| synth.cell(*v)).collect();
                op(&mut synth, &cells).expect("synthesis failed");
            }
            let stats = builder.statistics();
            (stats.gate.total_advice_per_phase, stats.total_lookup_advice_per_phase, stats.gate.total_fixed)
        };
        let with_witness = shape(Some(ResolvedWitness::new(CircuitInputs::new(), PRECISION_BITS)));
        let without = shape(None);
        assert_eq!(with_witness, without, "{name}: circuit shape depends on the witness");
    }
}

// ── The audit's attacks, against the pre-fix lowerings and the fixed ones ─
//
// `pre_fix` reproduces the lowerings of the snapshot e58e9a6 with their
// free cells routed through `hint`, so the tamper hook can play the attacks
// of the audit (core-backends B1/B2, data-baselines X-1). Each test shows
// that the attack is accepted by the pre-fix shape with a wrong output and
// rejected by the current lowering.

mod pre_fix {
    use super::*;

    /// `floor_div_i` at e58e9a6: `q`, `r` free, only `b*q + r == a`.
    pub(super) fn floor_div(s: &mut Halo2BaseSynthesizer<'_>, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let (sa, sb) = (kernel::fp_to_i64(a.value), kernel::fp_to_i64(b.value));
        let q = s.hint(int(sa.div_euclid(sb)));
        let r = s.hint(int(sa.rem_euclid(sb)));
        let recomposed = s.gate_cell(b.value * q.value + r.value, |gate, ctx| gate.mul_add(ctx, b.assigned, q.assigned, r.assigned));
        s.constrain_equal(&recomposed, a);
        Ok(q)
    }

    /// `mul_f` at e58e9a6: remainder range-checked, quotient not.
    pub(super) fn mul_f(s: &mut Halo2BaseSynthesizer<'_>, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let (result, rem) = kernel::fp_mul_rescale(a.value, b.value, PRECISION_BITS);
        let raw = s.mul_i(a, b)?;
        let result = s.hint(result);
        let rem = s.hint(rem);
        let scale = kernel::scale_fp(PRECISION_BITS);
        let recomposed = s.gate_cell(result.value * scale + rem.value, |gate, ctx| {
            gate.mul_add(ctx, result.assigned, QuantumCell::Constant(scale), rem.assigned)
        });
        s.constrain_equal(&recomposed, &raw);
        s.range_check(&rem, PRECISION_BITS as usize);
        Ok(result)
    }

    /// `|x - y| < eps` by two signed comparisons (e58e9a6 `assert_close_f`).
    fn assert_close(s: &mut Halo2BaseSynthesizer<'_>, x: &Halo2BaseCell, y: &Halo2BaseCell, eps: f64) -> Result<(), ProvingError> {
        let diff = s.sub_f(x, y)?;
        let eps_pos = s.constant_float(eps)?;
        let eps_neg = s.constant_float(-eps)?;
        let below = s.lt_f(&diff, &eps_pos)?;
        let above = s.lt_f(&eps_neg, &diff)?;
        s.assert_true(&below)?;
        s.assert_true(&above)
    }

    /// `sqrt_f` at e58e9a6: `|y*y - a| < 0.001`.
    pub(super) fn sqrt(s: &mut Halo2BaseSynthesizer<'_>, a: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let y = s.hint(q(kernel::dequantize_from_fp(a.value, PRECISION_BITS).max(0.0).sqrt()));
        let y_squared = mul_f(s, &y, &y)?;
        assert_close(s, &y_squared, a, 0.001)?;
        Ok(y)
    }

    /// `arcsin_f` at e58e9a6: `|sin(y) - a| < 0.001`, `y` unrestricted
    /// (here with the fixed `sin_f`, so only the missing branch
    /// restriction is exercised).
    pub(super) fn arcsin(s: &mut Halo2BaseSynthesizer<'_>, a: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let y = s.hint(q(kernel::dequantize_from_fp(a.value, PRECISION_BITS).asin()));
        let sin_y = s.sin_f(&y)?;
        assert_close(s, &sin_y, a, 0.001)?;
        Ok(y)
    }
}

/// An attack: perturb hints by `deltas` and observe the result.
fn attack(op: &Op, inputs: &[Fp], tamper: &[(usize, Fp)]) -> Outcome {
    let honest = synthesize(op, inputs, &[]);
    assert!(honest.accepted, "honest witness rejected");
    let outcome = synthesize(op, inputs, tamper);
    if outcome.accepted {
        assert_ne!(outcome.outputs, honest.outputs, "attack did not change the output");
    }
    outcome
}

#[test]
fn harness_detects_a_free_hint() {
    let free: Box<Op> = Box::new(|s, _| Ok(vec![s.hint(int(5))]));
    assert!(synthesize(&*free, &[], &[(0, Fp::one())]).accepted);
}

#[test]
fn quotient_shift_attack_on_floor_division() {
    // 7 // 2: q = 3, r = 1. Attack: q = 4, r = -1.
    let old = binary(pre_fix::floor_div);
    assert!(attack(&*old, &[int(7), int(2)], &[(0, Fp::one()), (1, -int(2))]).accepted);
    let new = binary(|s, a, b| s.floor_div_i(a, b));
    assert!(!attack(&*new, &[int(7), int(2)], &[(0, Fp::one())]).accepted);
}

#[test]
fn fraction_quotient_attack_on_fixed_point_multiplication() {
    // Move the remainder by one; the quotient becomes q - 2^-P in the field.
    let shift = -inv(1u128 << PRECISION_BITS);
    let inputs = [q(1.5), q(0.1)];
    let old = binary(pre_fix::mul_f);
    assert!(attack(&*old, &inputs, &[(0, shift), (1, Fp::one())]).accepted);
    let new = binary(|s, a, b| s.mul_f(a, b));
    assert!(!attack(&*new, &inputs, &[(0, shift)]).accepted);
}

#[test]
fn negative_root_attack_on_sqrt() {
    let flip = -q(4.0); // y = 2 becomes -2
    let old = unary(pre_fix::sqrt);
    assert!(attack(&*old, &[q(4.0)], &[(0, flip)]).accepted);
    let new = unary(|s, a| s.sqrt_f(a));
    // Hint 0 is the input's sign bit, hint 1 the root.
    assert!(!attack(&*new, &[q(4.0)], &[(1, flip)]).accepted);
}

#[test]
fn branch_attack_on_arcsin() {
    // arcsin(0.5) = π/6; π - π/6 has the same sine.
    let y = kernel::fp_arcsin(q(0.5), PRECISION_BITS);
    let other_branch = q(std::f64::consts::PI) - y - y;
    let old = unary(pre_fix::arcsin);
    assert!(attack(&*old, &[q(0.5)], &[(0, other_branch)]).accepted);
    // The fixed lowering has no free output: every hint is pinned
    // (`arcsin_is_pinned_to_the_principal_branch`), and the output is the
    // kernel's principal value.
    let new = unary(|s, a| s.arcsin_f(a));
    assert_eq!(synthesize(&*new, &[q(0.5)], &[]).outputs, vec![y]);
}
