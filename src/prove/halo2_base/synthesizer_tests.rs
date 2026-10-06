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

/// Honest synthesis is accepted with outputs `expected`; a perturbed hint
/// is rejected or leaves every output unchanged. For inputs where a hint is
/// free by design: `|x|` at `x = 0`, where either sign gives 0 (see
/// `magnitude`), so the outputs are still a function of the inputs.
fn assert_outputs_pinned(name: &str, op: &Op, inputs: &[Fp], expected: &[Fp], extra_divisors: &[u128]) {
    let honest = synthesize(op, inputs, &[]);
    assert!(honest.accepted, "{name}: honest witness rejected");
    assert_eq!(honest.outputs, expected, "{name}: honest outputs differ from the kernel");
    for index in 0..honest.hints {
        for delta in deltas(extra_divisors) {
            let tampered = synthesize(op, inputs, &[(index, delta)]);
            assert!(
                !tampered.accepted || tampered.outputs == honest.outputs,
                "{name}: hint #{index} perturbed by {delta:?} changed the outputs to {:?}",
                tampered.outputs,
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
fn pinned_width_floor_division_is_pinned() {
    for bits in [5u32, 12] {
        let op: Box<Op> = Box::new(move |s, c| Ok(vec![s.floor_div_i_pinned(&c[0], &c[1], bits)?]));
        for (a, b) in [(24i64, 4i64), (23, 5), (0, 3), (7, 0)] {
            let (expected, _) = kernel::fp_floor_div(int(a), int(b));
            assert_sound(&format!("{a} // {b} [{bits} bits]"), &*op, &[int(a), int(b)], &[expected], &[b.max(1) as u128]);
        }
        // A quotient that does not fit the certified width has no witness.
        assert!(!synthesize(&*op, &[int(1 << 20), int(1)], &[]).accepted);
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

/// The mock's power: 1.0 multiplied by the base |e| times, `1 / power`
/// for a negative exponent.
fn mock_power(base: Fp, e: i64) -> Fp {
    let mut power = q(1.0);
    for _ in 0..e.unsigned_abs() {
        power = kernel::fp_mul_rescale(power, base, PRECISION_BITS).0;
    }
    if e < 0 { kernel::fp_div_prescale(q(1.0), power, PRECISION_BITS) } else { power }
}

/// `x * float(n)` is the exact `x·n`; a constant `m·2^s` factor rescales by
/// `P - s` bits only; both equal the kernel's `fp_mul_rescale` and every
/// remaining hint is pinned.
#[test]
fn products_with_integer_valued_factors_are_exact_and_pinned() {
    let by_cast: Box<Op> = Box::new(|s, c| {
        let n = s.float_cast(&c[1])?;
        Ok(vec![s.mul_f(&c[0], &n)?, s.mul_f(&n, &c[0])?])
    });
    for (x, n) in [(1.5, 3i64), (-0.1, 7), (123.456, -2), (0.0, 5)] {
        let expected = kernel::fp_mul_rescale(q(x), q(n as f64), PRECISION_BITS).0;
        let outcome = synthesize(&*by_cast, &[q(x), int(n)], &[]);
        assert!(outcome.accepted && outcome.outputs == vec![expected, expected], "{x} * float({n})");
        assert_eq!(outcome.hints, 0, "an exact product needs no hint");
    }
    for c in [0.5, 3.0, -0.75, 2f64.powi(-32), 0.1, 0.0] {
        let by_constant = with_constant_rhs(|s, a, b| s.mul_f(a, b), q(c));
        for x in [1.5, -2.25, 1e6, -3e-9] {
            let expected = kernel::fp_mul_rescale(q(x), q(c), PRECISION_BITS).0;
            assert_sound(&format!("{x} * const {c}"), &*by_constant, &[q(x)], &[expected], &[]);
        }
    }
    let generic = binary(|s, a, b| s.mul_f(a, b));
    let half = with_constant_rhs(|s, a, b| s.mul_f(a, b), q(0.5));
    assert!(cells_of(&*half, &[q(3.0)]).1 < cells_of(&*generic, &[q(3.0), q(0.5)]).1);
    let dot: Box<Op> = Box::new(|s, c| {
        let n = s.float_cast(&c[2])?;
        Ok(vec![s.product_sum_f(&[(c[0], n), (c[1], c[3])])?])
    });
    let inputs = [q(0.75), q(-1.25), int(-3), q(0.3)];
    let pairs = vec![(inputs[0], q(-3.0)), (inputs[1], inputs[3])];
    let expected = kernel::fp_product_sum_rescale(&pairs, PRECISION_BITS).0;
    assert_sound("0.75*float(-3) + -1.25*0.3", &*dot, &inputs, &[expected], &[]);
}

/// `a / m·2^s` divides `a·2^(P-s)` by `m`; `a / float(n)` divides `a` by `n`;
/// both equal the kernel's `fp_div_prescale` (0 for a zero divisor).
#[test]
fn divisions_by_integer_valued_divisors_are_pinned() {
    for c in [30.0, -0.75, 0.5, 1.0, -1.0, 3.0e-3] {
        let by_constant = with_constant_rhs(|s, a, b| s.div_f(a, b), q(c));
        for a in [5.5, -5.5, 0.0, 1e6] {
            let expected = kernel::fp_div_prescale(q(a), q(c), PRECISION_BITS);
            let divisor = kernel::fp_to_i128(q(c)).unwrap().unsigned_abs();
            assert_sound(&format!("{a} / const {c}"), &*by_constant, &[q(a)], &[expected], &[divisor, 30, 3]);
        }
    }
    let by_cast: Box<Op> = Box::new(|s, c| {
        let n = s.float_cast(&c[1])?;
        Ok(vec![s.div_f(&c[0], &n)?])
    });
    for (a, n) in [(7.5, 2i64), (-7.5, 2), (7.5, -2), (1.0, 3), (4.0, 0)] {
        let expected = kernel::fp_div_prescale(q(a), q(n as f64), PRECISION_BITS);
        assert_sound(&format!("{a} / float({n})"), &*by_cast, &[q(a), int(n)], &[expected], &[n.unsigned_abs().max(1) as u128]);
    }
}

#[test]
fn powers_follow_the_mock_and_are_pinned() {
    for e in [0i64, 1, 2, 6, -2] {
        let by_constant = with_constant_rhs(|s, a, b| s.pow_f_int(a, b), int(e));
        for base in [1.5, -0.7, 2.0] {
            assert_sound(&format!("{base} ** {e}"), &*by_constant, &[q(base)], &[mock_power(q(base), e)], &[]);
        }
    }
    let dynamic = binary(|s, a, b| s.pow_f_int(a, b));
    for e in [0i64, 1, 5, 8] {
        assert_sound(&format!("1.25 ** {e} (runtime)"), &*dynamic, &[q(1.25), int(e)], &[mock_power(q(1.25), e)], &[]);
    }
    assert!(!synthesize(&*dynamic, &[q(1.25), int(9)], &[]).accepted, "an exponent above 8 has no witness");
    assert!(!synthesize(&*dynamic, &[q(1.25), int(-1)], &[]).accepted);
    let float_exponent = with_constant_rhs(|s, a, b| s.pow_f(a, b), q(3.0));
    assert_sound("1.5 ** 3.0", &*float_exponent, &[q(1.5)], &[mock_power(q(1.5), 3)], &[]);
    let fractional = with_constant_rhs(|s, a, b| s.pow_f(a, b), q(0.5));
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Keygen);
    builder.set_k(K as usize);
    builder.set_lookup_bits(LOOKUP_BITS as usize);
    let provenance = ProvenanceTable::new();
    let params = ProvingParams { k: K, precision_bits: PRECISION_BITS, lookup_bits: LOOKUP_BITS };
    let mut synth = Halo2BaseSynthesizer::new(&mut builder, None, params, &provenance);
    let x = synth.cell(q(2.0));
    assert!(fractional(&mut synth, &[x]).is_err(), "a fractional exponent must be rejected, not proved");
}

/// Quotients of dividends with a proven range are checked against the range
/// that implies, not the 64-bit window: still pinned, and cheaper.
#[test]
fn quotients_of_ranged_dividends_are_pinned_by_narrow_checks() {
    let ranged: Box<Op> = Box::new(|s, c| {
        let bound = s.constant_int(1 << 40)?;
        s.assert_u128_lt(&c[0], &bound)?;
        s.assert_u128_lt(&c[1], &bound)?;
        let seven = s.constant_int(7)?;
        let product = s.mul_f(&c[0], &c[1])?;
        Ok(vec![product, s.floor_div_i(&c[0], &seven)?, s.mod_i(&c[1], &seven)?])
    });
    for (a, b) in [(3i64 << 32, 5i64 << 31), (12345, 678910), (0, (1 << 40) - 1)] {
        let expected = [
            kernel::fp_mul_rescale(int(a), int(b), PRECISION_BITS).0,
            kernel::fp_floor_div(int(a), int(7)).0,
            kernel::fp_floor_div(int(b), int(7)).1,
        ];
        assert_sound(&format!("ranged {a}, {b}"), &*ranged, &[int(a), int(b)], &expected, &[7]);
    }
    let unranged: Box<Op> = Box::new(|s, c| Ok(vec![s.mul_f(&c[0], &c[1])?]));
    let ranged_product: Box<Op> = Box::new(|s, c| {
        let bound = s.constant_int(1 << 40)?;
        s.assert_u128_lt(&c[0], &bound)?;
        s.assert_u128_lt(&c[1], &bound)?;
        Ok(vec![s.mul_f(&c[0], &c[1])?])
    });
    let unranged_cost = cells_of(&*unranged, &[int(3), int(5)]);
    let ranged_cost = cells_of(&*ranged_product, &[int(3), int(5)]);
    let bounds_cost = cells_of(&*Box::new(|s: &mut Halo2BaseSynthesizer<'_>, c: &[Halo2BaseCell]| {
        let bound = s.constant_int(1 << 40)?;
        s.assert_u128_lt(&c[0], &bound)?;
        s.assert_u128_lt(&c[1], &bound)?;
        Ok(vec![])
    }) as &Op, &[int(3), int(5)]);
    assert!(ranged_cost.1 - bounds_cost.1 < unranged_cost.1, "the 48-bit quotient check is not narrower");
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

/// `IR::SqrtProductIF`: the floor root of `b·c·2^64`, every hint pinned
/// (the root, also against its negation `p - y` and `y + (p+1)/2`, the
/// field elements whose doubles are small); outside `0 <= b < 2^64`,
/// `0 <= b·c < 2^62` there is no witness.
#[test]
fn sqrt_of_an_integer_product_is_the_floor_root_and_pinned() {
    let root = binary(|s, b, c| s.sqrt_product_i_f(b, c));
    let cases: [(i64, i64); 7] =
        [(1, 1), (3, 5), (2_147_395_600, 2_147_395_600), ((1 << 62) - 1, 1), (0, 7), (1 << 40, 3), (7, 0)];
    for (b, c) in cases {
        let expected = kernel::fp_sqrt_product_int(int(b), int(c), PRECISION_BITS).expect("in the domain");
        assert_sound(&format!("sqrt({b} * {c})"), &*root, &[int(b), int(c)], &[expected], &[]);
        let negated = synthesize(&*root, &[int(b), int(c)], &[(0, -expected - expected)]);
        assert!(!negated.accepted || expected == Fp::zero(), "sqrt({b} * {c}): the negative root was accepted");
    }
    let outside = [
        (int(1 << 31), int(1 << 31)),                     // b·c = 2^62
        (int(-1), int(-1)),                               // b < 0
        (int(1), int(-1)),                                // b·c < 0
        (kernel::i128_to_fp(1 << 64), int(0)),            // b = 2^64
        (int(i64::MAX), int(i64::MAX)),                   // b·c far above
    ];
    for (b, c) in outside {
        assert_eq!(kernel::fp_sqrt_product_int(b, c, PRECISION_BITS), None);
        assert!(!synthesize(&*root, &[b, c], &[]).accepted, "an operand outside the domain was accepted");
    }
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
        // At 0 the sign hint of `|x|` is free (either sign gives 0).
        let check = if x == 0.0 { assert_outputs_pinned } else { assert_sound };
        check(&format!("arcsin({x})"), &*arcsin, &[q(x)], &[expected], &[(1u128 << 33) + 7]);
    }
}

// ── Other arithmetic primitives ────────────────────────────────────────

#[test]
fn abs_and_sign_are_pinned() {
    let abs = unary(|s, a| s.abs_i(a));
    let sign = unary(|s, a| s.sign_i(a));
    for (x, a, sg) in [(-5i64, 5i64, -1i64), (0, 0, 0), (5, 5, 1), (i64::MIN + 1, i64::MAX, -1)] {
        // At 0 the sign hint of `|x|` is free (either sign gives 0).
        let check_abs = if x == 0 { assert_outputs_pinned } else { assert_sound };
        check_abs(&format!("abs({x})"), &*abs, &[int(x)], &[int(a)], &[]);
        assert_sound(&format!("sign({x})"), &*sign, &[int(x)], &[int(sg)], &[]);
    }
    assert!(!synthesize(&*abs, &[int(i64::MIN)], &[]).accepted, "|i64::MIN| overflows int64");
}

/// `|x|` of a value already proven inside the window: a hinted sign `s`,
/// `x·(1 - 2s)` and one range check. The sign is pinned for `x != 0`; at
/// `x = 0` either sign gives 0, so it is not reused as `x`'s sign (a later
/// `x < 0` makes its own). `|i64::MIN|` overflows int64: no proof.
#[test]
fn magnitudes_of_window_values_are_pinned() {
    let abs_then_sign: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        let magnitude = s.abs_i(&c[0])?;
        let zero = s.constant_int(0)?;
        Ok(vec![magnitude, s.lt_i(&c[0], &zero)?])
    });
    for (x, a) in [(-5i64, 5i64), (5, 5), (i64::MIN + 1, i64::MAX), (i64::MAX, i64::MAX)] {
        assert_sound(&format!("|{x}|, {x} < 0"), &*abs_then_sign, &[int(x)], &[int(a), bit(x < 0)], &[]);
    }
    let honest = synthesize(&*abs_then_sign, &[int(0)], &[]);
    assert!(honest.accepted);
    assert_eq!(honest.outputs, vec![int(0), bit(false)]);
    for index in 0..honest.hints {
        for delta in deltas(&[]) {
            let tampered = synthesize(&*abs_then_sign, &[int(0)], &[(index, delta)]);
            assert!(!tampered.accepted || tampered.outputs == honest.outputs, "x = 0: hint #{index} + {delta:?} changed the outputs");
        }
    }
    assert!(!synthesize(&*abs_then_sign, &[int(i64::MIN)], &[]).accepted, "|i64::MIN| overflows int64");
    let window_abs: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        Ok(vec![s.abs_i(&c[0])?])
    });
    let window_only: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        Ok(vec![])
    });
    let cost = cells_of(&*window_abs, &[int(-5)]).0 - cells_of(&*window_only, &[int(-5)]).0;
    assert!(cost <= 40, "|x| of a window value costs {cost} advice cells");
}

/// `|a - b|` of two window values (`np.isclose`'s gap): the difference has
/// no window proof, and the magnitude's own 63-bit check bounds it. Every
/// perturbed hint is rejected; a difference outside `(-2^63, 2^63)` (which
/// overflows int64) has no proof; one range check, not `sign_and_abs`'s two.
#[test]
fn magnitudes_of_unproven_differences_are_pinned() {
    let gap: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        s.ensure_signed64(&c[1])?;
        let difference = s.sub_i(&c[0], &c[1])?;
        Ok(vec![s.abs_i(&difference)?])
    });
    let max = i64::MAX;
    for (a, b) in [(3i64, 10i64), (10, 3), (-7, -8), (max, 0), (0, max), (-1, max - 1), (max - 1, -1)] {
        let expected = (a as i128 - b as i128).unsigned_abs() as i64;
        assert_sound(&format!("|{a} - {b}|"), &*gap, &[int(a), int(b)], &[int(expected)], &[]);
    }
    // At a zero difference the sign hint is free, the magnitude is not.
    assert_outputs_pinned("|-7 - -7|", &*gap, &[int(-7), int(-7)], &[int(0)], &[]);
    for (a, b) in [(max, -1i64), (-1, max), (i64::MIN, 0), (i64::MIN, max), (max, i64::MIN)] {
        assert!(!synthesize(&*gap, &[int(a), int(b)], &[]).accepted, "|{a} - {b}| overflows int64");
    }
    let differences: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        s.ensure_signed64(&c[1])?;
        Ok(vec![s.sub_i(&c[0], &c[1])?])
    });
    // One 63-bit range check plus two gates, not `sign_and_abs`'s two checks.
    let check63: Box<Op> = Box::new(|s, c| {
        s.range_check(&c[0], 63);
        Ok(vec![])
    });
    let gap_cost = cells_of(&*gap, &[int(-5), int(9)]).0 - cells_of(&*differences, &[int(-5), int(9)]).0;
    let check_cost = cells_of(&*check63, &[int(5)]).0;
    assert!(gap_cost < 2 * check_cost, "|a - b| costs {gap_cost} advice cells, a 63-bit check {check_cost}");
}

/// Sign tests and `|x|` of a window-checked value read the top limb of the
/// window check (`sign_from_window`): the hinted bit `[x >= 0]` and, for
/// `|x|` at the bottom of the window, the inverse proving `x != i64::MIN`
/// are pinned (every perturbation rejected, `x = 0` included: the limb
/// decides its sign); `|i64::MIN|` has no proof; the sign test is cheaper
/// than a 64-bit comparison.
#[test]
fn window_signs_read_the_top_limb() {
    let sign_and_abs: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        let zero = s.constant_int(0)?;
        let negative = s.lt_i(&c[0], &zero)?;
        let magnitude = s.abs_i(&c[0])?;
        Ok(vec![negative, magnitude])
    });
    for x in [-5i64, 5, 0, -1, 1, i64::MAX, i64::MIN + 1, -(1 << 40), 1 << 40] {
        let expected = [bit(x < 0), int((x as i128).abs() as i64)];
        assert_sound(&format!("x < 0, |x| at {x}"), &*sign_and_abs, &[int(x)], &expected, &[]);
    }
    assert!(!synthesize(&*sign_and_abs, &[int(i64::MIN)], &[]).accepted, "|i64::MIN| overflows int64");
    let sign_only: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        let zero = s.constant_int(0)?;
        Ok(vec![s.lt_i(&c[0], &zero)?])
    });
    assert_sound("i64::MIN < 0", &*sign_only, &[int(i64::MIN)], &[bit(true)], &[]);
    let window_only: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        Ok(vec![])
    });
    let cost = cells_of(&*sign_only, &[int(-5)]).0 - cells_of(&*window_only, &[int(-5)]).0;
    let comparison: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        s.ensure_signed64(&c[1])?;
        Ok(vec![s.lt_i(&c[0], &c[1])?])
    });
    let two_windows: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        s.ensure_signed64(&c[1])?;
        Ok(vec![])
    });
    let compare_cost = cells_of(&*comparison, &[int(-5), int(3)]).0 - cells_of(&*two_windows, &[int(-5), int(3)]).0;
    assert!(2 * cost < compare_cost, "a window sign costs {cost} advice cells, a comparison {compare_cost}");
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

/// The hinted quotient and remainder of a rescale (and of every division)
/// are witnesses of the division gate `r + d·q = a` itself: the rescale costs
/// that gate's four cells plus the remainder's range check and the
/// quotient's window check, and a shifted quotient hint is still rejected.
#[test]
fn quotient_and_remainder_are_cells_of_the_division_gate() {
    let rescale: Box<Op> = Box::new(|s, c| Ok(vec![s.rescale_by(&c[0], 32)?]));
    let checks: Box<Op> = Box::new(|s, c| {
        s.range_check(&c[0], 32);
        s.ensure_signed64(&c[1])?;
        Ok(vec![])
    });
    let raw = kernel::i128_to_fp((5i128 << 32) + 7);
    let (rescale_advice, rescale_lookup) = cells_of(&*rescale, &[raw]);
    let (checks_advice, checks_lookup) = cells_of(&*checks, &[int(7), int(5)]);
    // One input instead of two, and the gate's four cells.
    assert_eq!((rescale_advice, rescale_lookup), (checks_advice - 1 + 4, checks_lookup));
    assert_eq!(synthesize(&*rescale, &[raw], &[]).outputs, vec![int(5)]);
    for delta in [Fp::one(), -Fp::one()] {
        assert!(!synthesize(&*rescale, &[raw], &[(0, delta)]).accepted);
    }
}

/// Advice and lookup cells of `op` synthesized on free inputs.
fn cells_of(op: &Op, inputs: &[Fp]) -> (usize, usize) {
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Keygen);
    builder.set_k(K as usize);
    builder.set_lookup_bits(LOOKUP_BITS as usize);
    builder.set_instance_columns(1);
    let provenance = ProvenanceTable::new();
    let params = ProvingParams { k: K, precision_bits: PRECISION_BITS, lookup_bits: LOOKUP_BITS };
    {
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, None, params, &provenance);
        let cells: Vec<Halo2BaseCell> = inputs.iter().map(|v| synth.cell(*v)).collect();
        op(&mut synth, &cells).expect("synthesis failed");
    }
    let stats = builder.statistics();
    (stats.gate.total_advice_per_phase[0], stats.total_lookup_advice_per_phase[0])
}

fn bit(b: bool) -> Fp {
    if b { Fp::one() } else { Fp::zero() }
}

type Relation = fn(i64, i64) -> bool;

/// Each comparison's result bit is a hint pinned by one range check of
/// `(a - b) + s·2^n`; flipping it (or making it a non-bit) is rejected.
#[test]
fn comparisons_are_pinned() {
    let signed: Vec<(&str, Box<Op>, Relation)> = vec![
        ("lt", binary(|s, a, b| s.lt_i(a, b)), |a, b| a < b),
        ("lte", binary(|s, a, b| s.lte_i(a, b)), |a, b| a <= b),
        ("gt", binary(|s, a, b| s.gt_i(a, b)), |a, b| a > b),
        ("gte", binary(|s, a, b| s.gte_i(a, b)), |a, b| a >= b),
    ];
    let pairs = [(3i64, 5i64), (5, 3), (4, 4), (0, 7), (-3, 2), (2, -3), (-1, -1), (i64::MIN, i64::MAX), (i64::MAX, i64::MIN)];
    for (name, op, relation) in &signed {
        for (a, b) in pairs {
            assert_sound(&format!("{name}({a}, {b})"), &**op, &[int(a), int(b)], &[bit(relation(a, b))], &[]);
        }
    }
    let unsigned = binary(|s, a, b| s.lt_i_bounded(a, b));
    for (a, b) in [(3i64, 5i64), (5, 3), (4, 4), (0, i64::MAX), (i64::MAX, 0)] {
        assert_sound(&format!("lt_bounded({a}, {b})"), &*unsigned, &[int(a), int(b)], &[bit(a < b)], &[]);
    }
    // An unsigned comparison has no witness for a negative operand.
    assert!(!synthesize(&*unsigned, &[int(-1), int(3)], &[]).accepted);
    // A constant operand is folded into the checked difference.
    let below_seven = with_constant_rhs(|s, a, b| s.lt_i(a, b), int(7));
    for x in [6i64, 7, -100, i64::MAX] {
        assert_sound(&format!("{x} < 7"), &*below_seven, &[int(x)], &[bit(x < 7)], &[]);
    }
}

#[test]
fn equality_and_logic_have_no_free_hints() {
    let ops: Vec<(&str, Box<Op>)> = vec![
        ("eq", binary(|s, a, b| s.eq_i(a, b))),
        ("ne", binary(|s, a, b| s.ne_i(a, b))),
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

/// `x < 101` asserted, then `7 < x`: the assertion proves `x` in [0, 100],
/// so the comparison checks one lookup-width limb instead of the 64-bit
/// window, and a comparison the range decides costs nothing.
#[test]
fn proven_ranges_narrow_comparisons() {
    let narrow: Box<Op> = Box::new(|s, c| {
        let bound = s.constant_int(101)?;
        s.assert_u128_lt(&c[0], &bound)?;
        let seven = s.constant_int(7)?;
        Ok(vec![s.lt_i(&seven, &c[0])?])
    });
    for x in [0i64, 7, 8, 100] {
        assert_sound(&format!("7 < {x} (x < 101)"), &*narrow, &[int(x)], &[bit(7 < x)], &[]);
    }
    assert!(!synthesize(&*narrow, &[int(101)], &[]).accepted);
    assert!(!synthesize(&*narrow, &[int(-1)], &[]).accepted);
    let wide = binary(|s, a, b| s.lt_i(a, b));
    let (narrow_advice, narrow_lookup) = cells_of(&*narrow, &[int(5)]);
    let (wide_advice, wide_lookup) = cells_of(&*wide, &[int(5), int(7)]);
    assert!(narrow_advice * 3 < wide_advice && narrow_lookup * 5 < wide_lookup,
        "narrow ({narrow_advice}, {narrow_lookup}) vs 64-bit ({wide_advice}, {wide_lookup})");
    let decided: Box<Op> = Box::new(|s, c| {
        let bound = s.constant_int(101)?;
        s.assert_u128_lt(&c[0], &bound)?;
        let far = s.constant_int(1000)?;
        Ok(vec![s.lt_i(&c[0], &far)?])
    });
    let outcome = synthesize(&*decided, &[int(50)], &[]);
    assert!(outcome.accepted && outcome.outputs == vec![Fp::one()] && outcome.hints == 0);
}

/// The input-domain check the frontend emits per input element
/// (`assert x >= i64::MIN; assert x <= i64::MAX`) is one 64-bit window
/// check: the comparisons are decided by the window, so their negations and
/// the asserted constants cost no gates, and each constant is loaded once.
/// Values outside the window have no witness.
#[test]
fn input_domain_checks_are_one_window_check() {
    let domain_check: Box<Op> = Box::new(|s, c| {
        let (lo, hi) = (s.constant_int(i64::MIN)?, s.constant_int(i64::MAX)?);
        for x in c {
            let above = s.gte_i(x, &lo)?;
            s.assert_true(&above)?;
            let below = s.lte_i(x, &hi)?;
            s.assert_true(&below)?;
        }
        Ok(vec![])
    });
    for x in [0i64, -1, i64::MIN, i64::MAX] {
        let outcome = synthesize(&*domain_check, &[int(x)], &[]);
        assert!(outcome.accepted && outcome.hints == 0, "{x} rejected");
    }
    for outside in [kernel::i128_to_fp(1i128 << 63), kernel::i128_to_fp(-(1i128 << 63) - 1), kernel::i128_to_fp(1i128 << 70)] {
        assert!(!synthesize(&*domain_check, &[outside], &[]).accepted);
    }
    let window: Box<Op> = Box::new(|s, c| {
        s.ensure_signed64(&c[0])?;
        Ok(vec![])
    });
    let (one_advice, one_lookup) = cells_of(&*domain_check, &[int(5)]);
    let (two_advice, two_lookup) = cells_of(&*domain_check, &[int(5), int(6)]);
    let (window_advice, window_lookup) = cells_of(&*window, &[int(5)]);
    // Per element: the window check alone. The bounds and the truth
    // constants the asserted comparisons decide to get no cells.
    assert_eq!((two_advice - one_advice, two_lookup - one_lookup), (window_advice, window_lookup));
    assert_eq!((one_advice, one_lookup), (window_advice, window_lookup));
}

#[test]
fn unsigned_bound_assertion_is_exact() {
    for bound in [8i64, 101, 1 << 20] {
        let assert_below: Box<Op> = Box::new(move |s, c| {
            let hi = s.constant_int(bound)?;
            s.assert_u128_lt(&c[0], &hi)?;
            Ok(vec![])
        });
        for x in [0, bound - 1] {
            assert!(synthesize(&*assert_below, &[int(x)], &[]).accepted, "{x} < {bound} rejected");
        }
        for x in [bound, bound + 1, -1, 1 << 40] {
            assert!(!synthesize(&*assert_below, &[int(x)], &[]).accepted, "{x} < {bound} accepted");
        }
    }
    let dynamic = binary(|s, a, b| s.assert_u128_lt(a, b).map(|_| s.constant_int(0).unwrap()));
    assert!(synthesize(&*dynamic, &[int(4), int(5)], &[]).accepted);
    assert!(!synthesize(&*dynamic, &[int(5), int(5)], &[]).accepted);
    assert!(!synthesize(&*dynamic, &[int(-1), int(5)], &[]).accepted);
}

#[test]
fn signed_comparison_rejects_inputs_outside_the_64_bit_window() {
    let lt = binary(|s, a, b| s.lt_i(a, b));
    let huge = kernel::i128_to_fp(1i128 << 70);
    assert!(!synthesize(&*lt, &[huge, int(1)], &[]).accepted);
}

/// A compare-exchange `(min, max)`: both selects reuse the comparison's
/// difference `a - b` (one gate each) and stay pinned by the comparison bit.
#[test]
fn compare_exchange_shares_the_difference() {
    let exchange: Box<Op> = Box::new(|s, c| {
        let below = s.lt_i(&c[1], &c[0])?;
        Ok(vec![s.select(&below, &c[1], &c[0])?, s.select(&below, &c[0], &c[1])?])
    });
    for (a, b) in [(3i64, 5i64), (5, 3), (4, 4), (-7, 2), (i64::MAX, i64::MIN)] {
        assert_sound(&format!("exchange({a}, {b})"), &*exchange, &[int(a), int(b)], &[int(a.min(b)), int(a.max(b))], &[]);
    }
    let separate: Box<Op> = Box::new(|s, c| {
        let below = s.lt_i(&c[1], &c[0])?;
        let x = s.add_i(&c[0], &c[2])?; // operands the comparison did not subtract
        let y = s.add_i(&c[1], &c[2])?;
        Ok(vec![s.select(&below, &y, &x)?, s.select(&below, &x, &y)?])
    });
    let shared_cost = cells_of(&*exchange, &[int(3), int(5)]).0;
    let separate_cost = cells_of(&*separate, &[int(3), int(5), int(0)]).0;
    assert!(shared_cost + 8 <= separate_cost, "shared {shared_cost}, separate {separate_cost}");
}

/// `select(c, x + y, x)` (a masked increment) needs no difference gate:
/// the sum's difference from its summand is the other summand, so the select
/// is one gate `x + c·y`; likewise `select(c, x - y, x)` uses `y`. Pinned by
/// the comparison bit as any select.
#[test]
fn selects_of_a_sum_and_its_summand_are_one_gate() {
    let increment: Box<Op> = Box::new(|s, c| {
        let below = s.lt_i(&c[0], &c[1])?;
        let sum = s.add_i(&c[0], &c[2])?;
        let difference = s.sub_i(&c[0], &c[2])?;
        Ok(vec![s.select(&below, &sum, &c[0])?, s.select(&below, &difference, &c[0])?])
    });
    for (x, t, y) in [(3i64, 5i64, 7i64), (5, 3, 7), (-2, -2, 1)] {
        let below = x < t;
        let expected = [int(if below { x + y } else { x }), int(if below { x - y } else { x })];
        assert_sound(&format!("masked step {x} {t} {y}"), &*increment, &[int(x), int(t), int(y)], &expected, &[]);
    }
    let plain: Box<Op> = Box::new(|s, c| {
        let below = s.lt_i(&c[0], &c[1])?;
        let sum = s.add_i(&c[0], &c[2])?;
        let difference = s.sub_i(&c[0], &c[2])?;
        Ok(vec![sum, difference, below])
    });
    let (with_selects, without) = (cells_of(&*increment, &[int(1), int(2), int(3)]).0, cells_of(&*plain, &[int(1), int(2), int(3)]).0);
    assert_eq!(with_selects, without + 2 * 4, "two selects, one gate each");
}

/// A select between bits is a bit, and a two-valued cell's truth value is
/// its bit: neither needs an `is_zero` conversion when used as a condition.
#[test]
fn selects_of_bits_are_conditions_without_conversion() {
    let nested: Box<Op> = Box::new(|s, c| {
        let below = s.lt_i(&c[0], &c[1])?;
        let (zero, one, seven) = (s.constant_int(0)?, s.constant_int(1)?, s.constant_int(7)?);
        let flipped = s.select(&below, &zero, &one)?; // not(below), as a select
        let either = s.select(&flipped, &below, &one)?; // a select between bits
        let scaled = s.select(&below, &seven, &zero)?; // two-valued {0, 7}
        Ok(vec![s.select(&flipped, &c[0], &c[1])?, s.select(&either, &c[1], &c[0])?, s.logical_not(&scaled)?])
    });
    for (a, b) in [(1i64, 2i64), (2, 1), (3, 3)] {
        let below = a < b;
        let either = if !below { below } else { true };
        let expected = [int(if !below { a } else { b }), int(if either { b } else { a }), int(!below as i64)];
        assert_sound(&format!("nested({a}, {b})"), &*nested, &[int(a), int(b)], &expected, &[]);
        assert_eq!(synthesize(&*nested, &[int(a), int(b)], &[]).hints, 1, "only the comparison bit is a hint");
    }
}

/// `x < 0` on a fresh value proves the window and the sign with one check;
/// asserting `x >= 0` makes `x`'s range non-negative, and `sqrt` then needs
/// no sign handling.
#[test]
fn sign_tests_and_asserted_ranges_are_pinned() {
    let sign = with_constant_rhs(|s, a, b| s.lt_i(a, b), int(0));
    for x in [5i64, 0, -5, i64::MIN, i64::MAX] {
        assert_sound(&format!("{x} < 0"), &*sign, &[int(x)], &[bit(x < 0)], &[]);
    }
    assert!(!synthesize(&*sign, &[kernel::i128_to_fp(1i128 << 63)], &[]).accepted, "outside the window");
    let checked_root: Box<Op> = Box::new(|s, c| {
        let zero = s.constant_float(0.0)?;
        let nonnegative = s.gte_f(&c[0], &zero)?;
        s.assert_true(&nonnegative)?;
        Ok(vec![s.sqrt_f(&c[0])?])
    });
    for x in [2.0, 0.0, 1e6, 3.0e-9] {
        let expected = kernel::fp_sqrt(q(x), PRECISION_BITS);
        assert_sound(&format!("assert {x} >= 0; sqrt"), &*checked_root, &[q(x)], &[expected], &[]);
    }
    assert!(!synthesize(&*checked_root, &[q(-1.0)], &[]).accepted, "a negative input fails the assertion");
    let unchecked_root = unary(|s, a| s.sqrt_f(a));
    let assertion_only: Box<Op> = Box::new(|s, c| {
        let zero = s.constant_float(0.0)?;
        let nonnegative = s.gte_f(&c[0], &zero)?;
        s.assert_true(&nonnegative)?;
        Ok(vec![])
    });
    let with_assertion = cells_of(&*checked_root, &[q(2.0)]).0 - cells_of(&*assertion_only, &[q(2.0)]).0;
    assert!(with_assertion + 20 < cells_of(&*unchecked_root, &[q(2.0)]).0, "the asserted root still pays for the sign");
    // The sign bit sqrt's |x| pinned answers a later `x >= 0`: no new hint.
    let root_then_check: Box<Op> = Box::new(|s, c| {
        let root = s.sqrt_f(&c[0])?;
        let zero = s.constant_float(0.0)?;
        let nonnegative = s.gte_f(&c[0], &zero)?;
        s.assert_true(&nonnegative)?;
        Ok(vec![root])
    });
    for x in [2.0, 0.0] {
        let expected = kernel::fp_sqrt(q(x), PRECISION_BITS);
        assert_sound(&format!("sqrt({x}); assert >= 0"), &*root_then_check, &[q(x)], &[expected], &[]);
        assert_eq!(synthesize(&*root_then_check, &[q(x)], &[]).hints, 2, "sign and root only");
    }
    assert!(!synthesize(&*root_then_check, &[q(-2.0)], &[]).accepted);
    let abs = unary(|s, a| s.abs_i(a));
    assert_sound("abs(i64::MIN + 1)", &*abs, &[int(i64::MIN + 1)], &[int(i64::MAX)], &[]);
    // `|i64::MIN| = 2^63` overflows int64: no proof (the sign_and_abs path).
    assert!(!synthesize(&*abs, &[int(i64::MIN)], &[]).accepted, "|i64::MIN| has no proof");
    assert!(!synthesize(&*abs, &[kernel::i128_to_fp(1i128 << 63)], &[]).accepted, "outside the window");
    // A comparison after an asserted bound uses the bound.
    let bounded: Box<Op> = Box::new(|s, c| {
        let hundred = s.constant_int(100)?;
        let below = s.lt_i(&c[0], &hundred)?;
        s.assert_true(&below)?;
        let zero = s.constant_int(0)?;
        let nonnegative = s.gte_i(&c[0], &zero)?;
        s.assert_true(&nonnegative)?;
        let ten = s.constant_int(10)?;
        Ok(vec![s.lt_i(&c[0], &ten)?])
    });
    for x in [0i64, 9, 10, 99] {
        assert_sound(&format!("{x} in [0, 100): {x} < 10"), &*bounded, &[int(x)], &[bit(x < 10)], &[]);
    }
    assert!(!synthesize(&*bounded, &[int(100)], &[]).accepted);
}

// ── Bitwise operations (two's complement, as NumPy int64) ──────────────

const BIT_PAIRS: [(i64, i64); 7] =
    [(12, 10), (-12, 10), (12, -10), (-1, i64::MIN), (i64::MAX, -7), (0, 5), (i64::MIN, i64::MIN)];

#[test]
fn bitwise_operations_follow_twos_complement_and_are_pinned() {
    let ops: Vec<(&str, Box<Op>, fn(i64, i64) -> i64)> = vec![
        ("&", binary(|s, a, b| s.bit_and_i(a, b)), |a, b| a & b),
        ("|", binary(|s, a, b| s.bit_or_i(a, b)), |a, b| a | b),
        ("^", binary(|s, a, b| s.bit_xor_i(a, b)), |a, b| a ^ b),
    ];
    for (name, op, f) in &ops {
        for (a, b) in BIT_PAIRS {
            assert_sound(&format!("{a} {name} {b}"), &**op, &[int(a), int(b)], &[int(f(a, b))], &[]);
        }
    }
    let not = unary(|s, a| s.bit_not_i(a));
    for x in [0i64, 5, -6, i64::MAX, i64::MIN] {
        assert_sound(&format!("~{x}"), &*not, &[int(x)], &[int(!x)], &[]);
    }
    // An operand outside the int64 window has no witness.
    let and = binary(|s, a, b| s.bit_and_i(a, b));
    assert!(!synthesize(&*and, &[kernel::i128_to_fp(1i128 << 64), int(3)], &[]).accepted);
}

#[test]
fn bitwise_operations_with_constants_are_pinned() {
    for c in [1024i64, 5, 0, -1, -16, 255] {
        let and = with_constant_rhs(|s, a, b| s.bit_and_i(a, b), int(c));
        let or = with_constant_rhs(|s, a, b| s.bit_or_i(a, b), int(c));
        let xor = with_constant_rhs(|s, a, b| s.bit_xor_i(a, b), int(c));
        for x in [1234i64, -1234, 0, i64::MIN, i64::MAX] {
            assert_sound(&format!("{x} & {c}"), &*and, &[int(x)], &[int(x & c)], &[]);
            assert_sound(&format!("{x} | {c}"), &*or, &[int(x)], &[int(x | c)], &[]);
            assert_sound(&format!("{x} ^ {c}"), &*xor, &[int(x)], &[int(x ^ c)], &[]);
        }
    }
    // Narrow operands decompose only their own width.
    let narrow: Box<Op> = Box::new(|s, c| {
        let bound = s.constant_int(16)?;
        s.assert_u128_lt(&c[0], &bound)?;
        s.assert_u128_lt(&c[1], &bound)?;
        Ok(vec![s.bit_xor_i(&c[0], &c[1])?])
    });
    assert_sound("9 ^ 12 (< 16)", &*narrow, &[int(9), int(12)], &[int(9 ^ 12)], &[]);
    let wide = binary(|s, a, b| s.bit_xor_i(a, b));
    assert!(cells_of(&*narrow, &[int(9), int(12)]).0 * 4 < cells_of(&*wide, &[int(9), int(12)]).0);
}

/// `(x & 2^j) > 0` is the bit itself: a value known to be `bit ? v1 : v0`
/// compared with a constant reduces to the bit, its negation or a constant.
#[test]
fn comparisons_of_two_valued_cells_reduce_to_their_bit() {
    let masked: Box<Op> = Box::new(|s, c| {
        let mask = s.constant_int(4)?;
        let zero = s.constant_int(0)?;
        let bit = s.bit_and_i(&c[0], &mask)?;
        Ok(vec![s.gt_i(&bit, &zero)?, s.lt_i(&bit, &zero)?, s.eq_i(&bit, &mask)?, s.eq_i(&bit, &zero)?])
    });
    for x in [4i64, 3, -4, -5] {
        let set = (x & 4) != 0;
        let expected = [int(set as i64), int(0), int(set as i64), int(!set as i64)];
        assert_sound(&format!("({x} & 4) cmp"), &*masked, &[int(x)], &expected, &[]);
    }
    let selected: Box<Op> = Box::new(|s, c| {
        let (three, seven, five) = (s.constant_int(3)?, s.constant_int(7)?, s.constant_int(5)?);
        let cond = s.lt_i(&c[0], &c[1])?;
        let picked = s.select(&cond, &seven, &three)?;
        let shifted = s.add_i(&picked, &five)?;
        Ok(vec![s.lt_i(&picked, &five)?, s.gt_i(&shifted, &five)?, s.eq_i(&shifted, &five)?])
    });
    for (a, b) in [(1i64, 2i64), (2, 1)] {
        let picked = if a < b { 7 } else { 3 };
        let expected = [int((picked < 5) as i64), int((picked + 5 > 5) as i64), int(0)];
        let outcome = synthesize(&*selected, &[int(a), int(b)], &[]);
        assert!(outcome.accepted && outcome.outputs == expected, "select({a} < {b}, 7, 3)");
        assert_eq!(outcome.hints, 1, "only the condition's comparison loads a hint");
    }
}

// ── Memory (M3) ─────────────────────────────────────────────────────────

/// `x[i] = 7; x[0] = 9; return (x[0], x[i])` on a 4-slot segment with i = 0.
/// The constant write comes later in program order, so both reads see 9.
#[test]
fn constant_write_after_dynamic_write_wins() {
    let op: Box<Op> = Box::new(|s, c| {
        s.allocate_memory(1, 4, 0)?;
        let seven = s.constant_int(7)?;
        let nine = s.constant_int(9)?;
        let zero = s.constant_int(0)?;
        s.write_memory(1, &c[0], &seven)?;
        s.write_memory(1, &zero, &nine)?;
        let at_zero = s.read_memory(1, &zero)?;
        let at_i = s.read_memory(1, &c[0])?;
        Ok(vec![at_zero, at_i])
    });
    assert_sound("x[i]=7; x[0]=9", &*op, &[int(0)], &[int(9), int(9)], &[]);
    // With i = 2 the constant write does not alias: x[0] = 9, x[i] = 7.
    assert_sound("x[i]=7; x[0]=9 (i=2)", &*op, &[int(2)], &[int(9), int(7)], &[]);
}

/// The later dynamic write still overrides an earlier constant one.
#[test]
fn dynamic_write_after_constant_write_wins() {
    let op: Box<Op> = Box::new(|s, c| {
        s.allocate_memory(1, 4, 0)?;
        let one = s.constant_int(1)?;
        let five = s.constant_int(5)?;
        let six = s.constant_int(6)?;
        s.write_memory(1, &one, &five)?;
        s.write_memory(1, &c[0], &six)?;
        Ok(vec![s.read_memory(1, &one)?])
    });
    assert_sound("x[1]=5; x[i]=6 (i=1)", &*op, &[int(1)], &[int(6)], &[]);
    assert_sound("x[1]=5; x[i]=6 (i=3)", &*op, &[int(3)], &[int(5)], &[]);
}

#[test]
fn dynamic_write_out_of_bounds_is_rejected() {
    let op: Box<Op> = Box::new(|s, c| {
        s.allocate_memory(1, 4, 0)?;
        let seven = s.constant_int(7)?;
        s.write_memory(1, &c[0], &seven)?;
        Ok(vec![])
    });
    assert!(synthesize(&*op, &[int(3)], &[]).accepted);
    assert!(!synthesize(&*op, &[int(4)], &[]).accepted);
    assert!(!synthesize(&*op, &[int(-1)], &[]).accepted);
}

/// A dynamic read is the inner product of the address's indicators with
/// the slots; the membership assertion makes exactly one indicator 1, so
/// the read is the addressed slot and an out-of-bounds address has no
/// witness.
#[test]
fn dynamic_read_selects_the_addressed_slot() {
    let op: Box<Op> = Box::new(|s, c| {
        s.allocate_memory(1, 4, 0)?;
        for (i, v) in [10i64, 11, 12].iter().enumerate() {
            let addr = s.constant_int(i as i64)?;
            let value = s.constant_int(*v)?;
            s.write_memory(1, &addr, &value)?;
        }
        Ok(vec![s.read_memory(1, &c[0])?])
    });
    for (i, expected) in [(0i64, 10i64), (1, 11), (2, 12), (3, 0)] {
        assert_sound(&format!("x[{i}]"), &*op, &[int(i)], &[int(expected)], &[]);
    }
    assert!(!synthesize(&*op, &[int(4)], &[]).accepted);
    assert!(!synthesize(&*op, &[int(-1)], &[]).accepted);
}

/// A scatter `b[i, idx_i] = 1` into a 3x3 zero matrix, with each `idx_i`
/// asserted below 3, then every cell read back: a read of row `r` only
/// compares its address with the write of row `r` (the other writes' proven
/// address ranges exclude it), and the result is NumPy's one-hot matrix.
#[test]
fn overlay_skips_writes_whose_addresses_cannot_reach_the_read() {
    let scatter: Box<Op> = Box::new(|s, c| {
        s.allocate_memory(1, 9, 0)?;
        let (one, three) = (s.constant_int(1)?, s.constant_int(3)?);
        for (row, idx) in c.iter().enumerate() {
            s.assert_u128_lt(idx, &three)?;
            let base = s.constant_int(3 * row as i64)?;
            let addr = s.add_i(&base, idx)?;
            s.write_memory(1, &addr, &one)?;
        }
        (0..9).map(|k| {
            let addr = s.constant_int(k)?;
            s.read_memory(1, &addr)
        }).collect()
    });
    for idx in [[0i64, 2, 1], [1, 1, 1], [2, 0, 0]] {
        let expected: Vec<Fp> = (0..9).map(|k| int((idx[(k / 3) as usize] == k % 3) as i64)).collect();
        let inputs: Vec<Fp> = idx.iter().map(|v| int(*v)).collect();
        assert_sound(&format!("one-hot {idx:?}"), &*scatter, &inputs, &expected, &[]);
    }
    assert!(!synthesize(&*scatter, &[int(3), int(0), int(0)], &[]).accepted, "index 3 accepted");
    // 9 reads, each comparing with one write: 9 equality tests, not 27.
    let one_row: Box<Op> = Box::new(|s, c| {
        s.allocate_memory(1, 9, 0)?;
        let (one, three) = (s.constant_int(1)?, s.constant_int(3)?);
        s.assert_u128_lt(&c[0], &three)?;
        s.write_memory(1, &c[0], &one)?;
        (0..3).map(|k| {
            let addr = s.constant_int(k)?;
            s.read_memory(1, &addr)
        }).collect()
    });
    let (rows, _) = cells_of(&*scatter, &[int(0), int(0), int(0)]);
    let (row, _) = cells_of(&*one_row, &[int(0)]);
    // (Comparing every read with every write would cost about 9x one row.)
    assert!(rows < 4 * row, "three rows {rows} cells, one row {row}");
}

/// Reads after many runtime-address writes cost one address comparison per
/// pending write; past the budget synthesis stops with an error naming the
/// construct (a 20,000 x 20,000 scatter exhausted 16 GB before).
#[test]
fn overlay_beyond_its_budget_is_a_synthesis_error() {
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Keygen);
    builder.set_k(K as usize);
    builder.set_lookup_bits(LOOKUP_BITS as usize);
    let provenance = ProvenanceTable::new();
    let params = ProvingParams { k: K, precision_bits: PRECISION_BITS, lookup_bits: LOOKUP_BITS };
    let mut synth = Halo2BaseSynthesizer::new(&mut builder, None, params, &provenance);
    synth.memory.overlay_budget = Some(10);
    synth.allocate_memory(1, 8, 0).unwrap();
    let addresses: Vec<Halo2BaseCell> = (0..4).map(|i| synth.cell(int(i))).collect();
    for addr in &addresses {
        let value = synth.constant_int(1).unwrap();
        synth.write_memory(1, addr, &value).unwrap();
    }
    let zero = synth.constant_int(0).unwrap();
    let one = synth.constant_int(1).unwrap();
    assert!(synth.read_memory(1, &zero).is_ok(), "4 tests are within the budget");
    assert!(synth.read_memory(1, &one).is_ok(), "8 tests are within the budget");
    let error = synth.read_memory(1, &zero).map(|_| ()).unwrap_err().to_string();
    assert!(error.contains("runtime-address writes"), "{error}");
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
        let recomposed = s.gate_cell(b.value * q.value + r.value, |gate, ctx| gate.mul_add(ctx, b.quantum(), q.quantum(), r.quantum()));
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
            gate.mul_add(ctx, result.quantum(), QuantumCell::Constant(scale), rem.quantum())
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

// ── Whole-graph plans: planned bounds, fused sums of products ──────────

mod graphs {
    use super::*;
    use crate::circuit_input::{InputNode, InputParam};
    use crate::ir::{IRGraph, IRStatement};
    use crate::ir_defs::IR;
    use crate::prove::interpreter::interpret_ir;
    use crate::types::ValueId;

    /// Small IR graph builder: statement ids are indices.
    #[derive(Default)]
    pub(super) struct Graph {
        stmts: Vec<IRStatement>,
        vids: Vec<ValueId>,
    }

    impl Graph {
        pub fn push(&mut self, ir: IR, args: &[u32]) -> u32 {
            let id = self.stmts.len() as u32;
            let vid = ValueId::next();
            let arg_values = args.iter().map(|a| self.vids[*a as usize]).collect();
            self.stmts.push(IRStatement::new(id, vid, ir, args.to_vec(), arg_values, None));
            self.vids.push(vid);
            id
        }

        pub fn read(&mut self, name: &str) -> u32 {
            self.push(IR::ReadInteger { path: InputPath::new(name, vec![]), is_public: false }, &[])
        }

        pub fn int(&mut self, value: i64) -> u32 {
            self.push(IR::ConstantInt { value }, &[])
        }

        /// The frontend's input-domain check of an int64 input.
        pub fn domain_check(&mut self, x: u32) {
            let (lo, hi) = (self.int(i64::MIN), self.int(i64::MAX));
            let above = self.push(IR::GteI, &[x, lo]);
            self.push(IR::Assert, &[above]);
            let below = self.push(IR::LteI, &[x, hi]);
            self.push(IR::Assert, &[below]);
        }

        pub fn finish(self) -> IRGraph {
            IRGraph::new(self.stmts)
        }
    }

    pub(super) struct Run {
        pub outputs: Vec<Fp>,
        pub accepted: bool,
        pub advice: usize,
    }

    /// Synthesize `ir` (with its plan) on the integer inputs `ints` and run
    /// the MockProver against `claimed` public values (the synthesized
    /// outputs when `None`).
    pub(super) fn run(ir: &IRGraph, ints: &[(&str, i64)], claimed: Option<Vec<Fp>>) -> Run {
        let mut inputs = CircuitInputs::new();
        for (name, value) in ints {
            inputs.params.push(InputParam {
                name: name.to_string(),
                is_public: false,
                dtype: serde_json::Value::Null,
                value: InputNode::Int(*value),
            });
        }
        let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Mock);
        builder.set_k(K as usize);
        builder.set_lookup_bits(LOOKUP_BITS as usize);
        builder.set_instance_columns(1);
        let params = ProvingParams { k: K, precision_bits: PRECISION_BITS, lookup_bits: LOOKUP_BITS };
        let outputs = {
            let witness = ResolvedWitness::new(inputs, PRECISION_BITS);
            let mut synth = Halo2BaseSynthesizer::new(&mut builder, Some(witness), params, &ir.provenance);
            synth.plan_ahead(ir);
            interpret_ir(ir, &mut synth).expect("synthesis failed");
            synth.public_values().to_vec()
        };
        let advice = builder.statistics().gate.total_advice_per_phase[0];
        builder.calculate_params(Some(20));
        let instance = claimed.unwrap_or_else(|| outputs.clone());
        let accepted = MockProver::run(K, &builder, vec![instance]).expect("MockProver::run").verify().is_ok();
        Run { outputs, accepted, advice }
    }
}

/// A constant gets an advice cell only where a gadget needs an assigned
/// cell: as an operand of a gate it is the gate's own constant cell (`x + 5`
/// costs one cell less than `x + y`), as a public output it is loaded and
/// bound to the public value, and asserting a false constant still leaves
/// the circuit unsatisfiable.
#[test]
fn constants_get_cells_only_where_needed() {
    use crate::ir_defs::IR;
    use graphs::*;
    let sum = |constant: bool| {
        let mut g = Graph::default();
        let x = g.read("x");
        let y = if constant { g.int(5) } else { g.read("y") };
        let s = g.push(IR::AddI, &[x, y]);
        g.push(IR::ExposePublicI, &[s]);
        g.finish()
    };
    let with_constant = run(&sum(true), &[("x", 3)], None);
    assert!(with_constant.accepted && with_constant.outputs == vec![int(8)]);
    assert!(!run(&sum(true), &[("x", 3)], Some(vec![int(9)])).accepted);
    assert_eq!(with_constant.advice + 1, run(&sum(false), &[("x", 3), ("y", 5)], None).advice);

    let mut g = Graph::default();
    let seven = g.int(7);
    g.push(IR::ExposePublicI, &[seven]);
    let exposed = g.finish();
    assert!(run(&exposed, &[], None).accepted);
    assert!(!run(&exposed, &[], Some(vec![int(8)])).accepted);

    for (value, holds) in [(true, true), (false, false)] {
        let mut g = Graph::default();
        let c = g.push(IR::ConstantBool { value }, &[]);
        g.push(IR::Assert, &[c]);
        assert_eq!(run(&g.finish(), &[], None).accepted, holds);
    }
}

/// An input that the circuit also bounds with `AssertU128Lt(x, 8)` (an
/// index) gets that bound check in place of its 64-bit window check: the
/// domain checks then cost nothing, and values outside `[0, 8)` (including
/// ones outside the int64 window) have no witness.
#[test]
fn domain_checks_of_bounded_inputs_use_the_bound() {
    use crate::ir_defs::IR;
    use graphs::*;
    let build = |with_domain_check: bool| {
        let mut g = Graph::default();
        let x = g.read("x");
        if with_domain_check {
            g.domain_check(x);
        }
        let eight = g.int(8);
        g.push(IR::AssertU128Lt, &[x, eight]);
        g.push(IR::ExposePublicI, &[x]);
        g.finish()
    };
    let (checked, bare) = (build(true), build(false));
    for x in [0i64, 7] {
        let outcome = run(&checked, &[("x", x)], None);
        assert!(outcome.accepted && outcome.outputs == vec![int(x)], "{x} rejected");
    }
    for x in [8i64, -1, i64::MAX, i64::MIN] {
        assert!(!run(&checked, &[("x", x)], None).accepted, "{x} accepted");
    }
    // The bound check implies the domain check, whose bounds and decided
    // `true` need no cells.
    assert_eq!(run(&checked, &[("x", 3)], None).advice, run(&bare, &[("x", 3)], None).advice);
}

/// `acc + a·b` and `acc - a·b` with a product used only there are one gate
/// (no product cell); a chain of them shares the accumulator cell between
/// gates (three cells per term). The outputs equal the integer arithmetic,
/// a wrong claimed output is rejected, and a product used twice is still a
/// cell of its own.
#[test]
fn sums_of_products_fuse_into_one_gate_per_term() {
    use crate::ir_defs::IR;
    use graphs::*;
    let dot = |terms: usize| {
        let mut g = Graph::default();
        let reads: Vec<(u32, u32)> = (0..terms).map(|i| (g.read(&format!("a{i}")), g.read(&format!("b{i}")))).collect();
        let mut acc = g.push(IR::MulI, &[reads[0].0, reads[0].1]);
        for (a, b) in &reads[1..] {
            let product = g.push(IR::MulI, &[*a, *b]);
            acc = g.push(IR::AddI, &[acc, product]);
        }
        let (c, d) = (g.read("c"), g.read("d"));
        let product = g.push(IR::MulI, &[c, d]);
        let diff = g.push(IR::SubI, &[acc, product]);
        g.push(IR::ExposePublicI, &[diff]);
        g.finish()
    };
    let values = |terms: usize| -> Vec<(String, i64)> {
        let mut v: Vec<(String, i64)> = (0..terms).flat_map(|i| [(format!("a{i}"), 3 + i as i64), (format!("b{i}"), -5 + 2 * i as i64)]).collect();
        v.push(("c".to_string(), 7));
        v.push(("d".to_string(), -2));
        v
    };
    let expected = |terms: usize| -> i64 { (0..terms as i64).map(|i| (3 + i) * (-5 + 2 * i)).sum::<i64>() + 14 };
    for terms in [2usize, 3, 4] {
        let ir = dot(terms);
        let named = values(terms);
        let ints: Vec<(&str, i64)> = named.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let honest = run(&ir, &ints, None);
        assert!(honest.accepted && honest.outputs == vec![int(expected(terms))], "{terms} terms");
        assert!(!run(&ir, &ints, Some(vec![int(expected(terms) + 1)])).accepted, "{terms} terms: wrong sum accepted");
    }
    let cells = |terms: usize| {
        let named = values(terms);
        let ints: Vec<(&str, i64)> = named.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        run(&dot(terms), &ints, None).advice
    };
    // One more term: two loaded inputs and three cells of the chained gate.
    assert_eq!(cells(4) - cells(3), 2 + 3);

    // A product read twice keeps its own cell and both uses see it.
    let mut g = Graph::default();
    let (a, b, c) = (g.read("a"), g.read("b"), g.read("c"));
    let product = g.push(IR::MulI, &[a, b]);
    let sum = g.push(IR::AddI, &[c, product]);
    let twice = g.push(IR::AddI, &[sum, product]);
    g.push(IR::ExposePublicI, &[twice]);
    let ir = g.finish();
    let outcome = run(&ir, &[("a", 4), ("b", 5), ("c", 1)], None);
    assert!(outcome.accepted && outcome.outputs == vec![int(41)]);
}

/// An asserted `==` is a pinned difference (or a pinned cell against a
/// constant), no equality bit: accepted exactly where the operands are
/// equal, and cheaper than the bit.
#[test]
fn asserted_equalities_are_pinned_differences() {
    use crate::ir_defs::IR;
    use graphs::*;
    let asserted = |constant: Option<i64>, assert: bool| {
        let mut g = Graph::default();
        let x = g.read("x");
        g.domain_check(x);
        let rhs = match constant {
            Some(c) => g.int(c),
            None => g.read("y"),
        };
        let equal = g.push(IR::EqI, &[x, rhs]);
        if assert {
            g.push(IR::Assert, &[equal]);
        } else {
            g.push(IR::ExposePublicI, &[equal]);
        }
        g.push(IR::ExposePublicI, &[x]);
        g.finish()
    };
    for constant in [Some(3), Some(0), None] {
        let ir = asserted(constant, true);
        for (x, y) in [(3i64, 3i64), (3, 4), (0, 0), (-5, -5), (7, 3)] {
            let holds = x == constant.unwrap_or(y);
            let accepted = run(&ir, &[("x", x), ("y", y)], None).accepted;
            assert_eq!(accepted, holds, "{constant:?} at x = {x}, y = {y}");
        }
        let pinned = run(&ir, &[("x", 3), ("y", 3)], None).advice;
        let bit = run(&asserted(constant, false), &[("x", 3), ("y", 3)], None).advice;
        assert!(pinned < bit, "{constant:?}: asserted {pinned} cells, bit {bit}");
    }
}

/// A comparison whose only use is `Assert` is one range check of its gap
/// (no result bit, no negation): accepted exactly where the relation holds,
/// for constants on either side, strict and non-strict, and two cells.
#[test]
fn asserted_comparisons_check_their_gap() {
    use crate::ir_defs::IR;
    use graphs::*;
    let asserted = |op: IR, constant: Option<(i64, bool)>| {
        let mut g = Graph::default();
        let x = g.read("x");
        g.domain_check(x);
        let (lhs, rhs) = match constant {
            Some((c, on_left)) => {
                let c = g.int(c);
                if on_left { (c, x) } else { (x, c) }
            }
            None => (x, g.read("y")),
        };
        let relation = g.push(op, &[lhs, rhs]);
        g.push(IR::Assert, &[relation]);
        g.push(IR::ExposePublicI, &[x]);
        g.finish()
    };
    type Case = (IR, Option<(i64, bool)>, fn(i64, i64) -> bool);
    let cases: Vec<Case> = vec![
        (IR::GteI, Some((0, false)), |x, _| x >= 0),
        (IR::GtI, Some((3, false)), |x, _| x > 3),
        (IR::LtI, Some((3, true)), |x, _| 3 < x),
        (IR::LteF, Some((-2, false)), |x, _| x <= -2),
        (IR::LtI, None, |x, y| x < y),
        (IR::GteF, None, |x, y| x >= y),
    ];
    for (op, constant, holds) in cases {
        let ir = asserted(op.clone(), constant);
        for (x, y) in [(-3i64, 4i64), (-2, -2), (0, -1), (3, 3), (4, 9), (i64::MAX, i64::MIN), (i64::MIN, i64::MAX)] {
            let accepted = run(&ir, &[("x", x), ("y", y)], None).accepted;
            assert_eq!(accepted, holds(x, y), "{op:?} {constant:?} at x = {x}, y = {y}");
        }
    }
    // `x >= 0` on a value in the window is one 64-bit check of x itself.
    let (with, without) = (asserted(IR::GteI, Some((0, false))), {
        let mut g = Graph::default();
        let x = g.read("x");
        g.domain_check(x);
        g.push(IR::ExposePublicI, &[x]);
        g.finish()
    });
    let extra = run(&with, &[("x", 5)], None).advice - run(&without, &[("x", 5)], None).advice;
    assert!(extra <= 1 + 3 * 8 - 2, "asserted x >= 0 costs {extra} cells");
}

/// Synthesize `op` honestly but check the circuit against `claimed` public
/// values: a wrong claim must be rejected.
fn accepts_claim(op: &Op, inputs: &[Fp], claimed: &[Fp]) -> bool {
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Mock);
    builder.set_k(K as usize);
    builder.set_lookup_bits(LOOKUP_BITS as usize);
    builder.set_instance_columns(1);
    let provenance = ProvenanceTable::new();
    let params = ProvingParams { k: K, precision_bits: PRECISION_BITS, lookup_bits: LOOKUP_BITS };
    let witness = ResolvedWitness::new(CircuitInputs::new(), PRECISION_BITS);
    {
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, Some(witness), params, &provenance);
        let cells: Vec<Halo2BaseCell> = inputs.iter().map(|v| synth.cell(*v)).collect();
        for out in op(&mut synth, &cells).expect("synthesis failed") {
            synth.expose_public(&out, "out").unwrap();
        }
    }
    builder.calculate_params(Some(20));
    MockProver::run(K, &builder, vec![claimed.to_vec()]).expect("MockProver::run").verify().is_ok()
}

/// `[a == b]` by the seven-cell zero test of `a - b` (of `a` itself against
/// the constant 0): the honest bit is accepted, the other one rejected.
#[test]
fn equality_tests_are_pinned() {
    let eq = binary(|s, a, b| s.eq_i(a, b));
    let eq_zero = with_constant_rhs(|s, a, b| s.eq_i(a, b), int(0));
    for (a, b) in [(3i64, 3i64), (3, 4), (-1, -1), (0, 0), (i64::MIN, i64::MAX), (0, 1)] {
        let honest = synthesize(&*eq, &[int(a), int(b)], &[]);
        assert!(honest.accepted && honest.outputs == vec![bit(a == b)], "{a} == {b}");
        assert!(!accepts_claim(&*eq, &[int(a), int(b)], &[bit(a != b)]), "{a} == {b}: flipped bit accepted");
        let honest = synthesize(&*eq_zero, &[int(a)], &[]);
        assert!(honest.accepted && honest.outputs == vec![bit(a == 0)], "{a} == 0");
        assert!(!accepts_claim(&*eq_zero, &[int(a)], &[bit(a != 0)]), "{a} == 0: flipped bit accepted");
    }
    let (with_zero, general) = (cells_of(&*eq_zero, &[int(5)]).0, cells_of(&*eq, &[int(5), int(6)]).0);
    assert!(with_zero + 4 <= general, "x == 0 costs {with_zero}, a == b {general}");
    // `!=` and truthiness by the eight-cell non-zero test.
    let ne = binary(|s, a, b| s.ne_i(a, b));
    let truth = unary(|s, a| s.bool_cast(a));
    for (a, b) in [(3i64, 3i64), (3, 4), (0, 0), (i64::MIN, i64::MAX)] {
        let honest = synthesize(&*ne, &[int(a), int(b)], &[]);
        assert!(honest.accepted && honest.outputs == vec![bit(a != b)], "{a} != {b}");
        assert!(!accepts_claim(&*ne, &[int(a), int(b)], &[bit(a == b)]), "{a} != {b}: flipped bit accepted");
        let honest = synthesize(&*truth, &[int(a)], &[]);
        assert!(honest.accepted && honest.outputs == vec![bit(a != 0)], "bool({a})");
        assert!(!accepts_claim(&*truth, &[int(a)], &[bit(a == 0)]), "bool({a}): flipped bit accepted");
    }
    // One cell more than `==` (eight-cell test against seven), not the four
    // of a negation gate.
    assert!(cells_of(&*ne, &[int(5), int(6)]).0 <= general + 1, "a != b costs more than a == b plus one");
}

/// An asserted `x != 0` is the pinned inverse `x·inv = 1`; division by a
/// divisor so asserted and proven non-negative needs neither the zero
/// mapping nor a sign: the quotient equals the kernel's, every hint is
/// pinned, and a zero divisor has no witness.
#[test]
fn divisions_by_asserted_nonzero_divisors_are_pinned() {
    use crate::ir_defs::IR;
    use graphs::*;
    let mut g = Graph::default();
    let (a, b) = (g.read("a"), g.read("b"));
    g.domain_check(a);
    g.domain_check(b);
    let zero = g.int(0);
    let non_negative = g.push(IR::GteI, &[b, zero]);
    g.push(IR::Assert, &[non_negative]);
    let nonzero = g.push(IR::NeI, &[b, zero]);
    g.push(IR::Assert, &[nonzero]);
    let quotient = g.push(IR::FloorDivI, &[a, b]);
    let remainder = g.push(IR::ModI, &[a, b]);
    g.push(IR::ExposePublicI, &[quotient]);
    g.push(IR::ExposePublicI, &[remainder]);
    let ir = g.finish();
    for (x, y) in [(7i64, 2i64), (-7, 2), (0, 5), (i64::MAX, 3), (i64::MIN, 7)] {
        let (q, r) = kernel::fp_floor_div(int(x), int(y));
        let honest = run(&ir, &[("a", x), ("b", y)], None);
        assert!(honest.accepted && honest.outputs == vec![q, r], "{x} // {y}");
        assert!(!run(&ir, &[("a", x), ("b", y)], Some(vec![q + Fp::one(), r - int(y)])).accepted, "{x} // {y}: shifted quotient accepted");
    }
    assert!(!run(&ir, &[("a", 7), ("b", 0)], None).accepted, "division by zero accepted");
    assert!(!run(&ir, &[("a", 7), ("b", -2)], None).accepted, "negative divisor accepted");
    // The asserted inverse is pinned: perturbing it is rejected.
    let distinct: Box<Op> = Box::new(|s, c| {
        s.plan.asserted_comparisons.insert(s.current_stmt);
        let zero = s.constant_int(0)?;
        Ok(vec![s.ne_i(&c[0], &zero)?])
    });
    for x in [1i64, -5, i64::MAX] {
        assert_sound(&format!("assert {x} != 0"), &*distinct, &[int(x)], &[Fp::one()], &[]);
    }
    assert!(!synthesize(&*distinct, &[int(0)], &[]).accepted, "0 != 0 accepted");
}

// ── Prover hints (IR::Hint) ─────────────────────────────────────────────

/// `prover_hint` loads the kernel's lane as a hint cell and constrains
/// nothing: a changed lane is accepted. That is the contract of `IR::Hint`
/// (the IR that uses a hint pins it); the pinned uses are tested on whole
/// lowered graphs in `lower::sort`.
#[test]
fn prover_hints_are_kernel_lanes_and_constrain_nothing() {
    use crate::ir_defs::HintKind;
    let values = [int(30), int(10), int(20), int(10), int(-40)];
    let op: Box<Op> = Box::new(|s, c| {
        let rank = s.constant_int(2)?;
        let argsort = s.prover_hint(HintKind::StableArgsort, &c[..5], &rank)?;
        let sorted = s.prover_hint(HintKind::SortedValues, &c[..5], &c[5])?;
        Ok(vec![argsort, sorted])
    });
    let mut inputs = values.to_vec();
    inputs.push(int(4)); // a runtime rank
    let honest = synthesize(&*op, &inputs, &[]);
    assert!(honest.accepted);
    assert_eq!(honest.outputs, vec![int(3), int(30)]);
    assert_eq!(honest.hints, 2);
    let tampered = synthesize(&*op, &inputs, &[(0, Fp::one())]);
    assert!(tampered.accepted, "an unpinned hint is a free cell");
    assert_eq!(tampered.outputs, vec![int(4), int(30)]);
}

// ── Sort lowerings (lower::sort) ──────────────────────────────

/// Synthesize a whole IR graph on `witness`, perturbing hints as in
/// [`synthesize`].
fn synthesize_graph(graph: &crate::ir::IRGraph, witness: &ResolvedWitness, tamper: &[(usize, Fp)]) -> Outcome {
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Mock);
    builder.set_k(K as usize);
    builder.set_lookup_bits(LOOKUP_BITS as usize);
    builder.set_instance_columns(1);
    let params = ProvingParams { k: K, precision_bits: PRECISION_BITS, lookup_bits: LOOKUP_BITS };
    let (outputs, hints) = {
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, Some(witness.clone()), params, &graph.provenance);
        synth.plan_ahead(graph);
        synth.tamper = tamper::HintTamper::targeting(tamper);
        crate::prove::interpreter::interpret_ir(graph, &mut synth).expect("synthesis failed");
        (synth.public_values().to_vec(), synth.tamper.hints_seen())
    };
    builder.calculate_params(Some(20));
    let accepted = MockProver::run(K, &builder, vec![outputs.clone()])
        .expect("MockProver::run")
        .verify()
        .is_ok();
    Outcome { outputs, hints, accepted }
}

/// Each sort lowering with hints, on values with ties: the honest witness
/// is accepted with NumPy's stable sort as outputs, and every hint (the
/// sort's own and every backend hint under it) moved by any of the deltas
/// is rejected. The deltas `±1 .. ±(n - 1)` turn an argsort lane into every
/// other index (a duplicate index, a non-permutation, the tie partner of a
/// tied element) and an order statistic into its neighbours.
#[test]
fn sort_hints_are_pinned() {
    use crate::lower::sort::test_support::{sort_graph, witness, Forced, Output};
    let values = [3, 1, 3, 0, 1];
    let n = values.len() as u32;
    let every: Vec<Output> = (0..n).flat_map(|r| [Output::Value(r), Output::Index(r)]).collect();
    let cases: Vec<(Forced, Vec<Output>)> = vec![
        (Forced::Batcher, every.clone()),
        (Forced::PackedBatcher, (0..n).map(Output::Index).collect()),
        (Forced::Permutation, every),
        (Forced::Runs, vec![Output::Index(3), Output::Index(4)]),
        (Forced::Runs, vec![Output::Index(1), Output::Index(2)]),
        (Forced::Counting, vec![Output::Value(2)]),
        (Forced::Extremes, vec![Output::Value(0), Output::Value(4)]),
        (Forced::MaskedExtremes, vec![Output::EqualsEach(4)]),
        (Forced::Runs, vec![Output::ValueAt(1), Output::IndexAt(3)]),
        (Forced::Batcher, vec![Output::Payload(0), Output::Payload(3)]),
        (Forced::Runs, vec![Output::Payload(1), Output::Payload(4)]),
    ];
    let mut shifts: Vec<Fp> = (1..n as i64).flat_map(|d| [int(d), int(-d)]).collect();
    shifts.extend(deltas(&[]));
    for (forced, outputs) in cases {
        let sort = sort_graph(&values, &outputs, forced);
        let w = witness(&sort.inputs);
        let honest = synthesize_graph(&sort.graph, &w, &[]);
        assert!(honest.accepted, "{forced:?} {outputs:?}: honest witness rejected");
        let expected: Vec<Fp> = sort.expected.as_ref().unwrap().iter().map(|&v| int(v)).collect();
        assert_eq!(honest.outputs, expected, "{forced:?} {outputs:?}");
        for index in 0..honest.hints {
            for shift in &shifts {
                let tampered = synthesize_graph(&sort.graph, &w, &[(index, *shift)]);
                assert!(
                    !tampered.accepted,
                    "{forced:?} {outputs:?}: hint #{index} moved by {shift:?} was accepted (outputs {:?})",
                    tampered.outputs,
                );
            }
        }
    }
}

/// `np.searchsorted` over a constant table by a hinted position
/// (`lower::sort::search`), closed form (an ECDF table) and reads (a table
/// with ties): NumPy's count, and every hint under it (the position, the
/// divisions' quotients, the comparisons' bits) moved by any delta is
/// rejected.
#[test]
fn searchsorted_position_is_pinned() {
    use crate::lower::sort::search::search_graph;
    use crate::lower::sort::test_support::witness;
    let ecdf: Vec<f64> = (1..=10).map(|k| k as f64 / 10.0).collect();
    let tied = vec![0.1, 0.25, 0.25, 0.7, 2.0];
    let q = |v: f64| kernel::fp_to_i64(q(v));
    for table in [ecdf, tied] {
        let n = table.len() as i64;
        let mut shifts: Vec<Fp> = (1..=n + 1).flat_map(|d| [int(d), int(-d)]).collect();
        shifts.extend(deltas(&[]));
        for right in [true, false] {
            for t in [q(-1.0), q(0.25) - 1, q(0.25), q(0.5), q(0.7), q(5.0)] {
                let search = search_graph(&table, true, right, t, true);
                let w = witness(&search.inputs);
                let honest = synthesize_graph(&search.graph, &w, &[]);
                assert!(honest.accepted, "t {t}: honest witness rejected");
                assert_eq!(honest.outputs, vec![int(search.expected)], "t {t} right {right}");
                for index in 0..honest.hints {
                    for shift in &shifts {
                        let tampered = synthesize_graph(&search.graph, &w, &[(index, *shift)]);
                        assert!(!tampered.accepted, "t {t} right {right}: hint #{index} moved by {shift:?} accepted");
                    }
                }
            }
        }
    }
}

/// `IR::KeyLess` by packed keys: the lexicographic order on window
/// extremes, ties and index order, and its comparison hint pinned.
#[test]
fn packed_key_comparison_is_lexicographic_and_pinned() {
    let op: Box<Op> = Box::new(|s, c| Ok(vec![s.key_less(&c[0], &c[1], &c[2], &c[3], 7)?]));
    let max = i64::MAX;
    let min = i64::MIN;
    let cases: [(i64, i64, i64, i64); 8] = [
        (1, 5, 2, 0),
        (2, 0, 1, 5),
        (3, 4, 3, 5),
        (3, 5, 3, 4),
        (3, 5, 3, 5),
        (min, 127, max, 0),
        (max, 0, min, 127),
        (-1, 0, 0, 0),
    ];
    for (v1, i1, v2, i2) in cases {
        let expected = (v1, i1) < (v2, i2);
        assert_sound(
            &format!("({v1}, {i1}) < ({v2}, {i2})"),
            &*op,
            &[int(v1), int(i1), int(v2), int(i2)],
            &[if expected { Fp::one() } else { Fp::zero() }],
            &[],
        );
    }
    // An index outside [0, 2^7) has no witness (the range check fails).
    assert!(!synthesize(&*op, &[int(1), int(128), int(1), int(0)], &[]).accepted);
}

// ── Scatter family (lower::scatter) ────────────────────────────────────

/// `np.insert` at a runtime position, in both forms and at negative,
/// interior and end positions: the honest witness gives NumPy's result, and
/// every hint (the position one-hot's lanes, each loaded inside its bit
/// gate) moved by any of the deltas is rejected.
#[test]
fn insert_hints_are_pinned() {
    use crate::lower::scatter::insert::Form;
    use crate::lower::scatter::test_support::{insert_graph, witness};
    let rows = vec![vec![5, 6], vec![7, 8], vec![9, 10]];
    let mut shifts: Vec<Fp> = vec![int(1), int(-1), int(2), int(-2)];
    shifts.extend(deltas(&[]));
    for form in [Form::SharedProduct, Form::Complement] {
        for pos in [-3, -1, 0, 2, 3] {
            let case = insert_graph(&rows, &[1, 2], pos, form);
            let w = witness(&case.inputs);
            let honest = synthesize_graph(&case.graph, &w, &[]);
            assert!(honest.accepted, "{form:?} pos={pos}: honest witness rejected");
            let expected: Vec<Fp> = case.expected.as_ref().unwrap().iter().map(|&v| int(v)).collect();
            assert_eq!(honest.outputs, expected, "{form:?} pos={pos}");
            assert_eq!(honest.hints, rows.len() + 1);
            for index in 0..honest.hints {
                for shift in &shifts {
                    let tampered = synthesize_graph(&case.graph, &w, &[(index, *shift)]);
                    assert!(!tampered.accepted, "{form:?} pos={pos}: hint #{index} moved by {shift:?} was accepted");
                }
            }
        }
        // A position NumPy rejects has no witness.
        let case = insert_graph(&rows, &[1, 2], 4, form);
        assert!(!synthesize_graph(&case.graph, &witness(&case.inputs), &[]).accepted);
    }
}

/// `IR::PackKey`, `IR::PackedKeyLess`, `IR::PackedKeyIndex`: packed keys
/// compare lexicographically, unpack to their index at the window
/// extremes, and the unpacking's hints (quotient and index) are pinned.
#[test]
fn packed_keys_compare_and_unpack_soundly() {
    let op: Box<Op> = Box::new(|s, c| {
        let k1 = s.pack_key(&c[0], &c[1], 7)?;
        let k2 = s.pack_key(&c[2], &c[3], 7)?;
        Ok(vec![s.packed_key_less(&k1, &k2, 7)?, s.packed_key_index(&k1, 7)?, s.packed_key_index(&k2, 7)?])
    });
    let (max, min) = (i64::MAX, i64::MIN);
    let cases: [(i64, i64, i64, i64); 6] =
        [(1, 5, 2, 0), (3, 4, 3, 5), (3, 5, 3, 4), (min, 127, max, 0), (max, 0, min, 127), (-1, 0, 0, 0)];
    for (v1, i1, v2, i2) in cases {
        let expected = (v1, i1) < (v2, i2);
        assert_sound(
            &format!("packed ({v1}, {i1}) < ({v2}, {i2})"),
            &*op,
            &[int(v1), int(i1), int(v2), int(i2)],
            &[if expected { Fp::one() } else { Fp::zero() }, int(i1), int(i2)],
            &[],
        );
    }
}

/// `assert x >= 0` before `sqrt(x)` (the frontend's precondition) is
/// proven by the root itself: no check of its own, while a negative `x`
/// still has no witness and the root stays pinned. `assert x >= 1` (not a
/// root guard) keeps its check.
#[test]
fn root_preconditions_are_proven_by_the_root() {
    use crate::ir_defs::IR;
    use graphs::*;
    let build = |bound: i64| {
        let mut g = Graph::default();
        let x = g.read("x");
        g.domain_check(x);
        let lo = g.int(bound);
        let holds = g.push(IR::GteI, &[x, lo]);
        g.push(IR::Assert, &[holds]);
        let root = g.push(IR::SqrtF, &[x]);
        g.push(IR::ExposePublicF, &[root]);
        g.finish()
    };
    let guarded = build(0);
    for x in [0i64, 1, 5, 1 << 40, i64::MAX] {
        let expected = kernel::fp_sqrt(int(x), PRECISION_BITS);
        let honest = run(&guarded, &[("x", x)], None);
        assert!(honest.accepted && honest.outputs == vec![expected], "sqrt({x})");
        assert!(!run(&guarded, &[("x", x)], Some(vec![expected + Fp::one()])).accepted, "sqrt({x}) + 1 accepted");
    }
    for x in [-1i64, -(1 << 40), i64::MIN] {
        assert!(!run(&guarded, &[("x", x)], None).accepted, "assert {x} >= 0 accepted");
    }
    let checked = build(1);
    assert!(run(&guarded, &[("x", 5)], None).advice + 20 < run(&checked, &[("x", 5)], None).advice, "the root's guard paid a check");
}

/// `a[i, :] = 0` by the one-hot write: NumPy's result for every index in
/// `[-n, n)` (negative ones from the end), no witness outside, and every
/// one-hot lane moved by any delta rejected (MockProver).
#[test]
fn index_write_hints_are_pinned() {
    use crate::lower::scatter::test_support::{witness, zero_row_graph};
    use crate::lower::scatter::OneHotWrite;
    let mut shifts: Vec<Fp> = vec![int(1), int(-1), int(2)];
    shifts.extend(deltas(&[]));
    for index in -4..=4 {
        let case = zero_row_graph(3, 2, index, &OneHotWrite);
        let w = witness(&case.inputs);
        let honest = synthesize_graph(&case.graph, &w, &[]);
        let Some(expected) = &case.expected else {
            assert!(!honest.accepted, "index {index}: accepted, NumPy raises");
            continue;
        };
        assert!(honest.accepted, "index {index}: honest witness rejected");
        assert_eq!(honest.outputs, expected.iter().map(|&v| int(v)).collect::<Vec<_>>(), "index {index}");
        for hint in 0..honest.hints {
            for shift in &shifts {
                assert!(!synthesize_graph(&case.graph, &w, &[(hint, *shift)]).accepted, "index {index}: hint #{hint} moved by {shift:?}");
            }
        }
    }
}

/// The index wrap's hinted sign (`normalize_dynamic_index` on an index of
/// unknown range): NumPy's position for every index in `[-n, n)`, no
/// witness outside, and the sign moved by any delta rejected (MockProver).
#[test]
fn index_wrap_sign_is_pinned_on_halo2_base() {
    use crate::lower::scatter::test_support::{witness, wrap_graph};
    let mut shifts: Vec<Fp> = vec![int(1), int(-1), int(2)];
    shifts.extend(deltas(&[]));
    for x in -6..=5 {
        let case = wrap_graph(x, 5);
        let w = witness(&case.inputs);
        let honest = synthesize_graph(&case.graph, &w, &[]);
        let Some(expected) = &case.expected else {
            assert!(!honest.accepted, "index {x}: accepted, NumPy raises");
            continue;
        };
        assert!(honest.accepted && honest.outputs == vec![int(expected[0])], "index {x}");
        assert_eq!(honest.hints, 1);
        for shift in &shifts {
            assert!(!synthesize_graph(&case.graph, &w, &[(0, *shift)]).accepted, "index {x}: sign moved by {shift:?}");
        }
    }
}

/// `x < y` after `y < x` reuses the first comparison (one equality instead
/// of a second range check): both bits are right below, at and above the
/// other operand and at the window's ends, a claimed flipped bit is
/// rejected, and the pair costs fewer cells than two independent orders.
#[test]
fn reverse_orders_use_the_trichotomy() {
    use crate::ir_defs::IR;
    use graphs::*;
    let build = |reverse: bool| {
        let mut g = Graph::default();
        let (x, y) = (g.read("x"), g.read("y"));
        g.domain_check(x);
        g.domain_check(y);
        let y_below = g.push(IR::LtI, &[y, x]);
        let zero = g.int(0);
        let x_below = if reverse { g.push(IR::LtI, &[x, y]) } else { g.push(IR::GtI, &[x, zero]) };
        g.push(IR::ExposePublicI, &[y_below]);
        g.push(IR::ExposePublicI, &[x_below]);
        g.finish()
    };
    let graph = build(true);
    let (max, min) = (i64::MAX, i64::MIN);
    for (x, y) in [(1, 2), (2, 1), (5, 5), (min, max), (max, min), (min, min), (-3, 4)] {
        let expected = vec![int((y < x) as i64), int((x < y) as i64)];
        let honest = run(&graph, &[("x", x), ("y", y)], None);
        assert!(honest.accepted && honest.outputs == expected, "({x}, {y})");
        let flipped = vec![expected[0], Fp::one() - expected[1]];
        assert!(!run(&graph, &[("x", x), ("y", y)], Some(flipped)).accepted, "({x}, {y}): flipped bit accepted");
    }
    let one_order_and_a_sign = run(&build(false), &[("x", 1), ("y", 2)], None).advice;
    assert!(run(&graph, &[("x", 1), ("y", 2)], None).advice < one_order_and_a_sign, "the reverse order paid a range check");
}
