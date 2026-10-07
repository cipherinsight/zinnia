//! Tests of `IR::IdentityCheck` and `IR::RescaleRawF` on the two-phase
//! circuit, and the harness the challenge constructions' tamper tests use
//! ([`mock_run`]): a synthesis with chosen hints perturbed and a chosen
//! offset added to every challenge cell, checked by halo2's `MockProver`.

use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage};
use halo2_base::halo2_proofs::dev::MockProver;

use super::{tamper, Fp, Halo2BaseSynthesizer};
use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::{Identity, IR};
use crate::prove::halo2_base::circuit::{layout, Halo2BaseCircuit};
use crate::prove::interpreter::interpret_ir;
use crate::prove::kernel;
use crate::prove::types::ProvingParams;
use crate::types::ValueId;

const K: u32 = 12;

/// The outcome of [`mock_run`].
pub(crate) struct Run {
    /// The public values the synthesis exposed.
    pub outputs: Vec<Fp>,
    pub accepted: bool,
    /// How many hints the synthesis loaded (the indices tampering reaches).
    pub hints: usize,
    /// Second-phase advice cells (with the challenge cells).
    pub second_phase_cells: usize,
}

/// Perturbations of an honest prover.
#[derive(Clone, Default)]
pub(crate) struct Tamper {
    /// `(index, delta)`: add `delta` to the `index`-th hint.
    pub hints: Vec<(usize, Fp)>,
    /// Added to every challenge before it is written to its cell.
    pub challenge_offset: Fp,
    /// Checked against these public values instead of the synthesized ones.
    pub claimed: Option<Vec<Fp>>,
}

/// Synthesize `ir` on `inputs` (`(name, value)`, all private) into the
/// two-phase circuit, laid out at `K`, perturbed by `tamper`, and run the
/// `MockProver`.
pub(crate) fn mock_run(ir: &IRGraph, inputs: &[(&str, InputNode)], tamper: &Tamper) -> Run {
    let mut circuit_inputs = CircuitInputs::new();
    for (name, value) in inputs {
        circuit_inputs.params.push(InputParam {
            name: name.to_string(),
            is_public: false,
            dtype: serde_json::Value::Null,
            value: value.clone(),
        });
    }
    let params = ProvingParams { k: K, precision_bits: 32, lookup_bits: 8 };
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Mock);
    builder.set_k(K as usize);
    builder.set_lookup_bits(8);
    builder.set_instance_columns(1);
    let (outputs, hints, second_phase) = {
        let witness = ResolvedWitness::new(circuit_inputs, params.precision_bits);
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, Some(witness), params);
        synth.tamper = tamper::HintTamper::targeting(&tamper.hints);
        synth.plan_ahead(ir);
        interpret_ir(ir, &mut synth).expect("synthesis");
        (synth.public_values().to_vec(), synth.tamper.hints_seen(), std::mem::take(&mut synth.second_phase))
    };
    let second_phase_cells = second_phase.shape().advice + second_phase.challenges();
    let mut circuit = Halo2BaseCircuit::new(builder, second_phase, Default::default(), Default::default());
    circuit.challenge_offset = tamper.challenge_offset;
    layout(&mut circuit);
    let instance = tamper.claimed.clone().unwrap_or_else(|| outputs.clone());
    let accepted = MockProver::run(K, &circuit, vec![instance]).expect("MockProver::run").verify().is_ok();
    Run { outputs, accepted, hints, second_phase_cells }
}

/// A field element as an input.
pub(crate) fn hex(v: Fp) -> InputNode {
    InputNode::Hex(crate::prove::halo2_base::fp_to_hex(&v))
}

/// Small IR graphs, statement ids are indices.
#[derive(Default)]
struct Graph {
    stmts: Vec<IRStatement>,
    vids: Vec<ValueId>,
}

impl Graph {
    fn push(&mut self, ir: IR, args: &[u32]) -> u32 {
        let id = self.stmts.len() as u32;
        let vid = ValueId::next();
        let arg_values = args.iter().map(|a| self.vids[*a as usize]).collect();
        self.stmts.push(IRStatement::new(id, vid, ir, args.to_vec(), arg_values, None));
        self.vids.push(vid);
        id
    }

    fn reads(&mut self, prefix: &str, n: usize) -> Vec<u32> {
        (0..n)
            .map(|i| self.push(IR::ReadInteger { path: InputPath::new(format!("{prefix}{i}"), vec![]), is_public: false }, &[]))
            .collect()
    }

    /// `identity` over `operands`, and the first operand public (a circuit
    /// needs an instance).
    fn finish(mut self, identity: Identity, operands: &[u32]) -> IRGraph {
        self.push(IR::IdentityCheck(identity), operands);
        self.push(IR::ExposePublicI, &operands[..1]);
        IRGraph::new(self.stmts)
    }
}

fn named(prefix: &str, values: &[i64]) -> Vec<(String, InputNode)> {
    values.iter().enumerate().map(|(i, v)| (format!("{prefix}{i}"), InputNode::Int(*v))).collect()
}

fn run_identity(identity: Identity, groups: &[(&str, &[i64])], challenge_offset: Fp) -> bool {
    let mut g = Graph::default();
    let mut operands = Vec::new();
    let mut inputs = Vec::new();
    for (prefix, values) in groups {
        operands.extend(g.reads(prefix, values.len()));
        inputs.extend(named(prefix, values));
    }
    let ir = g.finish(identity, &operands);
    let inputs: Vec<(&str, InputNode)> = inputs.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();
    let mock = {
        let values: Vec<Fp> = groups.iter().flat_map(|(_, v)| v.iter().map(|x| kernel::i64_to_fp(*x))).collect();
        kernel::identity_holds(identity, &values)
    };
    let accepted = mock_run(&ir, &inputs, &Tamper { challenge_offset, ..Tamper::default() }).accepted;
    if challenge_offset == Fp::zero() {
        assert_eq!(accepted, mock, "{identity:?} on {groups:?}: halo2-base and the mock disagree");
    }
    accepted
}

const MOVES: [i64; 3] = [1, -1, 1 << 40];

#[test]
fn multisets_of_values_are_compared() {
    let x: &[i64] = &[3, 1, 4, 1, 5, -9];
    let identity = Identity::MultisetEqual { len: 6, width: 1 };
    assert!(run_identity(identity, &[("x", x), ("y", &[-9, 1, 5, 4, 3, 1])], Fp::zero()));
    for y in [[-9, 1, 5, 4, 3, 2], [-9, 1, 5, 4, 3, 3], [1, 1, 1, 4, 5, -9], [9, 1, 5, 4, 3, 1]] {
        assert!(!run_identity(identity, &[("x", x), ("y", &y)], Fp::zero()), "y = {y:?} accepted");
    }
    for d in MOVES {
        assert!(!run_identity(identity, &[("x", x), ("y", &[-9, 1, 5, 4, 3, 1])], kernel::i64_to_fp(d)), "challenge moved by {d}");
    }
}

/// Tuples are compared as wholes: the same components in other tuples (each
/// component's multiset equal) are rejected, which compressing with powers
/// of the product's own challenge would not guarantee.
#[test]
fn multisets_of_tuples_are_compared_as_wholes() {
    let identity = Identity::MultisetEqual { len: 3, width: 3 };
    let left: &[i64] = &[1, 10, 100, 2, 20, 200, 3, 30, 300];
    assert!(run_identity(identity, &[("l", left), ("r", &[3, 30, 300, 1, 10, 100, 2, 20, 200])], Fp::zero()));
    for right in [[1, 20, 100, 2, 10, 200, 3, 30, 300], [1, 10, 300, 2, 20, 200, 3, 30, 100], [1, 10, 100, 2, 20, 200, 3, 30, 301]] {
        assert!(!run_identity(identity, &[("l", left), ("r", &right)], Fp::zero()), "{right:?} accepted");
    }
    for d in MOVES {
        assert!(!run_identity(identity, &[("l", left), ("r", &[3, 30, 300, 1, 10, 100, 2, 20, 200])], kernel::i64_to_fp(d)));
    }
}

#[test]
fn polynomial_products_are_checked_coefficient_by_coefficient() {
    let identity = Identity::PolynomialProduct { n: 3, m: 2 };
    // (1 + 2x - 3x^2)(4 - x) = 4 + 7x - 14x^2 + 3x^3
    let (a, b): (&[i64], &[i64]) = (&[1, 2, -3], &[4, -1]);
    let c = [4, 7, -14, 3];
    assert!(run_identity(identity, &[("a", a), ("b", b), ("c", &c)], Fp::zero()));
    for k in 0..4 {
        for d in [1, -1] {
            let mut wrong = c;
            wrong[k] += d;
            assert!(!run_identity(identity, &[("a", a), ("b", b), ("c", &wrong)], Fp::zero()), "c[{k}] + {d} accepted");
        }
    }
    for d in MOVES {
        assert!(!run_identity(identity, &[("a", a), ("b", b), ("c", &c)], kernel::i64_to_fp(d)));
    }
    // Lengths 1.
    assert!(run_identity(Identity::PolynomialProduct { n: 1, m: 1 }, &[("a", &[-7]), ("b", &[6]), ("c", &[-42])], Fp::zero()));
    assert!(!run_identity(Identity::PolynomialProduct { n: 1, m: 1 }, &[("a", &[-7]), ("b", &[6]), ("c", &[42])], Fp::zero()));
}

#[test]
fn matrix_products_are_checked_entry_by_entry() {
    let identity = Identity::MatrixProduct { rows: 2, inner: 3, cols: 2 };
    let (a, b): (&[i64], &[i64]) = (&[1, 2, 3, -4, 5, 6], &[7, 8, 9, -10, 11, 12]);
    let c = [58, 24, 83, -10];
    assert!(run_identity(identity, &[("a", a), ("b", b), ("c", &c)], Fp::zero()));
    for k in 0..4 {
        let mut wrong = c;
        wrong[k] += 1;
        assert!(!run_identity(identity, &[("a", a), ("b", b), ("c", &wrong)], Fp::zero()), "c[{k}] + 1 accepted");
    }
    // Two errors that cancel in a plain row sum (r = (1, 1)) are still caught.
    assert!(!run_identity(identity, &[("a", a), ("b", b), ("c", &[59, 23, 83, -10])], Fp::zero()));
    for d in MOVES {
        assert!(!run_identity(identity, &[("a", a), ("b", b), ("c", &c)], kernel::i64_to_fp(d)));
    }
    // A 1 x 1 x 1 product.
    assert!(run_identity(Identity::MatrixProduct { rows: 1, inner: 1, cols: 1 }, &[("a", &[3]), ("b", &[-5]), ("c", &[-15])], Fp::zero()));
}

/// `RescaleRawF` is `ProductSumF`'s rescale: the mock's value, and its hints
/// pinned (each moved by ±1 is rejected).
#[test]
fn a_raw_sum_is_rescaled_as_product_sum_does() {
    for raw in [0i128, 1, -1, 5 << 32, -(5 << 32) - 1, (1i128 << 94) + 12345, -(1i128 << 94)] {
        let mut g = Graph::default();
        let x = g.reads("raw", 1)[0];
        let q = g.push(IR::RescaleRawF, &[x]);
        g.push(IR::ExposePublicF, &[q]);
        let ir = IRGraph::new(g.stmts);
        let inputs = [("raw0", hex(kernel::i128_to_fp(raw)))];
        let honest = mock_run(&ir, &inputs, &Tamper::default());
        assert!(honest.accepted, "raw = {raw}");
        assert_eq!(honest.outputs, vec![kernel::fp_rescale_raw_i128(raw, 32).0], "raw = {raw}");
        for hint in 0..honest.hints {
            for d in [Fp::one(), -Fp::one()] {
                let tampered = mock_run(&ir, &inputs, &Tamper { hints: vec![(hint, d)], ..Tamper::default() });
                assert!(!tampered.accepted || tampered.outputs == honest.outputs, "raw = {raw}: hint {hint} moved by {d:?}");
            }
        }
    }
}
