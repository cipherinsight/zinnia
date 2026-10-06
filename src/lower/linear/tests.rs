//! The linear family: the identity constructions give the direct sums'
//! values (the reference: `ProductSumF`'s exact rounding) on edge shapes,
//! every hint is pinned (mock: hints as free inputs; halo2-base: hints
//! perturbed, challenge cells moved, a wrong output claimed), and the
//! constructions are offered where they should be.

use crate::builder::IRBuilder;
use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
use crate::ir::IRGraph;
use crate::ir_defs::{Identity, IR};
use crate::lower::{Construction, Target};
use crate::prove::field::F as Fp;
use crate::prove::kernel;
use crate::prove::types::ProvingParams;
use crate::types::Value;

use super::{DirectProductSums, Freivalds, PolynomialIdentity};

/// The operator under test.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// `np.correlate(x, y, 'full')` lanes `ks` of lengths `(long, short)`.
    Correlation { long: usize, short: usize },
    MatMul { rows: usize, inner: usize, cols: usize },
}

impl Shape {
    fn inputs(self) -> usize {
        match self {
            Shape::Correlation { long, short } => long + short,
            Shape::MatMul { rows, inner, cols } => rows * inner + inner * cols,
        }
    }

    fn lanes(self) -> usize {
        match self {
            Shape::Correlation { long, short } => long + short - 1,
            Shape::MatMul { rows, cols, .. } => rows * cols,
        }
    }

    fn kind(self) -> crate::lower::OpKind {
        match self {
            Shape::Correlation { long, short } => crate::lower::OpKind::Correlation { long: long as u32, short: short as u32 },
            Shape::MatMul { rows, inner, cols } => {
                crate::lower::OpKind::MatMul { rows: rows as u32, inner: inner as u32, cols: cols as u32 }
            }
        }
    }
}

/// Float inputs `v0..` and the lanes `live` of the operator, exposed in
/// order, lowered by `construction`.
fn graph(shape: Shape, live: &[usize], construction: &dyn Construction) -> IRGraph {
    let mut b = IRBuilder::new();
    let inputs: Vec<Value> = (0..shape.inputs()).map(|i| b.ir_read_float(InputPath::new(format!("v{i}"), vec![]), false)).collect();
    let lanes = match shape {
        Shape::Correlation { long, .. } => super::correlation_lanes(&mut b, &inputs[..long], &inputs[long..], 0..shape.lanes()),
        Shape::MatMul { rows, inner, cols } => {
            super::matmul_lanes(&mut b, &inputs[..rows * inner], &inputs[rows * inner..], rows, inner, cols)
        }
    };
    for &p in live {
        b.ir_expose_public_f(&lanes[p]);
    }
    let lowered = crate::lower::emit::lower(&b.export_ir_graph(), Target::Halo2Base, &[construction]).expect("lanes");
    crate::optim::IRPass::exec(&crate::optim::DeadCodeElimination, lowered)
}

/// Inputs in Q32: a spread of magnitudes and signs, distinct per position.
fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) % 20_001) as f64 / 1000.0 - 10.0
        })
        .collect()
}

/// The reference: each live lane's `ProductSumF` rounding of its pairs.
fn expected(shape: Shape, live: &[usize], x: &[f64]) -> Vec<Fp> {
    let q: Vec<Fp> = x.iter().map(|v| kernel::quantize_to_fp(*v, 32)).collect();
    live.iter()
        .map(|&p| {
            let pairs: Vec<(Fp, Fp)> = super::lane_pairs(shape.kind(), p).into_iter().map(|(a, b)| (q[a], q[b])).collect();
            kernel::fp_product_sum_rescale(&pairs, 32).0
        })
        .collect()
}

fn mock(graph: &IRGraph, inputs: &[(String, InputNode)]) -> (bool, Vec<Fp>) {
    use crate::prove::interpreter::interpret_ir;
    use crate::prove::mock::synthesizer::MockSynthesizer;
    let mut circuit_inputs = CircuitInputs::new();
    for (name, value) in inputs {
        circuit_inputs.params.push(InputParam { name: name.clone(), is_public: false, dtype: serde_json::Value::Null, value: value.clone() });
    }
    let witness = ResolvedWitness::new(circuit_inputs, 32);
    let mut synth = MockSynthesizer::new(witness, ProvingParams::default());
    interpret_ir(graph, &mut synth).expect("mock run");
    let mut outputs: Vec<(u32, String)> = synth
        .public_outputs
        .iter()
        .map(|(label, value)| (label.rsplit('_').next().unwrap().parse().unwrap(), value.clone()))
        .collect();
    outputs.sort();
    let parsed = outputs.into_iter().map(|(_, v)| kernel::i64_to_fp(v.parse::<i64>().unwrap())).collect();
    (synth.satisfied, parsed)
}

fn float_inputs(x: &[f64]) -> Vec<(String, InputNode)> {
    x.iter().enumerate().map(|(i, v)| (format!("v{i}"), InputNode::Float(*v))).collect()
}

fn shapes() -> Vec<Shape> {
    let mut shapes: Vec<Shape> = [(1, 1), (1, 3), (3, 1), (5, 2), (2, 5), (4, 4), (7, 3), (16, 16)]
        .into_iter()
        .map(|(long, short)| Shape::Correlation { long, short })
        .collect();
    shapes.extend(
        [(1, 1, 1), (2, 3, 2), (3, 1, 4), (4, 4, 4), (1, 5, 1), (3, 2, 1), (8, 8, 8)]
            .into_iter()
            .map(|(rows, inner, cols)| Shape::MatMul { rows, inner, cols }),
    );
    shapes
}

fn identity_construction(shape: Shape) -> &'static dyn Construction {
    match shape {
        Shape::Correlation { .. } => &PolynomialIdentity,
        Shape::MatMul { .. } => &Freivalds,
    }
}

/// Every lane, a middle run of lanes (`valid`-like) and one lane: the
/// identity construction's outputs are the direct sums' on the mock.
#[test]
fn identity_constructions_give_the_direct_sums() {
    for shape in shapes() {
        let n = shape.lanes();
        let x = values(shape.inputs(), 7 + n as u64);
        for live in [(0..n).collect::<Vec<_>>(), (n / 3..n - n / 3).collect(), vec![n / 2]] {
            if live.is_empty() {
                continue;
            }
            let want = expected(shape, &live, &x);
            for construction in [&DirectProductSums as &dyn Construction, identity_construction(shape)] {
                let lowered = graph(shape, &live, construction);
                assert!(!lowered.stmts.iter().any(|s| matches!(s.ir, IR::ArrayOp(_))));
                let (ok, got) = mock(&lowered, &float_inputs(&x));
                assert!(ok, "{shape:?} {live:?} {}: honest run rejected", construction.name());
                assert_eq!(got, want, "{shape:?} {live:?} {}", construction.name());
            }
        }
    }
}

/// One identity check of the operator's shape; a hint per raw sum.
#[test]
fn one_identity_check_over_every_raw_sum() {
    for shape in shapes() {
        let lowered = graph(shape, &[0], identity_construction(shape));
        let checks: Vec<&IR> = lowered.stmts.iter().map(|s| &s.ir).filter(|ir| matches!(ir, IR::IdentityCheck(_))).collect();
        let identity = match shape {
            Shape::Correlation { long, short } => Identity::PolynomialProduct { n: long as u32, m: short as u32 },
            Shape::MatMul { rows, inner, cols } => {
                Identity::MatrixProduct { rows: rows as u32, inner: inner as u32, cols: cols as u32 }
            }
        };
        assert_eq!(checks, vec![&IR::IdentityCheck(identity)], "{shape:?}");
        let hints = lowered.stmts.iter().filter(|s| matches!(s.ir, IR::Hint { .. })).count();
        assert_eq!(hints, shape.lanes(), "{shape:?}");
    }
}

/// `graph` with every hint a free input `hint{k}`, and the honest values.
fn hints_as_inputs(graph: &IRGraph, x: &[f64]) -> (IRGraph, Vec<Fp>) {
    let named: std::collections::HashMap<String, Fp> =
        x.iter().enumerate().map(|(i, v)| (format!("v{i}"), kernel::quantize_to_fp(*v, 32))).collect();
    let mut values: Vec<Option<Fp>> = vec![None; graph.stmts.len()];
    let mut out = graph.clone();
    let mut honest = Vec::new();
    for (i, stmt) in graph.stmts.iter().enumerate() {
        values[i] = match &stmt.ir {
            IR::ReadFloat { path, .. } => Some(named[&path.display()]),
            IR::ConstantInt { value } => Some(kernel::i64_to_fp(*value)),
            IR::Hint { kind, len } => {
                let operands: Vec<Fp> = stmt.arguments.iter().map(|a| values[*a as usize].expect("known")).collect();
                let value = kernel::hint_lane(*kind, &operands[..*len as usize], operands[*len as usize]);
                let replaced = &mut out.stmts[i];
                replaced.ir = IR::ReadInteger { path: InputPath::new(format!("hint{}", honest.len()), vec![]), is_public: false };
                replaced.arguments.clear();
                replaced.arg_values.clear();
                honest.push(value);
                Some(value)
            }
            _ => None,
        };
    }
    (out, honest)
}

/// On the mock, every raw sum moved by 1, -1 or 2^32 (one ulp of the
/// output after the rescale) is rejected, as is a swap of two raw sums.
#[test]
fn every_raw_sum_is_pinned_on_the_mock() {
    for shape in [
        Shape::Correlation { long: 5, short: 3 },
        Shape::Correlation { long: 1, short: 1 },
        Shape::MatMul { rows: 2, inner: 3, cols: 2 },
        Shape::MatMul { rows: 3, inner: 3, cols: 3 },
    ] {
        let x = values(shape.inputs(), 3);
        let live: Vec<usize> = (0..shape.lanes()).collect();
        let (graph, honest) = hints_as_inputs(&graph(shape, &live, identity_construction(shape)), &x);
        let with = |hints: &[Fp]| {
            let mut inputs = float_inputs(&x);
            let hex = |v: &Fp| InputNode::Hex(crate::prove::halo2_base::fp_to_hex(v));
            inputs.extend(hints.iter().enumerate().map(|(k, v)| (format!("hint{k}"), hex(v))));
            mock(&graph, &inputs)
        };
        let (ok, got) = with(&honest);
        assert!(ok && got == expected(shape, &live, &x), "{shape:?}: honest hints rejected");
        for k in 0..honest.len() {
            for delta in [Fp::one(), -Fp::one(), Fp::from(1u64 << 32)] {
                let mut moved = honest.clone();
                moved[k] += delta;
                assert!(!with(&moved).0, "{shape:?}: raw sum {k} moved by {delta:?} accepted");
            }
        }
        if honest.len() > 1 && honest[0] != honest[1] {
            let mut swapped = honest.clone();
            swapped.swap(0, 1);
            assert!(!with(&swapped).0, "{shape:?}: two raw sums swapped");
        }
    }
}

/// halo2-base: the honest synthesis gives the reference outputs; every
/// hint moved by ±1 (the raw sums and the rescales' quotients and
/// remainders), a moved challenge cell and a wrong claimed output are
/// rejected by the MockProver.
#[cfg(feature = "halo2-base-backend")]
#[test]
fn every_hint_and_challenge_is_pinned_on_halo2_base() {
    use crate::prove::halo2_base::synthesizer::two_phase_tests::{mock_run, Tamper};
    for shape in [Shape::Correlation { long: 5, short: 3 }, Shape::MatMul { rows: 2, inner: 3, cols: 2 }, Shape::MatMul { rows: 1, inner: 1, cols: 1 }] {
        let x = values(shape.inputs(), 11);
        let live: Vec<usize> = (0..shape.lanes()).collect();
        let lowered = graph(shape, &live, identity_construction(shape));
        let named: Vec<(String, InputNode)> = float_inputs(&x);
        let inputs: Vec<(&str, InputNode)> = named.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();
        let honest = mock_run(&lowered, &inputs, &Tamper::default());
        assert!(honest.accepted, "{shape:?}: honest rejected");
        assert_eq!(honest.outputs, expected(shape, &live, &x), "{shape:?}");
        assert!(honest.second_phase_cells > 0);
        for hint in 0..honest.hints {
            for delta in [Fp::one(), -Fp::one()] {
                let tampered = mock_run(&lowered, &inputs, &Tamper { hints: vec![(hint, delta)], ..Tamper::default() });
                assert!(!tampered.accepted, "{shape:?}: hint {hint} moved by {delta:?} accepted");
            }
        }
        for offset in [1u64, 1 << 40] {
            let moved = mock_run(&lowered, &inputs, &Tamper { challenge_offset: Fp::from(offset), ..Tamper::default() });
            assert!(!moved.accepted, "{shape:?}: challenge moved by {offset}");
        }
        let mut wrong = honest.outputs.clone();
        wrong[0] += Fp::one();
        assert!(!mock_run(&lowered, &inputs, &Tamper { claimed: Some(wrong), ..Tamper::default() }).accepted);
    }
}

/// The identity constructions are offered on halo2-base only; the direct
/// sums everywhere.
#[test]
fn identity_constructions_are_offered_on_halo2_base_only() {
    let shape = Shape::MatMul { rows: 4, inner: 4, cols: 4 };
    let mut b = IRBuilder::new();
    let inputs: Vec<Value> = (0..shape.inputs()).map(|i| b.ir_read_float(InputPath::new(format!("v{i}"), vec![]), false)).collect();
    let lanes = super::matmul_lanes(&mut b, &inputs[..16], &inputs[16..], 4, 4, 4);
    for lane in &lanes {
        b.ir_expose_public_f(lane);
    }
    let program = b.export_ir_graph();
    let noir = crate::lower::emit::lower(&program, Target::Noir, &crate::lower::registry()).unwrap();
    assert!(!noir.stmts.iter().any(|s| matches!(s.ir, IR::IdentityCheck(_) | IR::Hint { .. } | IR::RescaleRawF)));
    assert_eq!(noir.stmts.iter().filter(|s| matches!(s.ir, IR::ProductSumF { .. })).count(), 16);
}

/// Measurement aid (not a check): cells of each construction against the
/// cost model, `cargo test -- --ignored --nocapture measure_linear_cells`.
#[cfg(feature = "halo2-base-backend")]
#[test]
#[ignore]
fn measure_linear_cells() {
    for shape in [
        Shape::Correlation { long: 4, short: 4 },
        Shape::Correlation { long: 16, short: 16 },
        Shape::Correlation { long: 64, short: 64 },
        Shape::Correlation { long: 99, short: 50 },
        Shape::MatMul { rows: 4, inner: 4, cols: 4 },
        Shape::MatMul { rows: 8, inner: 8, cols: 8 },
        Shape::MatMul { rows: 16, inner: 16, cols: 16 },
    ] {
        for live in [(0..shape.lanes()).collect::<Vec<_>>(), vec![shape.lanes() / 2]] {
            let cells = |c: &dyn Construction| {
                let stats = crate::prove::halo2_base::cell_stats(&graph(shape, &live, c), 8, 32).expect("cells");
                (stats.advice + stats.lookup, stats.second_advice)
            };
            let (direct, _) = cells(&DirectProductSums);
            let (identity, second) = cells(identity_construction(shape));
            let group = crate::lower::Group {
                id: 0,
                kind: shape.kind(),
                shared: Vec::new(),
                lanes: live
                    .iter()
                    .map(|&p| crate::lower::Lane { stmt: 0, output: 0, position: Some(p as u32), own: Vec::new() })
                    .collect(),
            };
            println!(
                "{shape:?} live {}: direct {direct} est {} | identity {identity} (2nd {second}) est {} | diff measured {} est {}",
                live.len(),
                super::direct_cost(&group, Target::Halo2Base),
                super::identity_cost(&group),
                identity as i64 - direct as i64,
                super::identity_cost(&group) as i64 - super::direct_cost(&group, Target::Halo2Base) as i64,
            );
        }
    }
}
