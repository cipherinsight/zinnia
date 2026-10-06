//! The grand-product sort (`grand_product`): NumPy's stable sort on edge
//! shapes (mock), every hint pinned (mock: hints replaced by free inputs;
//! halo2-base: hints perturbed in the real synthesis, challenge cells
//! moved), and the plan offered on halo2-base only.

use super::cost::{grand_product_plan, unit_costs, Demand};
use super::test_support::{sort_graph, stable_sort, Forced, Output, SortGraph, PAYLOAD_BASE};
use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
use crate::ir::IRGraph;
use crate::ir_defs::{HintKind, Identity, IR};
use crate::lower::Target;
use crate::prove::field::F as Fp;
use crate::prove::kernel;
use crate::prove::types::ProvingParams;

fn every_output(n: u32) -> Vec<Output> {
    (0..n).flat_map(|r| [Output::Value(r), Output::Index(r)]).collect()
}

/// Payload entries, values and indices of every rank (payload first, see
/// `routed_tests`).
fn values_indices_payloads(n: u32) -> Vec<Output> {
    (0..n).map(Output::Payload).chain(every_output(n)).collect()
}

/// `(name, field value)` inputs on the mock: whether every assertion and
/// identity held, and the public outputs in order.
fn mock(graph: &IRGraph, inputs: &[(String, Fp)]) -> (bool, Vec<i64>) {
    use crate::prove::interpreter::interpret_ir;
    use crate::prove::mock::synthesizer::MockSynthesizer;
    let mut circuit_inputs = CircuitInputs::new();
    for (name, value) in inputs {
        circuit_inputs.params.push(InputParam {
            name: name.clone(),
            is_public: false,
            dtype: serde_json::Value::Null,
            value: InputNode::Hex(crate::prove::halo2_base::fp_to_hex(value)),
        });
    }
    let witness = ResolvedWitness::new(circuit_inputs, ProvingParams::default().precision_bits);
    let mut synth = MockSynthesizer::new(witness, ProvingParams::default());
    interpret_ir(graph, &mut synth).expect("mock run");
    let mut outputs: Vec<(u32, i64)> = synth
        .public_outputs
        .iter()
        .map(|(label, value)| (label.rsplit('_').next().unwrap().parse().unwrap(), value.parse().unwrap()))
        .collect();
    outputs.sort();
    (synth.satisfied, outputs.into_iter().map(|(_, v)| v).collect())
}

fn field_inputs(sort: &SortGraph) -> Vec<(String, Fp)> {
    sort.inputs.iter().map(|(n, v)| (n.clone(), kernel::i64_to_fp(*v))).collect()
}

fn assert_grand_product_is_numpy(values: &[i64], outputs: &[Output]) {
    let sort = sort_graph(values, outputs, Forced::GrandProduct);
    assert!(!sort.graph.stmts.iter().any(|s| matches!(s.ir, IR::ArrayOp(_))), "a placeholder survived");
    let (ok, got) = mock(&sort.graph, &field_inputs(&sort));
    match &sort.expected {
        Some(expected) => {
            assert!(ok, "{values:?} {outputs:?}: honest run rejected");
            assert_eq!(&got, expected, "{values:?} {outputs:?}");
        }
        None => assert!(!ok, "{values:?} {outputs:?}: an out-of-range rank was accepted"),
    }
}

fn edge_inputs() -> Vec<Vec<i64>> {
    let mut inputs = vec![
        vec![5, 5],
        vec![2, 1],
        vec![1, 2],
        vec![3, 1, 2],
        vec![7; 7],
        (0..9).collect(),
        (0..9).rev().collect(),
        vec![5, -3, 5, 0, -3, 7, 5, 0, -9],
        vec![i64::MAX, i64::MIN, 0, i64::MAX, -1, i64::MIN],
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for n in [5usize, 6, 11, 16, 33, 100] {
        inputs.push(
            (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) % 7) as i64 - 3
                })
                .collect(),
        );
    }
    inputs
}

#[test]
fn grand_product_sort_equals_numpy_on_edge_shapes() {
    for values in edge_inputs() {
        let n = values.len() as u32;
        assert_grand_product_is_numpy(&values, &every_output(n));
        assert_grand_product_is_numpy(&values, &(0..n).map(Output::Index).collect::<Vec<_>>());
        assert_grand_product_is_numpy(&values, &(0..n).map(Output::Value).collect::<Vec<_>>());
        assert_grand_product_is_numpy(&values, &(0..n).map(Output::Payload).collect::<Vec<_>>());
        assert_grand_product_is_numpy(&values, &values_indices_payloads(n));
        assert_grand_product_is_numpy(&values, &[Output::Payload(0), Output::Index(n - 1), Output::Value(n / 2)]);
        for r in [0, i64::from(n) - 1, -1, i64::from(n)] {
            assert_grand_product_is_numpy(&values, &[Output::ValueAt(r), Output::IndexAt(r)]);
        }
    }
}

/// One multiset check over tuples of the key and each carried array, no
/// routing bit, one hint per rank and carried array.
#[test]
fn one_multiset_check_carries_every_array() {
    let values: Vec<i64> = (0..9).rev().collect();
    for (outputs, carried) in [
        ((0..9).map(Output::Index).collect::<Vec<_>>(), 0u32),
        (every_output(9), 1),
        (values_indices_payloads(9), 2),
    ] {
        let sort = sort_graph(&values, &outputs, Forced::GrandProduct);
        let checks: Vec<&IR> = sort.graph.stmts.iter().map(|s| &s.ir).filter(|ir| matches!(ir, IR::IdentityCheck(_))).collect();
        assert_eq!(checks, vec![&IR::IdentityCheck(Identity::MultisetEqual { len: 9, width: 1 + carried })]);
        let hints = sort.graph.stmts.iter().filter(|s| matches!(s.ir, IR::Hint { .. })).count();
        assert_eq!(hints, 9 * (1 + carried as usize));
        assert!(!sort.graph.stmts.iter().any(|s| matches!(s.ir, IR::Hint { kind: HintKind::RoutingBits, .. })));
    }
}

/// `sort.graph` with every hint replaced by a free input `hint{k}`, and the
/// honest values (field elements: packed keys exceed int64).
fn hints_as_inputs(sort: &SortGraph) -> (IRGraph, Vec<(String, Fp)>) {
    let named: std::collections::HashMap<String, i64> = sort.inputs.iter().cloned().collect();
    let mut values: Vec<Option<Fp>> = vec![None; sort.graph.stmts.len()];
    let mut graph = sort.graph.clone();
    let mut sites = Vec::new();
    for (i, stmt) in sort.graph.stmts.iter().enumerate() {
        let arg = |a: &u32| values[*a as usize];
        values[i] = match &stmt.ir {
            IR::ReadInteger { path, .. } => Some(kernel::i64_to_fp(named[&path.display()])),
            IR::ConstantInt { value } => Some(kernel::i64_to_fp(*value)),
            IR::PackKey { index_bits } => Some(arg(&stmt.arguments[0]).unwrap() * kernel::i128_to_fp(1 << index_bits) + arg(&stmt.arguments[1]).unwrap()),
            IR::Hint { kind, len } => {
                let operands: Vec<Fp> = stmt.arguments.iter().map(|a| arg(a).expect("a known operand")).collect();
                let honest = kernel::hint_lane(*kind, &operands[..*len as usize], operands[*len as usize]);
                let input = format!("hint{}", sites.len());
                let replaced = &mut graph.stmts[i];
                replaced.ir = IR::ReadInteger { path: InputPath::new(input.clone(), vec![]), is_public: false };
                replaced.arguments.clear();
                replaced.arg_values.clear();
                sites.push((input, honest));
                Some(honest)
            }
            _ => None,
        };
    }
    (graph, sites)
}

/// Every hint, moved by each delta with the others honest, is rejected on
/// the mock: the sorted keys (a value off the multiset, or out of order) and
/// the carried entries (a tuple the input does not have). So is a swap of
/// two ranks' keys, of two ranks' payload entries (each component's
/// multiset is unchanged; the tuples are not), and a duplicated key.
#[test]
fn every_hint_is_pinned_on_the_mock() {
    for values in [vec![3, 1, 3, 0, 1], vec![9; 4], vec![2, -7, 5]] {
        let n = values.len();
        let sort = sort_graph(&values, &values_indices_payloads(n as u32), Forced::GrandProduct);
        let expected = sort.expected.clone().unwrap();
        let (graph, sites) = hints_as_inputs(&sort);
        let honest: Vec<(String, Fp)> = field_inputs(&sort).into_iter().chain(sites.iter().cloned()).collect();
        let (ok, got) = mock(&graph, &honest);
        assert!(ok && got == expected, "{values:?}: honest hints rejected");
        let base = sort.inputs.len();
        let with = |changes: &[(usize, Fp)]| {
            let mut inputs = honest.clone();
            for &(site, value) in changes {
                inputs[base + site].1 = value;
            }
            mock(&graph, &inputs).0
        };
        for site in 0..sites.len() {
            for delta in [1i64, -1, 1 << 20, i64::MIN] {
                let moved = sites[site].1 + kernel::i64_to_fp(delta);
                assert!(!with(&[(site, moved)]), "{values:?}: hint {site} moved by {delta} accepted");
            }
        }
        // Sites: n keys, then n per carried array (values, then the payload).
        let swap = |a: usize, b: usize| with(&[(a, sites[b].1), (b, sites[a].1)]);
        assert!(!swap(0, 1), "{values:?}: two sorted keys swapped");
        assert!(!swap(2 * n, 2 * n + 1), "{values:?}: two payload entries swapped");
        assert!(!with(&[(1, sites[0].1)]), "{values:?}: a duplicated key");
    }
}

/// Equal keys carry their payloads in the original order.
#[test]
fn equal_keys_carry_payloads_in_original_order() {
    let values = [9i64; 6];
    let sort = sort_graph(&values, &(0..6).map(Output::Payload).collect::<Vec<_>>(), Forced::GrandProduct);
    let (ok, got) = mock(&sort.graph, &field_inputs(&sort));
    assert!(ok);
    assert_eq!(got, (0..6).map(|j| PAYLOAD_BASE + j).collect::<Vec<_>>());
    assert_eq!(stable_sort(&values).1, (0..6).collect::<Vec<_>>());
}

#[test]
fn grand_product_plan_is_offered_on_halo2_base_only_and_from_two_elements() {
    let full = |n: usize| Demand { n, indices: (0..n).collect(), ..Demand::default() };
    for n in [1usize, 2, 3, 100, 1024] {
        assert!(grand_product_plan(&full(n), &unit_costs(Target::Noir)).is_none(), "Noir, n = {n}");
        assert_eq!(grand_product_plan(&full(n), &unit_costs(Target::Halo2Base)).is_some(), n >= 2, "n = {n}");
    }
}

/// halo2-base: the honest synthesis gives NumPy's outputs; every hint moved
/// by ±1, and every challenge cell moved, is rejected by the MockProver.
#[cfg(feature = "halo2-base-backend")]
#[test]
fn every_hint_and_challenge_is_pinned_on_halo2_base() {
    use crate::prove::halo2_base::synthesizer::two_phase_tests::{mock_run, Tamper};
    for values in [vec![3i64, 1, 3, 0, 1], vec![9; 4], vec![i64::MAX, i64::MIN, 0]] {
        let n = values.len();
        let sort = sort_graph(&values, &values_indices_payloads(n as u32), Forced::GrandProduct);
        let inputs: Vec<(&str, InputNode)> = sort.inputs.iter().map(|(name, v)| (name.as_str(), InputNode::Int(*v))).collect();
        let honest = mock_run(&sort.graph, &inputs, &Tamper::default());
        assert!(honest.accepted, "{values:?}: honest rejected");
        let expected: Vec<Fp> = sort.expected.clone().unwrap().iter().map(|v| kernel::i64_to_fp(*v)).collect();
        assert_eq!(honest.outputs, expected, "{values:?}");
        assert!(honest.second_phase_cells > 0);
        for hint in 0..honest.hints {
            for delta in [Fp::one(), -Fp::one()] {
                let tampered = mock_run(&sort.graph, &inputs, &Tamper { hints: vec![(hint, delta)], ..Tamper::default() });
                assert!(!tampered.accepted, "{values:?}: hint {hint} moved by {delta:?} accepted");
            }
        }
        for offset in [1u64, 1 << 40] {
            let moved = mock_run(&sort.graph, &inputs, &Tamper { challenge_offset: Fp::from(offset), ..Tamper::default() });
            assert!(!moved.accepted, "{values:?}: challenge moved by {offset}");
        }
        // A wrong public output is rejected.
        let mut wrong = honest.outputs.clone();
        wrong[0] += Fp::one();
        assert!(!mock_run(&sort.graph, &inputs, &Tamper { claimed: Some(wrong), ..Tamper::default() }).accepted);
    }
}

/// Measurement aid (not a check): cells of the grand-product and routed
/// plans against the cost model's estimates, `cargo test -- --ignored
/// --nocapture measure_grand_product`.
#[cfg(feature = "halo2-base-backend")]
#[test]
#[ignore]
fn measure_grand_product_cells() {
    use super::cost::{choose_plan, routed_plan};
    use std::collections::BTreeSet;
    let costs = unit_costs(Target::Halo2Base);
    for n in [2usize, 3, 4, 5, 8, 16, 32, 64, 100] {
        let values: Vec<i64> = (0..n as i64).map(|j| (j * 37 + 11) % 13 - 6).collect();
        let shapes: Vec<(&str, Vec<Output>)> = vec![
            ("argsort", (0..n as u32).map(Output::Index).collect()),
            ("sort", (0..n as u32).map(Output::Value).collect()),
            ("both", every_output(n as u32)),
            ("payload", (0..n as u32).map(Output::Payload).collect()),
        ];
        for (shape, outputs) in shapes {
            let cells = |forced| {
                let sort = sort_graph(&values, &outputs, forced);
                let stats = crate::prove::halo2_base::cell_stats(&sort.graph, 8, 32).expect("cells");
                (stats.advice + stats.lookup, stats.second_advice)
            };
            let (gp, second) = cells(Forced::GrandProduct);
            let (routed, _) = cells(Forced::Routed);
            let demand = Demand {
                n,
                values: if shape == "sort" || shape == "both" { (0..n).collect() } else { BTreeSet::new() },
                indices: if shape == "argsort" || shape == "both" { (0..n).collect() } else { BTreeSet::new() },
                payloads: if shape == "payload" { vec![(0..n).collect()] } else { Vec::new() },
                ..Demand::default()
            };
            let gp_est = grand_product_plan(&demand, &costs).map(|(_, c)| c);
            let routed_est = routed_plan(&demand, &costs).map(|(_, c)| c);
            println!(
                "n={n} {shape}: gp {gp} (2nd {second}) est {gp_est:?} | routed {routed} est {routed_est:?} | rule {:?}",
                choose_plan(&demand, &costs).0
            );
        }
    }
}
