//! Test fixtures shared by the sort lowering's tests and the backends'
//! soundness tests: a small sort over input integers, lowered with a chosen
//! plan; the same graph with every hint turned into a free private input
//! (a malicious prover chooses those freely); and the mock's verdict.

use std::collections::HashMap;
use std::sync::Arc;

use crate::builder::IRBuilder;
use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::kernel;
use crate::prove::types::ProvingParams;
use crate::types::Value;

use super::cost::{self, Demand, Network, Pin, Plan, Run};

/// The lowering a test forces on its sort.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Forced {
    /// Batcher's network (with the tie-break when indices are live).
    Batcher,
    /// Batcher's network on packed keys (indices only).
    PackedBatcher,
    /// The adjacent-swap network.
    AdjacentSwap,
    /// One pinned run over every rank (the permutation check).
    Permutation,
    /// The live constant ranks in maximal runs of consecutive ranks, each
    /// pinned by keys; runtime ranks pinned on their own.
    Runs,
    /// Every live constant-rank value pinned by its own count.
    Counting,
    /// The live smallest / largest values by asserted bounds and a
    /// membership product.
    Extremes,
    /// The cost rule of a target.
    CostRule(cost::SortTarget),
}

/// An output a test asks for: the value or argsort index of a constant
/// rank, or of a runtime rank supplied as the input `r{k}`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Output {
    Value(u32),
    Index(u32),
    ValueAt(i64),
    IndexAt(i64),
}

/// A lowered sort graph, its inputs (`a{j}` the elements, `r{k}` runtime
/// ranks) and the outputs NumPy's stable sort gives (`None` where a runtime
/// rank is out of range: no proof).
pub(crate) struct SortGraph {
    pub graph: IRGraph,
    pub inputs: Vec<(String, i64)>,
    pub expected: Option<Vec<i64>>,
}

fn plan_for(forced: Forced, demand: &Demand) -> Plan {
    let live: Vec<usize> = demand.values.union(&demand.indices).copied().collect();
    let needs_indices = !demand.indices.is_empty() || demand.runtime_indices > 0;
    match forced {
        Forced::Batcher => Plan::Network {
            network: if needs_indices { Network::MergeExchangeTieBreak } else { Network::MergeExchange },
        },
        Forced::PackedBatcher => Plan::Network { network: Network::PackedMergeExchange },
        Forced::AdjacentSwap => Plan::Network { network: Network::AdjacentSwap },
        Forced::Permutation => {
            Plan::Hinted { runs: vec![Run { first: 0, last: demand.n - 1, pin: Pin::Keys }] }
        }
        Forced::Runs => {
            let mut runs: Vec<Run> = Vec::new();
            for r in live {
                match runs.last_mut() {
                    Some(run) if run.last + 1 == r => run.last = r,
                    _ => runs.push(Run { first: r, last: r, pin: Pin::Keys }),
                }
            }
            Plan::Hinted { runs }
        }
        Forced::Counting => {
            assert!(demand.indices.is_empty(), "counting pins values only");
            Plan::Hinted { runs: live.iter().map(|&r| Run { first: r, last: r, pin: Pin::Counts }).collect() }
        }
        Forced::Extremes => {
            assert!(demand.indices.is_empty(), "extremes are values");
            Plan::Hinted { runs: live.iter().map(|&r| Run { first: r, last: r, pin: Pin::Extreme }).collect() }
        }
        Forced::CostRule(target) => cost::choose_plan(demand, &target.costs()).0,
    }
}

/// NumPy's stable sort of `values`: (sorted values, argsort).
pub(crate) fn stable_sort(values: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by_key(|&i| values[i]);
    (order.iter().map(|&i| values[i]).collect(), order.iter().map(|&i| i as i64).collect())
}

/// The sort of the inputs `a0..` (holding `values`), with `outputs`
/// exposed in order and lowered by `forced`.
pub(crate) fn sort_graph(values: &[i64], outputs: &[Output], forced: Forced) -> SortGraph {
    let mut b = IRBuilder::new();
    let elements: Arc<[Value]> = (0..values.len())
        .map(|j| b.ir_read_integer(InputPath::new(format!("a{j}"), vec![]), false))
        .collect();
    let mut inputs: Vec<(String, i64)> =
        values.iter().enumerate().map(|(j, v)| (format!("a{j}"), *v)).collect();
    let (sorted, argsort) = stable_sort(values);
    let mut expected = Some(Vec::new());
    for (k, output) in outputs.iter().enumerate() {
        let (cell, want) = match *output {
            Output::Value(r) => (b.ir_sorted(&elements, r, false), sorted.get(r as usize)),
            Output::Index(r) => (b.ir_sorted(&elements, r, true), argsort.get(r as usize)),
            Output::ValueAt(r) | Output::IndexAt(r) => {
                let index = matches!(output, Output::IndexAt(_));
                let rank = b.ir_read_integer(InputPath::new(format!("r{k}"), vec![]), false);
                inputs.push((format!("r{k}"), r));
                let lanes = if index { &argsort } else { &sorted };
                let want = usize::try_from(r).ok().and_then(|r| lanes.get(r));
                (b.ir_sorted_at_runtime_rank(&elements, &rank, index), want)
            }
        };
        b.ir_expose_public_i(&cell);
        match (want, expected.as_mut()) {
            (Some(w), Some(list)) => list.push(*w),
            _ => expected = None,
        }
    }
    let graph = super::lower_sorts(b.export_ir_graph(), |demand| plan_for(forced, demand));
    let graph = crate::optim::IRPass::exec(&crate::optim::DeadCodeElimination, graph);
    SortGraph { graph, inputs, expected }
}

/// The value of rank `rank` (the input `r`) among the active elements of
/// the inputs `a0..` (activity flags `f0..`), pinned by
/// `masked_order_statistic`, exposed. `expected` is NumPy's value of that
/// rank among the active elements (`None`: no such element, no proof).
pub(crate) fn masked_order_statistic_graph(values: &[i64], active: &[bool], rank: i64) -> SortGraph {
    let mut b = IRBuilder::new();
    let elements: Vec<Value> = (0..values.len())
        .map(|j| b.ir_read_integer(InputPath::new(format!("a{j}"), vec![]), false))
        .collect();
    let flags: Vec<Value> = (0..values.len())
        .map(|j| b.ir_read_integer(InputPath::new(format!("f{j}"), vec![]), false))
        .collect();
    let r = b.ir_read_integer(InputPath::new("r", vec![]), false);
    let v = super::emit::masked_order_statistic(&mut b, &elements, &flags, crate::types::NumberType::Integer, &r);
    b.ir_expose_public_i(&v);
    let mut inputs: Vec<(String, i64)> =
        values.iter().enumerate().map(|(j, v)| (format!("a{j}"), *v)).collect();
    inputs.extend(active.iter().enumerate().map(|(j, &f)| (format!("f{j}"), f as i64)));
    inputs.push(("r".to_string(), rank));
    let kept: Vec<i64> = values.iter().zip(active).filter(|(_, &f)| f).map(|(v, _)| *v).collect();
    let (sorted, _) = stable_sort(&kept);
    let expected = usize::try_from(rank).ok().and_then(|r| sorted.get(r)).map(|v| vec![*v]);
    SortGraph { graph: b.export_ir_graph(), inputs, expected }
}

pub(crate) fn witness(inputs: &[(String, i64)]) -> ResolvedWitness {
    let mut circuit_inputs = CircuitInputs::new();
    for (name, value) in inputs {
        circuit_inputs.params.push(InputParam {
            name: name.clone(),
            is_public: false,
            dtype: serde_json::Value::Null,
            value: InputNode::Int(*value),
        });
    }
    ResolvedWitness::new(circuit_inputs, ProvingParams::default().precision_bits)
}

/// The mock's verdict on `graph`: whether every assert held, and the
/// public outputs in statement order.
pub(crate) fn mock_run(graph: &IRGraph, inputs: &[(String, i64)]) -> (bool, Vec<i64>) {
    use crate::prove::interpreter::interpret_ir;
    use crate::prove::mock::synthesizer::MockSynthesizer;
    let mut synth = MockSynthesizer::new(witness(inputs), ProvingParams::default());
    interpret_ir(graph, &mut synth).expect("mock run");
    let mut outputs: Vec<(u32, i64)> = synth
        .public_outputs
        .iter()
        .map(|(label, value)| (label.rsplit('_').next().unwrap().parse().unwrap(), value.parse().unwrap()))
        .collect();
    outputs.sort();
    (synth.satisfied, outputs.into_iter().map(|(_, v)| v).collect())
}

/// A hint of a lowered graph, as a tamper target.
pub(crate) struct HintSite {
    /// The private input that replaces it in [`hints_as_inputs`].
    pub input: String,
    pub honest: i64,
}

/// `graph` with every `IR::Hint` replaced by a private input `hint{k}` (a
/// prover's free choice), and the honest value of each: the circuit is sound
/// when it accepts these inputs only at their honest values. Hint operands
/// must be input reads or constants (true of [`sort_graph`]).
pub(crate) fn hints_as_inputs(sort: &SortGraph) -> (IRGraph, Vec<HintSite>) {
    let named: HashMap<String, i64> = sort.inputs.iter().cloned().collect();
    let value_of = |stmt: u32| -> crate::prove::field::F {
        let ir = &sort.graph.stmts[stmt as usize].ir;
        match ir {
            IR::ReadInteger { path, .. } => kernel::i64_to_fp(named[&path.display()]),
            IR::ConstantInt { value } => kernel::i64_to_fp(*value),
            other => panic!("hint operand {other:?} is not an input or a constant"),
        }
    };
    let mut sites = Vec::new();
    let mut graph = sort.graph.clone();
    for stmt in graph.stmts.iter_mut() {
        let IR::Hint { kind, len } = stmt.ir else { continue };
        let operands: Vec<crate::prove::field::F> = stmt.arguments.iter().map(|&a| value_of(a)).collect();
        let len = len as usize;
        let honest = kernel::hint_lane(kind, &operands[..len], operands[len]);
        let input = format!("hint{}", sites.len());
        stmt.ir = IR::ReadInteger { path: InputPath::new(input.clone(), vec![]), is_public: false };
        stmt.arguments.clear();
        stmt.arg_values.clear();
        sites.push(HintSite { input, honest: kernel::fp_to_i64(honest) });
    }
    (graph, sites)
}

/// The inputs of the hinted graph with every hint honest except `site`,
/// which takes `value`.
pub(crate) fn tampered_inputs(sort: &SortGraph, sites: &[HintSite], site: usize, value: i64) -> Vec<(String, i64)> {
    let mut inputs = sort.inputs.clone();
    for (k, s) in sites.iter().enumerate() {
        inputs.push((s.input.clone(), if k == site { value } else { s.honest }));
    }
    inputs
}
