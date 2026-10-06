//! Test fixtures of the scatter family, shared with the backends' soundness
//! tests: small graphs over integer inputs, their NumPy results, the same
//! graphs with every hint turned into a free private input (a malicious
//! prover's choice), and the mock's verdict.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::circuit_input::InputPath;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::kernel;
use crate::types::{NumberType, Value};

use super::insert::{insert_row, Form};
use super::position::{position_one_hot, Positions};

pub(crate) use crate::optim::sort_lowering::test_support::mock_run;

/// A graph, its inputs, and the public outputs NumPy gives (`None`: NumPy
/// raises, so the circuit must have no witness).
pub(crate) struct ScatterGraph {
    pub graph: IRGraph,
    pub inputs: Vec<(String, i64)>,
    pub expected: Option<Vec<i64>>,
}

fn read(b: &mut IRBuilder, inputs: &mut Vec<(String, i64)>, name: String, value: i64) -> Value {
    inputs.push((name.clone(), value));
    b.ir_read_integer(InputPath::new(name, vec![]), false)
}

/// NumPy's position of `x` on an axis of length `n`, if it names one.
pub(crate) fn numpy_position(x: i64, n: usize, positions: Positions) -> Option<usize> {
    let p = if x < 0 { x + n as i64 } else { x };
    (0..positions.count(n) as i64).contains(&p).then_some(p as usize)
}

/// The one-hot and thermometer of the input `x` on an axis of length `n`,
/// exposed: `hot` then `below`.
pub(crate) fn position_graph(x: i64, n: usize, positions: Positions) -> ScatterGraph {
    let mut b = IRBuilder::new();
    let mut inputs = Vec::new();
    let x_cell = read(&mut b, &mut inputs, "x".into(), x);
    let point = position_one_hot(&mut b, &x_cell, n, positions);
    for cell in point.hot.iter().chain(&point.below) {
        b.ir_expose_public_i(cell);
    }
    let lanes = positions.count(n);
    let expected = numpy_position(x, n, positions).map(|p| {
        let hot = (0..lanes).map(|q| (q == p) as i64);
        let below = (0..=lanes).map(|k| (p < k) as i64);
        hot.chain(below).collect()
    });
    ScatterGraph { graph: b.export_ir_graph(), inputs, expected }
}

/// NumPy's `np.insert(a, pos, e, axis=0)` on `rows` (row-major), flattened.
pub(crate) fn numpy_insert(rows: &[Vec<i64>], inserted: &[i64], pos: i64) -> Option<Vec<i64>> {
    let p = numpy_position(pos, rows.len(), Positions::InsertionPoint)?;
    let mut out: Vec<Vec<i64>> = rows.to_vec();
    out.insert(p, inserted.to_vec());
    Some(out.concat())
}

/// `np.insert(a, pos, e, axis=0)` over the inputs `a{k}_{j}`, `e{j}`, `pos`,
/// computed by `form`, every output cell exposed in row-major order.
pub(crate) fn insert_graph(rows: &[Vec<i64>], inserted: &[i64], pos: i64, form: Form) -> ScatterGraph {
    let mut b = IRBuilder::new();
    let mut inputs = Vec::new();
    let row_cells: Vec<Vec<Value>> = rows
        .iter()
        .enumerate()
        .map(|(k, row)| {
            row.iter().enumerate().map(|(j, v)| read(&mut b, &mut inputs, format!("a{k}_{j}"), *v)).collect()
        })
        .collect();
    let e_cells: Vec<Value> =
        inserted.iter().enumerate().map(|(j, v)| read(&mut b, &mut inputs, format!("e{j}"), *v)).collect();
    let pos_cell = read(&mut b, &mut inputs, "pos".into(), pos);
    let out = insert_row(&mut b, &row_cells, &e_cells, &pos_cell, NumberType::Integer, form);
    for cell in out.iter().flatten() {
        b.ir_expose_public_i(cell);
    }
    ScatterGraph { graph: optimized(b.export_ir_graph()), inputs, expected: numpy_insert(rows, inserted, pos) }
}

/// The IR after dead-code elimination (what the backends receive).
fn optimized(graph: IRGraph) -> IRGraph {
    crate::optim::IRPass::exec(&crate::optim::DeadCodeElimination, graph)
}

/// A hint of a graph, as a tamper target.
pub(crate) struct HintSite {
    /// The private input that replaces it in [`hints_as_inputs`].
    pub input: String,
    pub honest: i64,
}

/// `case.graph` with every `IR::Hint` replaced by a private input `hint{k}`,
/// and the honest value of each. Hint operands must be input reads or
/// constants.
pub(crate) fn hints_as_inputs(case: &ScatterGraph) -> (IRGraph, Vec<HintSite>) {
    let named: HashMap<String, i64> = case.inputs.iter().cloned().collect();
    let value_of = |stmt: u32| -> crate::prove::field::F {
        match &case.graph.stmts[stmt as usize].ir {
            IR::ReadInteger { path, .. } => kernel::i64_to_fp(named[&path.display()]),
            IR::ConstantInt { value } => kernel::i64_to_fp(*value),
            other => panic!("hint operand {other:?} is not an input or a constant"),
        }
    };
    let mut sites = Vec::new();
    let mut graph = case.graph.clone();
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

/// The inputs of the hinted graph with the hints set to `hints`.
pub(crate) fn with_hints(case: &ScatterGraph, sites: &[HintSite], hints: &[i64]) -> Vec<(String, i64)> {
    let mut inputs = case.inputs.clone();
    inputs.extend(sites.iter().zip(hints).map(|(site, v)| (site.input.clone(), *v)));
    inputs
}
