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
use std::sync::Arc;

use crate::lower::{ArrayOp, Construction, OpKind, Target};
use crate::types::Value;

use super::insert::Form;
use super::position::{position_one_hot, Positions};

pub(crate) use crate::lower::sort::test_support::{mock_run, witness};

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

/// `np.insert(a, pos, e, axis=0)` over the inputs `a{k}_{j}`, `e{j}`, `pos`:
/// one `OpKind::Insert` placeholder per output cell, lowered by the
/// construction of `form` alone, every output cell exposed in row-major
/// order.
pub(crate) fn insert_graph(rows: &[Vec<i64>], inserted: &[i64], pos: i64, form: Form) -> ScatterGraph {
    let mut b = IRBuilder::new();
    let mut inputs = Vec::new();
    let mut shared: Vec<Value> = Vec::new();
    for (k, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            shared.push(read(&mut b, &mut inputs, format!("a{k}_{j}"), *v));
        }
    }
    for (j, v) in inserted.iter().enumerate() {
        shared.push(read(&mut b, &mut inputs, format!("e{j}"), *v));
    }
    shared.push(read(&mut b, &mut inputs, "pos".into(), pos));
    let shared: Arc<[Value]> = shared.into();
    let (n, lanes) = (rows.len(), inserted.len());
    let kind = OpKind::Insert { rows: n as u32, lanes: lanes as u32 };
    for p in 0..(n + 1) * lanes {
        let op = ArrayOp { kind, shared: shared.len() as u32, output: 0, position: Some(p as u32) };
        let cell = b.ir_array_op(op, &shared, &[]);
        b.ir_expose_public_i(&cell);
    }
    let construction: &dyn Construction = match form {
        Form::SharedProduct => &super::InsertBySharedProducts,
        Form::Complement => &super::InsertByComplement,
    };
    let graph = b.export_ir_graph();
    let lowered = crate::lower::emit::lower(&graph, Target::Halo2Base, &[construction]).expect("an insertion to lower");
    ScatterGraph { graph: optimized(lowered), inputs, expected: numpy_insert(rows, inserted, pos) }
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

/// The three masked writes of 0490 over one integer cell `x` with
/// thresholds `t1`, `t2` (`a[x < t1] = 0; a[x >= t2] = 30; a[t1 <= x < t2]
/// += 5`), as `OpKind::MaskWrite` lanes lowered by `construction` alone;
/// the final cell exposed. `expected` replays the writes.
pub(crate) fn mask_write_chain_graph(x: i64, t1: i64, t2: i64, construction: &dyn Construction) -> ScatterGraph {
    use crate::lower::scatter::mask_write::{MASK, OLD, WRITTEN};
    let mut b = IRBuilder::new();
    let mut inputs = Vec::new();
    let cell = read(&mut b, &mut inputs, "x".into(), x);
    let low = read(&mut b, &mut inputs, "t1".into(), t1);
    let high = read(&mut b, &mut inputs, "t2".into(), t2);
    let below_t1 = b.ir_less_than_i(&cell, &low);
    let below_t2 = b.ir_less_than_i(&cell, &high);
    let at_least_t2 = b.ir_logical_not(&below_t2);
    let at_least_t1 = b.ir_logical_not(&below_t1);
    let between = b.ir_logical_and(&at_least_t1, &below_t2);
    let empty: Arc<[Value]> = Arc::from(Vec::new());
    let mut lane = |b: &mut IRBuilder, old: &Value, written: &Value, mask: &Value| {
        let op = ArrayOp { kind: OpKind::MaskWrite, shared: 0, output: 0, position: Some(0) };
        let mut own = vec![Value::None; 3];
        own[OLD] = old.clone();
        own[WRITTEN] = written.clone();
        own[MASK] = mask.clone();
        b.ir_array_op(op, &empty, &own)
    };
    let zero = b.ir_constant_int(0);
    let thirty = b.ir_constant_int(30);
    let five = b.ir_constant_int(5);
    let first = lane(&mut b, &cell, &zero, &below_t1);
    let second = lane(&mut b, &first, &thirty, &at_least_t2);
    let plus_five = b.ir_add_i(&second, &five);
    let third = lane(&mut b, &second, &plus_five, &between);
    b.ir_expose_public_i(&third);
    let mut expected = x;
    if x < t1 {
        expected = 0;
    }
    if x >= t2 {
        expected = 30;
    }
    if t1 <= x && x < t2 {
        expected += 5;
    }
    let graph = b.export_ir_graph();
    let lowered = crate::lower::emit::lower(&graph, Target::Halo2Base, &[construction]).expect("masked writes to lower");
    ScatterGraph { graph: optimized(lowered), inputs, expected: Some(vec![expected]) }
}

/// How many statements of `graph` are selects.
pub(crate) fn selects(graph: &IRGraph) -> usize {
    graph.stmts.iter().filter(|s| matches!(s.ir, IR::SelectI | IR::SelectF)).count()
}
