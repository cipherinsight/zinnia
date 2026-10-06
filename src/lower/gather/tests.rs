//! Tests of the gather family's constructions: each one built on input
//! reads, run on the mock backend, and compared with the NumPy result it
//! implements, over every input of small sizes (including size 1).
//!
//! The constructions here use no prover hint (every value is a polynomial of
//! the inputs and of comparison bits), so their soundness rests on the
//! outputs being those polynomials; the exhaustive comparisons below check
//! that the polynomials are the right ones.

use crate::builder::IRBuilder;
use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
use crate::ir::IRGraph;
use crate::prove::types::ProvingParams;
use crate::types::{NumberType, Value};

use super::encoding::{encoding_of, Bit};
use super::scan::{first_set, last_set};
use super::vanishing::forced_zero_by_unset;
use super::window::{window_shift, AxisWindow};

fn input(b: &mut IRBuilder, name: String) -> Value {
    b.ir_read_integer(InputPath::new(name, vec![]), false)
}

/// The mock's verdict: whether every assert held, and the public outputs in
/// statement order.
fn mock_run(graph: &IRGraph, inputs: &[(String, i64)]) -> (bool, Vec<i64>) {
    use crate::prove::interpreter::interpret_ir;
    use crate::prove::mock::synthesizer::MockSynthesizer;
    let mut circuit_inputs = CircuitInputs::new();
    for (name, value) in inputs {
        circuit_inputs.params.push(InputParam {
            name: name.clone(),
            is_public: false,
            dtype: serde_json::Value::Null,
            value: InputNode::Int(*value),
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

/// Every 0/1 vector of length `n`.
fn all_flag_vectors(n: usize) -> impl Iterator<Item = Vec<i64>> {
    (0..1u32 << n).map(move |m| (0..n).map(|i| ((m >> i) & 1) as i64).collect())
}

/// The scan's value, one-hot and thermometer, exposed in that order, for
/// flags `f{i}` (inputs; the scan treats them as 0/1 values).
fn scan_graph(n: usize, last: bool) -> IRGraph {
    let mut b = IRBuilder::new();
    let flags: Vec<Value> = (0..n).map(|i| input(&mut b, format!("f{i}"))).collect();
    let v = if last { last_set(&mut b, &flags) } else { first_set(&mut b, &flags) }.unwrap();
    let enc = encoding_of(&b, &v).expect("the scan records an encoding");
    b.ir_expose_public_i(&v);
    for bit in &enc.onehot {
        b.ir_expose_public_i(bit);
    }
    let below: Vec<Bit> = enc.below.clone().expect("the scan records a thermometer");
    for bit in &below[1..] {
        let value = bit.materialize(&mut b);
        b.ir_expose_public_i(&value);
    }
    b.export_ir_graph()
}

#[test]
fn scans_match_first_and_last_set_position_on_every_nonempty_flag_vector() {
    for n in 1..=6 {
        for last in [false, true] {
            let graph = scan_graph(n, last);
            for flags in all_flag_vectors(n).filter(|f| f.contains(&1)) {
                let inputs: Vec<(String, i64)> =
                    flags.iter().enumerate().map(|(i, f)| (format!("f{i}"), *f)).collect();
                let set: Vec<usize> = (0..n).filter(|&i| flags[i] == 1).collect();
                let v = if last { *set.last().unwrap() } else { set[0] };
                let mut expected = vec![v as i64];
                expected.extend((0..n).map(|k| (k == v) as i64));
                expected.extend((1..n).map(|k| (v < k) as i64));
                let (ok, outputs) = mock_run(&graph, &inputs);
                assert!(ok, "n={n} last={last} flags={flags:?}");
                assert_eq!(outputs, expected, "n={n} last={last} flags={flags:?}");
            }
        }
    }
}

#[test]
fn scan_results_stay_in_range_on_the_empty_flag_vector() {
    // The callers assert non-emptiness; without it the value still lies in
    // [0, n), as the recorded range fact states.
    for n in 1..=5 {
        for last in [false, true] {
            let inputs: Vec<(String, i64)> = (0..n).map(|i| (format!("f{i}"), 0)).collect();
            let (ok, outputs) = mock_run(&scan_graph(n, last), &inputs);
            assert!(ok);
            assert!((0..n as i64).contains(&outputs[0]));
        }
    }
}

/// NumPy's `x[s0:e0, s1:e1]` of a row-major `rows x cols` array, written at
/// the top-left of a zero buffer of the same shape (the padded layout).
fn padded_crop(x: &[i64], rows: usize, cols: usize, bounds: [(usize, usize); 2]) -> Vec<i64> {
    let mut out = vec![0; rows * cols];
    let [(s0, e0), (s1, e1)] = bounds;
    for u in 0..e0.saturating_sub(s0) {
        for v in 0..e1.saturating_sub(s1) {
            out[u * cols + v] = x[(s0 + u) * cols + s1 + v];
        }
    }
    out
}

#[test]
fn window_shift_matches_numpy_crop_for_every_bound_pair() {
    // start in [0, n] (the precondition), stop in [0, n] (Python-normalized).
    for (rows, cols) in [(1, 1), (1, 3), (3, 1), (2, 3), (3, 4)] {
        let mut b = IRBuilder::new();
        let cells: Vec<Value> = (0..rows * cols).map(|k| input(&mut b, format!("x{k}"))).collect();
        let bounds: Vec<Value> = (0..4).map(|k| input(&mut b, format!("b{k}"))).collect();
        let windows = [
            AxisWindow { axis: 0, start: bounds[0].clone(), stop: bounds[1].clone() },
            AxisWindow { axis: 1, start: bounds[2].clone(), stop: bounds[3].clone() },
        ];
        let out = window_shift(&mut b, cells, &[rows, cols], &windows, NumberType::Integer);
        for cell in &out {
            b.ir_expose_public_i(cell);
        }
        let graph = b.export_ir_graph();
        let x: Vec<i64> = (0..rows * cols).map(|k| 10 + k as i64 * 7 - 3 * (k as i64 % 2)).collect();
        for s0 in 0..=rows {
            for e0 in 0..=rows {
                for s1 in 0..=cols {
                    for e1 in 0..=cols {
                        let mut inputs: Vec<(String, i64)> =
                            x.iter().enumerate().map(|(k, v)| (format!("x{k}"), *v)).collect();
                        for (k, v) in [s0, e0, s1, e1].iter().enumerate() {
                            inputs.push((format!("b{k}"), *v as i64));
                        }
                        let (ok, outputs) = mock_run(&graph, &inputs);
                        assert!(ok);
                        assert_eq!(
                            outputs,
                            padded_crop(&x, rows, cols, [(s0, e0), (s1, e1)]),
                            "{rows}x{cols} [{s0}:{e0}, {s1}:{e1}]"
                        );
                    }
                }
            }
        }
    }
}

/// The bounding-box crop of the nonzero entries of a `rows x cols` input:
/// flags `count_nonzero(row) != 0` and the same per column, the window from
/// their first set position to their last set position plus one.
fn bbox_crop_graph(rows: usize, cols: usize) -> (IRGraph, bool) {
    let mut b = IRBuilder::new();
    let cells: Vec<Value> = (0..rows * cols).map(|k| input(&mut b, format!("x{k}"))).collect();
    let zero = b.ir_constant_int(0);
    let nonzero: Vec<Value> = cells.iter().map(|c| b.ir_not_equal_i(c, &zero)).collect();
    let count = |b: &mut IRBuilder, members: Vec<&Value>| -> Value {
        let mut members = members.into_iter();
        let first = members.next().unwrap().clone();
        let total = members.fold(first, |acc, m| b.ir_add_i(&acc, m));
        b.ir_not_equal_i(&total, &zero)
    };
    let row_flags: Vec<Value> =
        (0..rows).map(|i| count(&mut b, (0..cols).map(|j| &nonzero[i * cols + j]).collect())).collect();
    let col_flags: Vec<Value> =
        (0..cols).map(|j| count(&mut b, (0..rows).map(|i| &nonzero[i * cols + j]).collect())).collect();
    let one = b.ir_constant_int(1);
    let mut windows = Vec::new();
    for (axis, flags) in [(0, &row_flags), (1, &col_flags)] {
        let start = first_set(&mut b, flags).unwrap();
        let last = last_set(&mut b, flags).unwrap();
        let stop = b.ir_add_i(&last, &one);
        windows.push(AxisWindow { axis, start, stop });
    }
    let padding_free = (0..rows)
        .all(|i| forced_zero_by_unset(&b.stmts, row_flags[i].stmt_id().unwrap()).contains(&cells[i * cols].stmt_id().unwrap()));
    let out = window_shift(&mut b, cells, &[rows, cols], &windows, NumberType::Integer);
    for cell in &out {
        b.ir_expose_public_i(cell);
    }
    (b.export_ir_graph(), padding_free)
}

#[test]
fn bounding_box_crop_without_padding_mask_matches_numpy() {
    for (rows, cols) in [(1, 1), (1, 3), (2, 2), (3, 3)] {
        let (graph, padding_free) = bbox_crop_graph(rows, cols);
        assert!(padding_free, "row flags force their row to 0");
        // The mask was dropped: no comparison of a position with `stop`.
        assert!(!graph.stmts.iter().any(|s| matches!(s.ir, crate::ir_defs::IR::LtIBounded)));
        let n = rows * cols;
        for pattern in all_flag_vectors(n).filter(|p| p.contains(&1)) {
            let x: Vec<i64> = pattern.iter().enumerate().map(|(k, p)| p * (k as i64 + 2)).collect();
            let nz_rows: Vec<usize> = (0..rows).filter(|i| (0..cols).any(|j| x[i * cols + j] != 0)).collect();
            let nz_cols: Vec<usize> = (0..cols).filter(|j| (0..rows).any(|i| x[i * cols + j] != 0)).collect();
            let bounds = [
                (nz_rows[0], nz_rows.last().unwrap() + 1),
                (nz_cols[0], nz_cols.last().unwrap() + 1),
            ];
            let inputs: Vec<(String, i64)> = x.iter().enumerate().map(|(k, v)| (format!("x{k}"), *v)).collect();
            let (ok, outputs) = mock_run(&graph, &inputs);
            assert!(ok);
            assert_eq!(outputs, padded_crop(&x, rows, cols, bounds), "{rows}x{cols} x={x:?}");
        }
    }
}

#[test]
fn a_zero_test_flag_does_not_force_its_cells_to_zero() {
    // flag = (count of zeros in the row) != 0: an unset flag means the row
    // has no zero, which forces nothing to 0.
    let mut b = IRBuilder::new();
    let cells: Vec<Value> = (0..3).map(|k| input(&mut b, format!("x{k}"))).collect();
    let zero = b.ir_constant_int(0);
    let is_zero: Vec<Value> = cells.iter().map(|c| b.ir_equal_i(c, &zero)).collect();
    let s = b.ir_add_i(&is_zero[0], &is_zero[1]);
    let s = b.ir_add_i(&s, &is_zero[2]);
    let flag = b.ir_not_equal_i(&s, &zero);
    let forced = forced_zero_by_unset(&b.stmts, flag.stmt_id().unwrap());
    assert!(cells.iter().all(|c| !forced.contains(&c.stmt_id().unwrap())));
    // A sum with a term of unknown sign forces nothing past it either.
    let mixed = b.ir_add_i(&cells[0], &cells[1]);
    let flag = b.ir_not_equal_i(&mixed, &zero);
    let forced = forced_zero_by_unset(&b.stmts, flag.stmt_id().unwrap());
    assert!(!forced.contains(&cells[0].stmt_id().unwrap()));
}

#[test]
fn argmax_of_flags_and_a_read_at_it_match_numpy_on_every_flag_vector() {
    // v = argmax(f) (0 when no flag is set), then x[v] through the recorded
    // one-hot (`gather_by_encoding`).
    for n in 1..=6 {
        let mut b = IRBuilder::new();
        let flags: Vec<Value> = (0..n).map(|i| input(&mut b, format!("f{i}"))).collect();
        let x: Vec<Value> = (0..n).map(|i| input(&mut b, format!("x{i}"))).collect();
        let v = super::scan::argmax_of_flags(&mut b, &flags).unwrap();
        let read = super::element::gather_by_encoding(&mut b, &x, &v, NumberType::Integer).unwrap();
        b.ir_expose_public_i(&v);
        b.ir_expose_public_i(&read);
        let graph = b.export_ir_graph();
        for f in all_flag_vectors(n) {
            let xs: Vec<i64> = (0..n as i64).map(|i| 100 - 7 * i).collect();
            let mut inputs: Vec<(String, i64)> = f.iter().enumerate().map(|(i, v)| (format!("f{i}"), *v)).collect();
            inputs.extend(xs.iter().enumerate().map(|(i, v)| (format!("x{i}"), *v)));
            let argmax = f.iter().position(|&v| v == 1).unwrap_or(0);
            let (ok, outputs) = mock_run(&graph, &inputs);
            assert!(ok);
            assert_eq!(outputs, vec![argmax as i64, xs[argmax]], "flags {f:?}");
        }
    }
}
