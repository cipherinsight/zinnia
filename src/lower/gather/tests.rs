//! Tests of the gather family: each generator built on input reads, run on
//! the mock backend and compared with the NumPy result it implements, over
//! every input of small sizes (size 1 included); and the placeholders
//! lowered by the layer with each construction forced, against NumPy.
//!
//! The constructions use no prover hint (every value is a polynomial of the
//! inputs and of comparison bits), so their soundness rests on those
//! polynomials being NumPy's result, which the exhaustive comparisons check.

use crate::builder::IRBuilder;
use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::lower::{Construction, Target};
use crate::prove::types::ProvingParams;
use crate::types::{NumberType, Value};

use crate::lower::memory::ElementByTable;
use super::constructions::{
    ElementByIndicators, ElementByOneHot, ElementBySelectChain, ElementOfWrite, ScanByRunningProducts,
    ScanBySelectChain, WindowShift,
};
use super::encoding::IndexEncoding;
use super::scan::{self, ScanKind};
use super::vanishing::forced_zero_by_unset;
use super::window::{window_shift, AxisWindow};

fn input(b: &mut IRBuilder, name: String) -> Value {
    b.ir_read_integer(InputPath::new(name, vec![]), false)
}

fn named(prefix: &str, values: &[i64]) -> Vec<(String, i64)> {
    values.iter().enumerate().map(|(k, v)| (format!("{prefix}{k}"), *v)).collect()
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

/// NumPy's answer for a scan of `flags` (`None`: no answer, empty min/max).
fn reference_scan(kind: ScanKind, flags: &[i64]) -> Option<usize> {
    let set: Vec<usize> = (0..flags.len()).filter(|&i| flags[i] == 1).collect();
    match kind {
        ScanKind::First => set.first().copied(),
        ScanKind::Last => set.last().copied(),
        ScanKind::Argmax => Some(set.first().copied().unwrap_or(0)),
    }
}

const KINDS: [ScanKind; 3] = [ScanKind::First, ScanKind::Last, ScanKind::Argmax];

// ── Generators ────────────────────────────────────────────────────────

/// `scan`'s value, one-hot and (when present) thermometer, exposed in that
/// order, then `select_chain`'s value, for flags `f{i}`.
fn scan_graph(n: usize, kind: ScanKind) -> IRGraph {
    let mut b = IRBuilder::new();
    let flags: Vec<Value> = (0..n).map(|i| input(&mut b, format!("f{i}"))).collect();
    let (v, enc) = scan::scan(&mut b, kind, &flags);
    b.ir_expose_public_i(&v);
    for bit in &enc.onehot {
        b.ir_expose_public_i(bit);
    }
    for bit in enc.below.iter().flat_map(|below| &below[1..]) {
        let value = bit.materialize(&mut b);
        b.ir_expose_public_i(&value);
    }
    let chain = scan::select_chain(&mut b, kind, &flags);
    b.ir_expose_public_i(&chain);
    b.export_ir_graph()
}

#[test]
fn scans_match_numpy_on_every_flag_vector() {
    for n in 1..=6 {
        for kind in KINDS {
            let graph = scan_graph(n, kind);
            for flags in all_flag_vectors(n) {
                let Some(v) = reference_scan(kind, &flags) else { continue };
                let mut expected = vec![v as i64];
                expected.extend((0..n).map(|k| (k == v) as i64));
                if kind.has_thermometer() {
                    expected.extend((1..n).map(|k| (v < k) as i64));
                }
                expected.push(v as i64);
                let (ok, outputs) = mock_run(&graph, &named("f", &flags));
                assert!(ok);
                assert_eq!(outputs, expected, "n={n} {kind:?} flags={flags:?}");
            }
        }
    }
}

#[test]
fn first_and_last_stay_in_range_on_the_empty_flag_vector() {
    // The frontend asserts non-emptiness; without it the value still lies in
    // [0, n), as the range fact recorded on the placeholder states.
    for n in 1..=5 {
        for kind in [ScanKind::First, ScanKind::Last] {
            let (ok, outputs) = mock_run(&scan_graph(n, kind), &named("f", &vec![0; n]));
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

fn plain_axis(axis: usize, start: Value, stop: Value) -> AxisWindow {
    AxisWindow { axis, start, stop, start_encoding: None, stop_encoding: None, padding_zero: false }
}

#[test]
fn window_shift_matches_numpy_crop_for_every_bound_pair() {
    // start in [0, n] (the precondition), stop in [0, n] (Python-normalized).
    for (rows, cols) in [(1, 1), (1, 3), (3, 1), (2, 3), (3, 4)] {
        let mut b = IRBuilder::new();
        let cells: Vec<Value> = (0..rows * cols).map(|k| input(&mut b, format!("x{k}"))).collect();
        let bounds: Vec<Value> = (0..4).map(|k| input(&mut b, format!("b{k}"))).collect();
        let windows =
            [plain_axis(0, bounds[0].clone(), bounds[1].clone()), plain_axis(1, bounds[2].clone(), bounds[3].clone())];
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
                        let mut inputs = named("x", &x);
                        inputs.extend(named("b", &[s0 as i64, e0 as i64, s1 as i64, e1 as i64]));
                        let (ok, outputs) = mock_run(&graph, &inputs);
                        assert!(ok);
                        let want = padded_crop(&x, rows, cols, [(s0, e0), (s1, e1)]);
                        assert_eq!(outputs, want, "{rows}x{cols} [{s0}:{e0}, {s1}:{e1}]");
                    }
                }
            }
        }
    }
}

/// `count_nonzero(line) != 0` for each row (`axis` 0) or column of a
/// row-major `rows x cols` array.
fn nonzero_line_flags(b: &mut IRBuilder, cells: &[Value], rows: usize, cols: usize, axis: usize) -> Vec<Value> {
    let zero = b.ir_constant_int(0);
    let nonzero: Vec<Value> = cells.iter().map(|c| b.ir_not_equal_i(c, &zero)).collect();
    let (lines, len) = if axis == 0 { (rows, cols) } else { (cols, rows) };
    (0..lines)
        .map(|l| {
            let at = |k: usize| if axis == 0 { l * cols + k } else { k * cols + l };
            let total = (1..len).fold(nonzero[at(0)].clone(), |acc, k| b.ir_add_i(&acc, &nonzero[at(k)]));
            b.ir_not_equal_i(&total, &zero)
        })
        .collect()
}

/// The nonzero bounding box of a row-major `rows x cols` array (not all 0).
fn nonzero_box(x: &[i64], rows: usize, cols: usize) -> [(usize, usize); 2] {
    let nz_rows: Vec<usize> = (0..rows).filter(|i| (0..cols).any(|j| x[i * cols + j] != 0)).collect();
    let nz_cols: Vec<usize> = (0..cols).filter(|j| (0..rows).any(|i| x[i * cols + j] != 0)).collect();
    [(nz_rows[0], nz_rows.last().unwrap() + 1), (nz_cols[0], nz_cols.last().unwrap() + 1)]
}

#[test]
fn bounding_box_crop_from_scan_encodings_without_padding_mask_matches_numpy() {
    for (rows, cols) in [(1, 1), (1, 3), (2, 2), (3, 3)] {
        let mut b = IRBuilder::new();
        let cells: Vec<Value> = (0..rows * cols).map(|k| input(&mut b, format!("x{k}"))).collect();
        let mut windows = Vec::new();
        for axis in 0..2 {
            let flags = nonzero_line_flags(&mut b, &cells, rows, cols, axis);
            for (k, flag) in flags.iter().enumerate() {
                // Each flag being 0 forces its line to 0 (vanishing).
                let zeros = forced_zero_by_unset(&b.stmts, flag.stmt_id().unwrap());
                let line: Vec<usize> =
                    if axis == 0 { (0..cols).map(|j| k * cols + j).collect() } else { (0..rows).map(|i| i * cols + k).collect() };
                assert!(line.iter().all(|&c| zeros.contains(&cells[c].stmt_id().unwrap())));
            }
            let (start, first) = scan::scan(&mut b, ScanKind::First, &flags);
            let (last, last_enc) = scan::scan(&mut b, ScanKind::Last, &flags);
            let one = b.ir_constant_int(1);
            let stop = b.ir_add_i(&last, &one);
            windows.push(AxisWindow {
                axis,
                start,
                stop,
                start_encoding: Some(first),
                stop_encoding: Some(last_enc.shifted(1)),
                padding_zero: true,
            });
        }
        let out = window_shift(&mut b, cells, &[rows, cols], &windows, NumberType::Integer);
        for cell in &out {
            b.ir_expose_public_i(cell);
        }
        let graph = b.export_ir_graph();
        // No equality test and no comparison with `stop`: all from the scans.
        assert!(!graph.stmts.iter().any(|s| matches!(s.ir, IR::LtIBounded | IR::EqI)));
        for pattern in all_flag_vectors(rows * cols).filter(|p| p.contains(&1)) {
            let x: Vec<i64> = pattern.iter().enumerate().map(|(k, p)| p * (k as i64 + 2)).collect();
            let (ok, outputs) = mock_run(&graph, &named("x", &x));
            assert!(ok);
            assert_eq!(outputs, padded_crop(&x, rows, cols, nonzero_box(&x, rows, cols)), "{rows}x{cols} x={x:?}");
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
fn a_read_by_the_argmax_one_hot_matches_numpy() {
    for n in 1..=6 {
        let mut b = IRBuilder::new();
        let flags: Vec<Value> = (0..n).map(|i| input(&mut b, format!("f{i}"))).collect();
        let x: Vec<Value> = (0..n).map(|i| input(&mut b, format!("x{i}"))).collect();
        let (v, enc): (Value, IndexEncoding) = scan::scan(&mut b, ScanKind::Argmax, &flags);
        let read = super::element::gather_by_onehot(&mut b, &x, &enc, NumberType::Integer);
        b.ir_expose_public_i(&v);
        b.ir_expose_public_i(&read);
        let graph = b.export_ir_graph();
        let xs: Vec<i64> = (0..n as i64).map(|i| 100 - 7 * i).collect();
        for f in all_flag_vectors(n) {
            let mut inputs = named("f", &f);
            inputs.extend(named("x", &xs));
            let argmax = reference_scan(ScanKind::Argmax, &f).unwrap();
            let (ok, outputs) = mock_run(&graph, &inputs);
            assert!(ok);
            assert_eq!(outputs, vec![argmax as i64, xs[argmax]], "flags {f:?}");
        }
    }
}

// ── Placeholders lowered by the layer ─────────────────────────────────

/// Lower `graph` with exactly `constructions` (forcing a choice).
fn lowered(graph: &IRGraph, constructions: &[&dyn Construction]) -> IRGraph {
    let out = crate::lower::emit::lower(graph, Target::Halo2Base, constructions).expect("placeholders to lower");
    assert!(!out.stmts.iter().any(|s| matches!(s.ir, IR::ArrayOp(_))), "every placeholder lowered");
    out
}

/// The bounding-box crop of the nonzero entries of a `rows x cols` input as
/// placeholders: first/last set row and column, then the window.
fn bbox_crop_placeholders(rows: usize, cols: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let cells: Vec<Value> = (0..rows * cols).map(|k| input(&mut b, format!("x{k}"))).collect();
    let one = b.ir_constant_int(1);
    let mut bounds = Vec::new();
    for axis in 0..2 {
        let flags = nonzero_line_flags(&mut b, &cells, rows, cols, axis);
        let start = super::scan_lane(&mut b, ScanKind::First, &flags);
        let last = super::scan_lane(&mut b, ScanKind::Last, &flags);
        bounds.push((start, b.ir_add_i(&last, &one)));
    }
    for cell in super::window_lanes(&mut b, &cells, &[rows, cols], &bounds).unwrap() {
        b.ir_expose_public_i(&cell);
    }
    b.export_ir_graph()
}

#[test]
fn bounding_box_crop_placeholders_match_numpy_under_every_construction_choice() {
    let choices: [&[&dyn Construction]; 3] = [
        &[&ScanByRunningProducts, &WindowShift],
        &[&ScanBySelectChain, &WindowShift],
        &[&ScanByRunningProducts, &ScanBySelectChain, &WindowShift],
    ];
    for (rows, cols) in [(1, 2), (2, 2), (2, 3)] {
        let graph = bbox_crop_placeholders(rows, cols);
        for constructions in choices {
            let graph = lowered(&graph, constructions);
            for pattern in all_flag_vectors(rows * cols).filter(|p| p.contains(&1)) {
                let x: Vec<i64> = pattern.iter().enumerate().map(|(k, p)| p * (3 * k as i64 - 4)).collect();
                let (ok, outputs) = mock_run(&graph, &named("x", &x));
                assert!(ok);
                assert_eq!(outputs, padded_crop(&x, rows, cols, nonzero_box(&x, rows, cols)), "{rows}x{cols} x={x:?}");
            }
        }
    }
}

#[test]
fn the_selection_reads_window_selectors_off_the_scans() {
    // With both scan constructions offered, the window that reuses the
    // running products' one-hot and thermometer is cheaper than equality
    // tests and comparisons; the lowered crop then has neither.
    let graph = lowered(&bbox_crop_placeholders(3, 3), &[&ScanByRunningProducts, &ScanBySelectChain, &WindowShift]);
    assert!(!graph.stmts.iter().any(|s| matches!(s.ir, IR::LtIBounded | IR::EqI)));
}

/// `x[argmax(f)]` as placeholders.
fn argmax_read_placeholders(n: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let flags: Vec<Value> = (0..n).map(|i| input(&mut b, format!("f{i}"))).collect();
    let x: Vec<Value> = (0..n).map(|i| input(&mut b, format!("x{i}"))).collect();
    let v = super::scan_lane(&mut b, ScanKind::Argmax, &flags);
    let read = super::element_lane(&mut b, &x, &v).expect("the index is a scan's result");
    b.ir_expose_public_i(&v);
    b.ir_expose_public_i(&read);
    b.export_ir_graph()
}

#[test]
fn argmax_read_placeholders_match_numpy_under_every_construction_choice() {
    let choices: [&[&dyn Construction]; 3] = [
        &[&ScanByRunningProducts, &ElementByOneHot],
        &[&ScanBySelectChain, &ElementByTable],
        &[&ScanByRunningProducts, &ElementByTable],
    ];
    for n in 1..=5 {
        let graph = argmax_read_placeholders(n);
        for constructions in choices {
            let graph = lowered(&graph, constructions);
            let xs: Vec<i64> = (0..n as i64).map(|i| 5 * i - 9).collect();
            for f in all_flag_vectors(n) {
                let mut inputs = named("f", &f);
                inputs.extend(named("x", &xs));
                let argmax = reference_scan(ScanKind::Argmax, &f).unwrap();
                let (ok, outputs) = mock_run(&graph, &inputs);
                assert!(ok);
                assert_eq!(outputs, vec![argmax as i64, xs[argmax]], "flags {f:?}");
            }
        }
    }
}

// ── Reads at a checked runtime index ──────────────────────────────────

/// `x[i]` and `y[i]` for inputs `x`, `y` of `n` cells (two element groups at
/// one index) and an index `i` wrapped and checked as the frontend does
/// (NumPy's negative indices included), then `x[i]` again with `i` as a
/// second lane of `x`'s group at the index `n - 1 - w`.
fn checked_reads(n: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let x: Vec<Value> = (0..n).map(|k| input(&mut b, format!("x{k}"))).collect();
    let y: Vec<Value> = (0..n).map(|k| input(&mut b, format!("y{k}"))).collect();
    let i = input(&mut b, "i".to_string());
    let w = crate::helpers::static_array::base::normalize_dynamic_index(&mut b, &i, n);
    let last = b.ir_constant_int(n as i64 - 1);
    let mirrored = b.ir_sub_i(&last, &w);
    let reads = [
        super::checked_element_lane(&mut b, &x, &w),
        super::checked_element_lane(&mut b, &y, &w),
        super::checked_element_lane(&mut b, &x, &mirrored),
    ];
    for read in &reads {
        b.ir_expose_public_i(read);
    }
    b.export_ir_graph()
}

#[test]
fn checked_reads_match_numpy_under_every_construction_choice() {
    use crate::lower::memory::wrap::{WrapByPinnedSign, WrapByPositionHits};
    // The index is NumPy-wrapped (`lower::memory::wrap`): each choice with
    // each way of wrapping it.
    let choices: [&[&dyn Construction]; 8] = [
        &[&ElementBySelectChain, &WrapByPinnedSign],
        &[&ElementByIndicators, &WrapByPinnedSign],
        &[&ElementByTable, &WrapByPinnedSign],
        &[&ElementOfWrite, &ElementByOneHot, &ElementBySelectChain, &ElementByIndicators, &ElementByTable, &WrapByPinnedSign],
        &[&ElementBySelectChain, &WrapByPositionHits],
        &[&ElementByIndicators, &WrapByPositionHits],
        &[&ElementByOneHot, &ElementByIndicators, &WrapByPositionHits],
        &[&ElementOfWrite, &ElementByOneHot, &ElementBySelectChain, &ElementByIndicators, &ElementByTable, &WrapByPositionHits],
    ];
    for n in 1..=5usize {
        let graph = checked_reads(n);
        let xs: Vec<i64> = (0..n as i64).map(|k| 7 * k - 11).collect();
        let ys: Vec<i64> = (0..n as i64).map(|k| 100 - 3 * k).collect();
        for constructions in choices {
            for target in [Target::Halo2Base, Target::Noir] {
                let lowered = crate::lower::emit::lower(&graph, target, constructions).expect("placeholders");
                for i in -(n as i64) - 2..n as i64 + 2 {
                    let mut inputs = named("x", &xs);
                    inputs.extend(named("y", &ys));
                    inputs.push(("i".to_string(), i));
                    let (ok, outputs) = mock_run(&lowered, &inputs);
                    // NumPy raises outside [-n, n): no witness.
                    assert_eq!(ok, -(n as i64) <= i && i < n as i64, "n={n} i={i}");
                    if ok {
                        let w = i.rem_euclid(n as i64) as usize;
                        assert_eq!(outputs, vec![xs[w], ys[w], xs[n - 1 - w]], "n={n} i={i}");
                    }
                }
            }
        }
    }
}

/// The whole pass (lowering, then deduplication and dead-code elimination).
fn pass(graph: &IRGraph, target: Target) -> IRGraph {
    use crate::optim::IRPass;
    crate::lower::ArrayLowering { target }.exec(graph.clone())
}

/// `rows` reads of `cols` cells each, all at one checked index.
fn reads_at_one_index(rows: usize, cols: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let i = input(&mut b, "i".to_string());
    let w = crate::helpers::static_array::base::normalize_dynamic_index(&mut b, &i, cols);
    for r in 0..rows {
        let cells: Vec<Value> = (0..cols).map(|k| input(&mut b, format!("x{}", r * cols + k))).collect();
        let read = super::checked_element_lane(&mut b, &cells, &w);
        b.ir_expose_public_i(&read);
    }
    b.export_ir_graph()
}

#[test]
fn reads_sharing_an_index_are_inner_products_of_the_wraps_one_hot() {
    // 0345's column read: 8 rows of 6 cells at one column index. The wrap
    // is the pinned position one-hot (`lower::memory::wrap`) and its bits
    // are the reads' one-hot: no select at all (the only equality tests are
    // its `sum == 1` and its wrap relation), and it proves the index's
    // range, so no range check remains. On Noir a select of two variable
    // branches is two gates and a multiply-add one; on halo2-base a select
    // chain over the index's equality bits becomes an inner product of one
    // more term than it has links (`plan::OneHotChain`), 15 cells more than
    // the one-hot here (0345: 3,224 against 3,209).
    let graph = reads_at_one_index(8, 6);
    let count = |g: &IRGraph, f: fn(&IR) -> bool| g.stmts.iter().filter(|s| f(&s.ir)).count();
    for target in [Target::Noir, Target::Halo2Base] {
        let lowered = pass(&graph, target);
        assert_eq!(count(&lowered, |ir| matches!(ir, IR::SelectI)), 0, "{target:?}: no select");
        assert_eq!(count(&lowered, |ir| matches!(ir, IR::EqI)), 2, "{target:?}: the reads use the wrap's one-hot");
        assert_eq!(count(&lowered, |ir| matches!(ir, IR::AssertU128Lt)), 0, "{target:?}: the range check is replaced");
    }
}

/// The cells of `a[w] = v` (the frontend's one-hot write: cell `k` is
/// `select([w == k], v, a[k])`), read at `w` and at `n - 1 - w`.
fn read_after_write(n: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let a: Vec<Value> = (0..n).map(|k| input(&mut b, format!("a{k}"))).collect();
    let v = input(&mut b, "v".to_string());
    let i = input(&mut b, "i".to_string());
    let w = crate::helpers::static_array::base::normalize_dynamic_index(&mut b, &i, n);
    let written: Vec<Value> = a
        .iter()
        .enumerate()
        .map(|(k, old)| {
            let k = b.ir_constant_int(k as i64);
            let hit = b.ir_equal_i(&w, &k);
            b.ir_select_i(&hit, &v, old)
        })
        .collect();
    let last = b.ir_constant_int(n as i64 - 1);
    let mirrored = b.ir_sub_i(&last, &w);
    let same = super::checked_element_lane(&mut b, &written, &w);
    let other = super::checked_element_lane(&mut b, &written, &mirrored);
    b.ir_expose_public_i(&same);
    b.ir_expose_public_i(&other);
    b.export_ir_graph()
}

#[test]
fn a_read_at_the_index_just_written_is_the_written_value() {
    for n in 1..=4usize {
        let graph = read_after_write(n);
        for target in [Target::Halo2Base, Target::Noir] {
            let lowered = crate::lower::emit::lower(&graph, target, &crate::lower::registry()).unwrap();
            let a: Vec<i64> = (0..n as i64).map(|k| 3 * k + 1).collect();
            for i in -(n as i64) - 1..=n as i64 {
                let mut inputs = named("a", &a);
                inputs.push(("v".to_string(), -50));
                inputs.push(("i".to_string(), i));
                let (ok, outputs) = mock_run(&lowered, &inputs);
                assert_eq!(ok, -(n as i64) <= i && i < n as i64, "n={n} i={i}");
                if ok {
                    let w = i.rem_euclid(n as i64) as usize;
                    let mut after = a.clone();
                    after[w] = -50;
                    assert_eq!(outputs, vec![after[w], after[n - 1 - w]], "n={n} i={i}");
                }
            }
        }
    }
    // The read at the written index needs nothing; the other read keeps
    // the written cells.
    let only_same = {
        let mut b = IRBuilder::new();
        let a: Vec<Value> = (0..3).map(|k| input(&mut b, format!("a{k}"))).collect();
        let v = input(&mut b, "v".to_string());
        let w = input(&mut b, "w".to_string());
        let written: Vec<Value> = a
            .iter()
            .enumerate()
            .map(|(k, old)| {
                let k = b.ir_constant_int(k as i64);
                let hit = b.ir_equal_i(&w, &k);
                b.ir_select_i(&hit, &v, old)
            })
            .collect();
        let read = super::checked_element_lane(&mut b, &written, &w);
        b.ir_expose_public_i(&read);
        b.export_ir_graph()
    };
    let lowered = pass(&only_same, Target::Noir);
    assert!(!lowered.stmts.iter().any(|s| matches!(s.ir, IR::SelectI | IR::EqI)));
}

// ── Permutation scatter ───────────────────────────────────────────────

/// Every permutation of `0..n`.
fn permutations(n: usize) -> Vec<Vec<i64>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in permutations(n - 1) {
        for at in 0..=rest.len() {
            let mut p = rest.clone();
            p.insert(at, (n - 1) as i64);
            out.push(p);
        }
    }
    out
}

/// `c[p] = arange(n)` (the precondition records `p` as a permutation), then
/// `x[c[j]]` for every `j`, as placeholders; outputs `c` then the reads.
fn inverse_permutation_gather_placeholders(n: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let p: Vec<Value> = (0..n).map(|k| input(&mut b, format!("p{k}"))).collect();
    let x: Vec<Value> = (0..n).map(|k| input(&mut b, format!("x{k}"))).collect();
    b.record_permutation(&p);
    let values: Vec<Value> = (0..n).map(|k| b.ir_constant_int(k as i64)).collect();
    let c = super::permutation_scatter_lanes(&mut b, &p, &values).expect("a recorded permutation");
    let reads: Vec<Value> =
        c.iter().map(|cj| super::element_lane(&mut b, &x, cj).expect("the scatter's one-hot")).collect();
    for v in c.iter().chain(&reads) {
        b.ir_expose_public_i(v);
    }
    b.export_ir_graph()
}

#[test]
fn inverse_permutation_and_gathers_by_it_match_numpy_under_every_construction_choice() {
    let choices: [&[&dyn Construction]; 2] = [
        &[&super::constructions::ScatterByHits, &ElementByOneHot],
        &[&super::constructions::ScatterByHits, &ElementByTable],
    ];
    for n in 1..=4 {
        let graph = inverse_permutation_gather_placeholders(n);
        for constructions in choices {
            let graph = lowered(&graph, constructions);
            let xs: Vec<i64> = (0..n as i64).map(|k| 11 * k - 7).collect();
            for p in permutations(n) {
                let mut c = vec![0i64; n];
                for (k, &pk) in p.iter().enumerate() {
                    c[pk as usize] = k as i64;
                }
                let mut expected = c.clone();
                expected.extend(c.iter().map(|&cj| xs[cj as usize]));
                let mut inputs = named("p", &p);
                inputs.extend(named("x", &xs));
                let (ok, outputs) = mock_run(&graph, &inputs);
                assert!(ok);
                assert_eq!(outputs, expected, "p = {p:?}");
            }
        }
    }
}

#[test]
fn a_permutation_scatter_needs_a_recorded_permutation() {
    let mut b = IRBuilder::new();
    let p: Vec<Value> = (0..3).map(|k| input(&mut b, format!("p{k}"))).collect();
    let values: Vec<Value> = (0..3).map(|k| b.ir_constant_int(k)).collect();
    assert!(super::permutation_scatter_lanes(&mut b, &p, &values).is_none());
}

/// `rows` reads of 6 cells at NumPy's wrap `w = select(s, i + 6, i)` of the
/// input `i`, checked by `AssertU128Lt(w, 6)`, with the sign `s` a free
/// input (a tampered hint) asserted to be a bit.
fn reads_at_a_wrap_with_a_free_sign(rows: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let i = input(&mut b, "i".to_string());
    let s = input(&mut b, "s".to_string());
    b.ir_assert_bit(&s);
    let n = b.ir_constant_int(6);
    let shifted = b.ir_add_i(&i, &n);
    let w = b.ir_select_i(&s, &shifted, &i);
    b.ir_assert_u128_lt(&w, &n);
    for r in 0..rows {
        let cells: Vec<Value> = (0..6).map(|k| input(&mut b, format!("x{}", r * 6 + k))).collect();
        let read = super::checked_element_lane(&mut b, &cells, &w);
        b.ir_expose_public_i(&read);
    }
    b.export_ir_graph()
}

/// On Noir the indicators' asserted sum stands for the range check of a
/// non-power-of-two axis (0345's column index): the check is gone, and the
/// circuit still accepts exactly NumPy's indices with the honest sign.
#[test]
fn an_asserted_one_hot_replaces_the_index_check_and_keeps_the_sign_pinned() {
    let graph = reads_at_a_wrap_with_a_free_sign(8);
    let lowered = pass(&graph, Target::Noir);
    assert_eq!(lowered.stmts.iter().filter(|s| matches!(s.ir, IR::AssertU128Lt)).count(), 0);
    let xs: Vec<i64> = (0..48).map(|k| 3 * k - 40).collect();
    for i in -8..8i64 {
        for s in [-1, 0, 1, 2] {
            let mut inputs = named("x", &xs);
            inputs.push(("i".to_string(), i));
            inputs.push(("s".to_string(), s));
            let (ok, outputs) = mock_run(&lowered, &inputs);
            let honest = (-6..6).contains(&i) && s == i64::from(i < 0);
            assert_eq!(ok, honest, "i={i} s={s}");
            if ok {
                let w = i.rem_euclid(6) as usize;
                assert_eq!(outputs, (0..8).map(|r| xs[r * 6 + w]).collect::<Vec<_>>(), "i={i}");
            }
        }
    }
    // A power-of-two axis keeps its one range check (cheaper than the sum).
    let mut b = IRBuilder::new();
    let w = input(&mut b, "w".to_string());
    let eight = b.ir_constant_int(8);
    b.ir_assert_u128_lt(&w, &eight);
    for r in 0..6 {
        let cells: Vec<Value> = (0..8).map(|k| input(&mut b, format!("x{}", r * 8 + k))).collect();
        let read = super::checked_element_lane(&mut b, &cells, &w);
        b.ir_expose_public_i(&read);
    }
    let lowered = pass(&b.export_ir_graph(), Target::Noir);
    assert_eq!(lowered.stmts.iter().filter(|s| matches!(s.ir, IR::AssertU128Lt)).count(), 1);
}
