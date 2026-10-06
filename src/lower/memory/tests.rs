//! Tests of the memory family: a read placeholder lowered with each
//! construction forced, run on the mock backend and compared with NumPy's
//! `x[i]` for every index around the line (no witness where NumPy raises),
//! and the selection's choice on lines of growing length.
//!
//! The constructions use no prover hint: the read is a polynomial of the
//! cells and of exact gadget outputs (equality tests, index bits), so a
//! wrong claimed output is a different public value, which every backend's
//! instance check rejects; the backends' tests (`halo2_base`
//! `memory_reads_reject_out_of_range_and_wrong_outputs`, Noir
//! `memory_reads_reject_out_of_range_in_noir`) run the same graphs.

use crate::builder::IRBuilder;
use crate::circuit_input::InputPath;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::lower::{Construction, OpKind, Target};
use crate::types::Value;

use super::constructions::{ReadByIndexBits, ReadByMemory, ReadByOneHot, ReadBySelectChain};

pub(crate) use crate::lower::sort::test_support::{mock_run, witness};

/// A graph of reads, its inputs, and NumPy's outputs (`None`: NumPy raises).
pub(crate) struct ReadGraph {
    pub graph: IRGraph,
    pub inputs: Vec<(String, i64)>,
    pub expected: Option<Vec<i64>>,
}

pub(crate) fn all_constructions() -> [&'static dyn Construction; 4] {
    [&ReadBySelectChain, &ReadByIndexBits, &ReadByOneHot, &ReadByMemory]
}

/// `rows` lines of `n` integer cells (`x{r}_{k} = 10 r + k + 1`), each read
/// at the input `i` and at the input `j`, exposed in that order; lowered by
/// `construction` alone (`None`: the layer's own selection on `target`).
/// Without a program check on the indices: a construction that does not
/// prove the bound gets one from the layer.
pub(crate) fn read_graph(
    n: usize,
    rows: usize,
    i: i64,
    j: i64,
    construction: Option<&dyn Construction>,
    target: Target,
) -> ReadGraph {
    let mut b = IRBuilder::new();
    let mut inputs = Vec::new();
    let mut read = |b: &mut IRBuilder, name: String, value: i64| {
        inputs.push((name.clone(), value));
        b.ir_read_integer(InputPath::new(name, vec![]), false)
    };
    let lines: Vec<Vec<Value>> =
        (0..rows).map(|r| (0..n).map(|k| read(&mut b, format!("x{r}_{k}"), 10 * r as i64 + k as i64 + 1)).collect()).collect();
    let (iv, jv) = (read(&mut b, "i".into(), i), read(&mut b, "j".into(), j));
    for line in &lines {
        for index in [&iv, &jv] {
            let lane = super::read_lane(&mut b, line, index).expect("a read of two or more cells");
            b.ir_expose_public_i(&lane);
        }
    }
    let graph = b.export_ir_graph();
    let registry = crate::lower::registry();
    let constructions: Vec<&dyn Construction> = match construction {
        Some(c) => vec![c],
        None => registry.clone(),
    };
    let lowered = crate::lower::emit::lower(&graph, target, &constructions).expect("reads to lower");
    let in_range = |x: i64| (0..n as i64).contains(&x);
    let expected = (in_range(i) && in_range(j)).then(|| {
        (0..rows).flat_map(|r| [i, j].map(|x| 10 * r as i64 + x + 1)).collect()
    });
    ReadGraph { graph: crate::optim::IRPass::exec(&crate::optim::DeadCodeElimination, lowered), inputs, expected }
}

fn has_placeholder(graph: &IRGraph) -> bool {
    graph.stmts.iter().any(|s| matches!(&s.ir, IR::ArrayOp(op) if op.kind == OpKind::Read))
}

#[test]
fn every_construction_reads_numpys_element_and_rejects_out_of_range() {
    for construction in all_constructions() {
        for n in [2usize, 3, 4, 5, 7, 8, 9] {
            for rows in [1usize, 3] {
                for i in -2..=n as i64 + 1 {
                    let j = (i + 1).rem_euclid(n as i64);
                    let case = read_graph(n, rows, i, j, Some(construction), Target::Halo2Base);
                    assert!(!has_placeholder(&case.graph), "{}: placeholder left", construction.name());
                    let (ok, outputs) = mock_run(&case.graph, &case.inputs);
                    match &case.expected {
                        Some(expected) => {
                            assert!(ok, "{} n={n} rows={rows} i={i}: rejected", construction.name());
                            assert_eq!(&outputs, expected, "{} n={n} rows={rows} i={i}", construction.name());
                        }
                        None => assert!(!ok, "{} n={n} rows={rows} i={i}: accepted, NumPy raises", construction.name()),
                    }
                }
            }
        }
    }
}

#[test]
fn the_selection_reads_numpys_element_on_both_targets() {
    for target in [Target::Halo2Base, Target::Noir] {
        for n in [2usize, 5, 16, 33] {
            for rows in [1usize, 4] {
                for i in [-1, 0, n as i64 / 2, n as i64 - 1, n as i64] {
                    let case = read_graph(n, rows, i, 0, None, target);
                    let (ok, outputs) = mock_run(&case.graph, &case.inputs);
                    assert_eq!(ok, case.expected.is_some(), "{target:?} n={n} rows={rows} i={i}");
                    if let Some(expected) = &case.expected {
                        assert_eq!(&outputs, expected, "{target:?} n={n} rows={rows} i={i}");
                    }
                }
            }
        }
    }
}

/// The selection's choice for one line of `n` cells read at `reads` distinct
/// indices: the construction names whose statements remain.
fn chosen(n: usize, rows: usize, target: Target) -> &'static str {
    let case = read_graph(n, rows, 0, 1, None, target);
    let ir = |f: fn(&IR) -> bool| case.graph.stmts.iter().any(|s| f(&s.ir));
    if ir(|ir| matches!(ir, IR::ReadMemory { .. })) {
        "memory"
    } else if ir(|ir| matches!(ir, IR::AssertU128LtBit { .. } | IR::SelectBitI | IR::SelectBitF)) {
        "bits"
    } else if ir(|ir| matches!(ir, IR::Assert)) && !ir(|ir| matches!(ir, IR::SelectI | IR::SelectF)) {
        "onehot"
    } else {
        "chain"
    }
}

#[test]
fn the_selection_changes_with_the_line_length_and_the_rows_sharing_an_index() {
    // One line, two reads: the multiplexer over the index bits on both
    // targets for short lines; Noir's native array for long lines.
    assert_eq!(chosen(4, 1, Target::Halo2Base), "bits");
    assert_eq!(chosen(4, 1, Target::Noir), "bits");
    assert_eq!(chosen(64, 1, Target::Noir), "memory");
    // Many lines read at the same two indices share the equality tests: the
    // one-hot's inner products are cheaper than a multiplexer per line.
    assert_eq!(chosen(16, 8, Target::Halo2Base), "onehot");
}
