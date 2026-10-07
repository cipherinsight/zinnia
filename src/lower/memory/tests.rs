//! Tests of the memory family: element reads lowered with each of its
//! constructions forced, run on the mock backend and compared with NumPy's
//! `x[i]` for every index around the line (no witness where NumPy raises);
//! and the layer's selection over all element constructions.
//!
//! The constructions use no prover hint: a read is a polynomial of the cells
//! and of exact gadget outputs (index bits), so a wrong claimed output is a
//! different public value, which every backend's instance check rejects; the
//! backends' tests (`halo2_base::synthesizer_tests::memory_reads_*`,
//! `noir::soundness_tests::memory_reads_*`) run the same graphs.

use crate::builder::IRBuilder;
use crate::circuit_input::InputPath;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::lower::{Construction, OpKind, Target};
use crate::types::Value;


pub(crate) use crate::lower::sort::test_support::mock_run;

/// A graph of reads, its inputs, and NumPy's outputs (`None`: NumPy raises).
pub(crate) struct ReadGraph {
    pub graph: IRGraph,
    pub inputs: Vec<(String, i64)>,
    pub expected: Option<Vec<i64>>,
}

/// `rows` lines of `n` integer cells (`x{r}_{k} = 10 r + k + 1`), each read
/// at the inputs `i` and `j` (in that order, exposed), the indices checked
/// by the program as the frontend checks them (`discharge_index_in_range`)
/// when `checked`; lowered by `only` (a set of constructions), or by the
/// whole registry's selection on `target`.
pub(crate) fn read_graph(
    n: usize,
    rows: usize,
    (i, j): (i64, i64),
    checked: bool,
    only: Option<&[&dyn Construction]>,
    target: Target,
) -> ReadGraph {
    let mut b = IRBuilder::new();
    let mut inputs = Vec::new();
    let mut read = |b: &mut IRBuilder, name: String, value: i64| {
        inputs.push((name.clone(), value));
        b.ir_read_integer(InputPath::new(name, vec![]), false)
    };
    let lines: Vec<Vec<Value>> = (0..rows)
        .map(|r| (0..n).map(|k| read(&mut b, format!("x{r}_{k}"), 10 * r as i64 + k as i64 + 1)).collect())
        .collect();
    let mut indices = [read(&mut b, "i".into(), i), read(&mut b, "j".into(), j)];
    if checked {
        for index in indices.iter_mut() {
            *index = crate::optim::resolver::discharge_index_in_range(&mut b, index, 0, n as i64, "test read");
        }
    }
    for line in &lines {
        for index in &indices {
            let lane = crate::lower::gather::checked_element_lane(&mut b, line, index);
            b.ir_expose_public_i(&lane);
        }
    }
    let graph = b.export_ir_graph();
    let registry = crate::lower::registry();
    let constructions: Vec<&dyn Construction> = match only {
        Some(only) => only.to_vec(),
        None => registry,
    };
    let lowered = crate::lower::emit::lower(&graph, target, &constructions).expect("reads to lower");
    let in_range = |x: i64| (0..n as i64).contains(&x);
    let expected = (in_range(i) && in_range(j))
        .then(|| (0..rows).flat_map(|r| [i, j].map(|x| 10 * r as i64 + x + 1)).collect());
    ReadGraph { graph: crate::optim::IRPass::exec(&crate::optim::DeadCodeElimination, lowered), inputs, expected }
}

fn has_placeholder(graph: &IRGraph) -> bool {
    graph.stmts.iter().any(|s| matches!(&s.ir, IR::ArrayRead(_)) || matches!(&s.ir, IR::ArrayOp(op) if op.kind == OpKind::Element))
}

fn check_against_numpy(case: &ReadGraph, what: &str) {
    assert!(!has_placeholder(&case.graph), "{what}: placeholder left");
    let (ok, outputs) = mock_run(&case.graph, &case.inputs);
    match &case.expected {
        Some(expected) => {
            assert!(ok, "{what}: rejected");
            assert_eq!(&outputs, expected, "{what}");
        }
        None => assert!(!ok, "{what}: accepted, NumPy raises"),
    }
}

/// Each construction of the family, alone (with the nodes it needs).
fn each_construction() -> Vec<(&'static str, Vec<&'static dyn Construction>)> {
    use super::constructions::{ElementByIndexBits, ElementByTable, IndexBitsNode};
    vec![("bits", vec![&ElementByIndexBits, &IndexBitsNode]), ("table", vec![&ElementByTable])]
}

#[test]
fn every_construction_reads_numpys_element_and_rejects_out_of_range() {
    for (name, constructions) in each_construction() {
        for target in [Target::Halo2Base, Target::Noir] {
            for n in [2usize, 3, 4, 5, 6, 7, 8, 9, 16, 17] {
                for rows in [1usize, 3] {
                    for i in -2..=n as i64 + 1 {
                        let j = (i + 1).rem_euclid(n as i64);
                        for checked in [false, true] {
                            let case = read_graph(n, rows, (i, j), checked, Some(&constructions), target);
                            check_against_numpy(&case, &format!("{name} {target:?} n={n} rows={rows} i={i} checked={checked}"));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn the_constructions_drop_the_programs_check() {
    // Each construction proves `0 <= i < n`; the frontend's check on the same
    // index is then not emitted.
    for (name, constructions) in each_construction() {
        let case = read_graph(5, 1, (1, 2), true, Some(&constructions), Target::Halo2Base);
        let checks = case.graph.stmts.iter().filter(|s| matches!(s.ir, IR::AssertU128Lt)).count();
        assert_eq!(checks, 0, "{name}: {:#?}", case.graph.stmts.iter().map(|s| &s.ir).collect::<Vec<_>>());
    }
}

#[test]
fn the_selection_reads_numpys_element_on_both_targets() {
    for target in [Target::Halo2Base, Target::Noir] {
        for n in [2usize, 3, 5, 16, 33, 64] {
            for rows in [1usize, 4] {
                for i in [-1, 0, n as i64 / 2, n as i64 - 1, n as i64] {
                    let case = read_graph(n, rows, (i, 0), true, None, target);
                    check_against_numpy(&case, &format!("{target:?} n={n} rows={rows} i={i}"));
                }
            }
        }
    }
}
