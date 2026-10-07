//! Memory groups (`versions.rs`): a chain of writes at runtime positions,
//! some predicated, with reads at runtime and constant addresses after each
//! write, lowered with each representation forced (overlay with any element
//! read, RAM trace, native RAM) and by the whole registry on both targets,
//! run on the mock and compared with the array's values (NumPy's): every
//! read sees the last write the program executed, a write whose predicate
//! is false changes nothing, and a position outside its axis has no witness.

use crate::builder::IRBuilder;
use crate::circuit_input::InputPath;
use crate::ir::IRGraph;
use crate::ir_array::{AccessAxis, ArrayAccess, ArrayElem};
use crate::ir_defs::IR;
use crate::lower::{Construction, Target};
use crate::types::Value;

use super::versions::{NativeRam, RamTrace};
use crate::lower::sort::test_support::mock_run;

/// The program: `n` cells `x_k = 10 k + 1`; `writes` writes, write `t` of
/// `v_t` at position `i_t` under predicate `p_t` (when `predicated`), each
/// followed by a read at position `j_t` and a read of the constant address
/// `t mod n`, all exposed. The positions are checked by the program
/// (`discharge_index_in_range`).
struct Program {
    graph: IRGraph,
    inputs: Vec<(String, i64)>,
    /// The outputs, or `None` when a position is outside the array.
    expected: Option<Vec<i64>>,
}

fn program(n: usize, writes: &[(i64, i64, bool)], reads: &[i64], predicated: bool) -> Program {
    let mut b = IRBuilder::new();
    let mut inputs = Vec::new();
    let mut input = |b: &mut IRBuilder, name: String, value: i64| {
        inputs.push((name.clone(), value));
        b.ir_read_integer(InputPath::new(name, vec![]), false)
    };
    let cells: Vec<Value> = (0..n).map(|k| input(&mut b, format!("x{k}"), 10 * k as i64 + 1)).collect();
    let flat = ArrayAccess::new(n as u32, ArrayElem::Int, 0, vec![AccessAxis { len: n as u32, stride: 1 }]);
    let mut version = b.ir_array_init(&cells, ArrayElem::Int);
    let mut model: Vec<i64> = (0..n).map(|k| 10 * k as i64 + 1).collect();
    let mut expected = Vec::new();
    let mut in_range = true;
    for (t, (&(i, v, p), &j)) in writes.iter().zip(reads).enumerate() {
        let position = input(&mut b, format!("i{t}"), i);
        let position = crate::optim::resolver::discharge_index_in_range(&mut b, &position, 0, n as i64, "test write");
        let value = input(&mut b, format!("v{t}"), v);
        let predicate = if predicated { input(&mut b, format!("p{t}"), i64::from(p)) } else { b.ir_constant_bool(true) };
        version = b.ir_array_write(&version, flat.clone(), &[position], &value, &predicate);
        let at = input(&mut b, format!("j{t}"), j);
        let at = crate::optim::resolver::discharge_index_in_range(&mut b, &at, 0, n as i64, "test read");
        let read = b.ir_array_read(&version, flat.clone(), &[at]);
        b.ir_expose_public_i(&read);
        let constant = b.ir_array_read(&version, ArrayAccess::constant(n as u32, ArrayElem::Int, (t % n) as u32), &[]);
        b.ir_expose_public_i(&constant);
        let inside = |x: i64| (0..n as i64).contains(&x);
        in_range &= inside(i) && inside(j);
        if inside(i) && (p || !predicated) {
            model[i as usize] = v;
        }
        if inside(j) {
            expected.push(model[j as usize]);
        }
        expected.push(model[t % n]);
    }
    Program { graph: b.export_ir_graph(), inputs, expected: in_range.then_some(expected) }
}

/// Lower with `constructions` (all of them may be needed together) and run
/// the result on the mock.
fn check(p: &Program, constructions: &[&dyn Construction], target: Target, what: &str) {
    let lowered = crate::lower::emit::lower(&p.graph, target, constructions).expect("something to lower");
    let lowered = crate::optim::IRPass::exec(&crate::optim::DeadCodeElimination, lowered);
    assert!(
        !lowered.stmts.iter().any(|s| matches!(s.ir, IR::ArrayRead(_) | IR::ArrayWrite(_) | IR::ArrayOp(_))),
        "{what}: an array operation is left"
    );
    let (ok, outputs) = mock_run(&lowered, &p.inputs);
    match &p.expected {
        Some(expected) => {
            assert!(ok, "{what}: rejected");
            assert_eq!(&outputs, expected, "{what}");
        }
        None => assert!(!ok, "{what}: accepted, NumPy raises"),
    }
}

/// Every construction but the RAM ones: the overlay and any element read.
fn overlay_and_element_reads() -> Vec<&'static dyn Construction> {
    let ram = [RamTrace.name(), NativeRam.name()];
    crate::lower::registry().into_iter().filter(|c| !ram.contains(&c.name())).collect()
}

fn representations(target: Target) -> Vec<(&'static str, Vec<&'static dyn Construction>)> {
    let overlay = overlay_and_element_reads();
    let ram: &'static dyn Construction = match target {
        Target::Halo2Base => &RamTrace,
        Target::Noir => &NativeRam,
    };
    vec![("overlay", overlay), ("ram", vec![ram]), ("registry", crate::lower::registry())]
}

#[test]
fn every_representation_reads_the_last_executed_write() {
    for target in [Target::Halo2Base, Target::Noir] {
        for (name, constructions) in representations(target) {
            for n in [1usize, 2, 3, 5, 8] {
                for predicated in [false, true] {
                    let last = n as i64 - 1;
                    let writes = [(0, 100, true), (last, 200, false), (n as i64 / 2, 300, true), (last, 400, true)];
                    let reads = [0, last, n as i64 / 2, last];
                    let p = program(n, &writes, &reads, predicated);
                    check(&p, &constructions, target, &format!("{name} {target:?} n={n} predicated={predicated}"));
                }
            }
        }
    }
}

#[test]
fn a_position_outside_the_array_has_no_witness() {
    for target in [Target::Halo2Base, Target::Noir] {
        for (name, constructions) in representations(target) {
            let n = 4usize;
            for bad in [-1i64, 4, 5] {
                // A write outside, taken or not; a read outside.
                for (writes, reads) in [
                    ([(bad, 7, true), (1, 8, true)], [0, 1]),
                    ([(bad, 7, false), (1, 8, true)], [0, 1]),
                    ([(0, 7, true), (1, 8, true)], [bad, 1]),
                ] {
                    for predicated in [false, true] {
                        let p = program(n, &writes, &reads, predicated);
                        assert!(p.expected.is_none());
                        check(&p, &constructions, target, &format!("{name} {target:?} bad={bad} predicated={predicated}"));
                    }
                }
            }
        }
    }
}

#[test]
fn the_ram_trace_is_offered_only_for_a_chain_read_in_order() {
    // A read of the root after a write is an element read of the root's
    // cells (SSA: unaffected); a read of an older written version after a
    // newer write breaks the order and RAM is not offered (the overlay is).
    let mut b = IRBuilder::new();
    let cells: Vec<Value> = (0..3).map(|k| b.ir_constant_int(k)).collect();
    let flat = ArrayAccess::new(3, ArrayElem::Int, 0, vec![AccessAxis { len: 3, stride: 1 }]);
    let yes = b.ir_constant_bool(true);
    let i = b.ir_read_integer(InputPath::new("i", vec![]), false);
    let i = crate::optim::resolver::discharge_index_in_range(&mut b, &i, 0, 3, "test");
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let v0 = b.ir_array_init(&cells, ArrayElem::Int);
    let v1 = b.ir_array_write(&v0, flat.clone(), std::slice::from_ref(&i), &x, &yes);
    let v2 = b.ir_array_write(&v1, flat.clone(), std::slice::from_ref(&i), &i, &yes);
    let stale = b.ir_array_read(&v1, ArrayAccess::constant(3, ArrayElem::Int, 1), &[]);
    let fresh = b.ir_array_read(&v2, ArrayAccess::constant(3, ArrayElem::Int, 2), &[]);
    b.ir_expose_public_i(&stale);
    b.ir_expose_public_i(&fresh);
    let graph = b.export_ir_graph();
    let trees = super::versions::trees(&graph.stmts);
    let tree = trees.values().next().expect("one tree");
    assert!(!tree.is_chain_in_order());
    let lowered = crate::lower::emit::lower(&graph, Target::Halo2Base, &crate::lower::registry()).expect("lowered");
    assert!(!lowered.stmts.iter().any(|s| matches!(s.ir, IR::RamInit { .. })), "no RAM for an out-of-order read");
    for (i_value, x_value) in [(1, 9), (2, 9), (0, 9)] {
        let (ok, outputs) = mock_run(&lowered, &[("i".into(), i_value), ("x".into(), x_value)]);
        assert!(ok);
        let mut model = vec![0, 1, 2];
        model[i_value as usize] = x_value;
        let stale_expected = model[1];
        model[i_value as usize] = i_value;
        assert_eq!(outputs, vec![stale_expected, model[2]]);
    }
}
