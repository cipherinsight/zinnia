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

use super::versions::{NativeRam, Overlay, RamTrace};
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

/// The registry without the constructions named `without`.
fn registry_without(without: &[&str]) -> Vec<&'static dyn Construction> {
    crate::lower::registry().into_iter().filter(|c| !without.contains(&c.name())).collect()
}

/// Each representation of a memory group, with every element construction
/// (for reads of known cells): the overlay; the target's RAM; and the
/// selection's choice from the whole registry.
fn representations(target: Target) -> Vec<(&'static str, Vec<&'static dyn Construction>)> {
    let ram = [RamTrace.name(), NativeRam.name()];
    let other_ram = match target {
        Target::Halo2Base => NativeRam.name(),
        Target::Noir => RamTrace.name(),
    };
    vec![
        ("overlay", registry_without(&ram)),
        ("ram", registry_without(&[Overlay.name(), other_ram])),
        ("registry", crate::lower::registry()),
    ]
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

#[test]
fn a_write_of_a_value_read_from_its_own_version_lowers() {
    // `v_{t+1} = write(v_t, p_t, select(c_t, x, read(v_t, p_t)))`: each
    // write's value reads the version it writes (a read-modify-write chain,
    // as a masked runtime slice write once built), on every representation.
    for target in [Target::Halo2Base, Target::Noir] {
        for (name, constructions) in representations(target) {
            let n = 6usize;
            let mut b = IRBuilder::new();
            let mut inputs = Vec::new();
            let cells: Vec<Value> = (0..n)
                .map(|k| {
                    inputs.push((format!("x{k}"), k as i64));
                    b.ir_read_integer(InputPath::new(format!("x{k}"), vec![]), false)
                })
                .collect();
            let flat = ArrayAccess::new(n as u32, ArrayElem::Int, 0, vec![AccessAxis { len: n as u32, stride: 1 }]);
            let yes = b.ir_constant_bool(true);
            let mut version = b.ir_array_init(&cells, ArrayElem::Int);
            let mut model: Vec<i64> = (0..n as i64).collect();
            for (t, (p, c)) in [(1i64, true), (2, false), (1, true)].into_iter().enumerate() {
                inputs.push((format!("p{t}"), p));
                inputs.push((format!("c{t}"), i64::from(c)));
                let position = b.ir_read_integer(InputPath::new(format!("p{t}"), vec![]), false);
                let position = crate::optim::resolver::discharge_index_in_range(&mut b, &position, 0, n as i64, "test");
                let condition = b.ir_read_integer(InputPath::new(format!("c{t}"), vec![]), false);
                let condition = b.ir_bool_cast(&condition);
                let old = b.ir_array_read(&version, flat.clone(), std::slice::from_ref(&position));
                let ten = b.ir_constant_int(10);
                let new = b.ir_add_i(&old, &ten);
                let value = b.ir_select_i(&condition, &new, &old);
                version = b.ir_array_write(&version, flat.clone(), &[position], &value, &yes);
                if c {
                    model[p as usize] += 10;
                }
            }
            for k in 0..n {
                let read = b.ir_array_read(&version, ArrayAccess::constant(n as u32, ArrayElem::Int, k as u32), &[]);
                b.ir_expose_public_i(&read);
            }
            let p = Program { graph: b.export_ir_graph(), inputs, expected: Some(model) };
            check(&p, &constructions, target, &format!("{name} {target:?}"));
        }
    }
}

#[test]
fn zero_writes_by_keep_products_zero_exactly_the_cells_hit() {
    // A 3x4 array: `a[r_t, :] = 0` for two rows, then `a[:, c_t] = 0` for
    // two columns (one write per line), read back at every cell and at a
    // runtime position of row 1 between the two families.
    let (rows, cols) = (3u32, 4u32);
    let n = (rows * cols) as usize;
    for target in [Target::Halo2Base, Target::Noir] {
        for (r0, r1, c0, c1, j) in [(0i64, 2, 1, 3, 2i64), (1, 1, 0, 0, 0), (2, 0, 3, 2, 3)] {
            let mut b = IRBuilder::new();
            let mut inputs = Vec::new();
            let mut input = |b: &mut IRBuilder, name: String, value: i64| {
                inputs.push((name.clone(), value));
                b.ir_read_integer(InputPath::new(name, vec![]), false)
            };
            let cells: Vec<Value> = (0..n).map(|k| input(&mut b, format!("x{k}"), 10 * k as i64 + 1)).collect();
            let yes = b.ir_constant_bool(true);
            let zero = b.ir_constant_int(0);
            let mut version = b.ir_array_init(&cells, ArrayElem::Int);
            let mut model: Vec<i64> = (0..n as i64).map(|k| 10 * k + 1).collect();
            let mut checked = |b: &mut IRBuilder, name: &str, value: i64, len: u32| {
                let p = input(b, name.into(), value);
                crate::optim::resolver::discharge_index_in_range(b, &p, 0, i64::from(len), "test")
            };
            for (t, r) in [r0, r1].into_iter().enumerate() {
                let p = checked(&mut b, &format!("r{t}"), r, rows);
                for col in 0..cols {
                    let access = ArrayAccess::new(n as u32, ArrayElem::Int, col, vec![AccessAxis { len: rows, stride: cols }]);
                    version = b.ir_array_write(&version, access, std::slice::from_ref(&p), &zero, &yes);
                    model[(r as u32 * cols + col) as usize] = 0;
                }
            }
            let jp = checked(&mut b, "j", j, cols);
            let row1 = ArrayAccess::new(n as u32, ArrayElem::Int, cols, vec![AccessAxis { len: cols, stride: 1 }]);
            let read = b.ir_array_read(&version, row1, &[jp]);
            b.ir_expose_public_i(&read);
            let mut expected = vec![model[(cols as i64 + j) as usize]];
            for (t, c) in [c0, c1].into_iter().enumerate() {
                let p = checked(&mut b, &format!("c{t}"), c, cols);
                for row in 0..rows {
                    let access = ArrayAccess::new(n as u32, ArrayElem::Int, row * cols, vec![AccessAxis { len: cols, stride: 1 }]);
                    version = b.ir_array_write(&version, access, std::slice::from_ref(&p), &zero, &yes);
                    model[(row * cols + c as u32) as usize] = 0;
                }
            }
            for k in 0..n {
                let read = b.ir_array_read(&version, ArrayAccess::constant(n as u32, ArrayElem::Int, k as u32), &[]);
                b.ir_expose_public_i(&read);
            }
            expected.extend(model);
            let p = Program { graph: b.export_ir_graph(), inputs, expected: Some(expected) };
            let zero_writes = registry_without(&[Overlay.name(), RamTrace.name(), NativeRam.name()]);
            check(&p, &zero_writes, target, &format!("zero writes {target:?} {r0} {r1} {c0} {c1} {j}"));
            check(&p, &crate::lower::registry(), target, &format!("registry {target:?}"));
        }
    }
}

#[test]
fn the_ram_reads_an_older_version_when_the_later_writes_miss_its_cell() {
    // A 2x4 array: `v1` writes row 0 at `c`, `v2` row 1 at `c`; a read of
    // row 0 at `d` from `v1` after `v2` is built (as the folds leave it)
    // is a read of the memory then: the write to row 1 cannot change it.
    let mut b = IRBuilder::new();
    let cells: Vec<Value> = (0..8).map(|k| b.ir_constant_int(10 * k)).collect();
    let input = |b: &mut IRBuilder, name: &str| {
        let x = b.ir_read_integer(InputPath::new(name, vec![]), false);
        crate::optim::resolver::discharge_index_in_range(b, &x, 0, 4, "test")
    };
    let (c, d) = (input(&mut b, "c"), input(&mut b, "d"));
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let yes = b.ir_constant_bool(true);
    let row = |k: u32| ArrayAccess::new(8, ArrayElem::Int, 4 * k, vec![AccessAxis { len: 4, stride: 1 }]);
    let v0 = b.ir_array_init(&cells, ArrayElem::Int);
    let v1 = b.ir_array_write(&v0, row(0), std::slice::from_ref(&c), &x, &yes);
    let v2 = b.ir_array_write(&v1, row(1), std::slice::from_ref(&c), &x, &yes);
    let old = b.ir_array_read(&v1, row(0), std::slice::from_ref(&d));
    let new = b.ir_array_read(&v2, row(1), std::slice::from_ref(&d));
    b.ir_expose_public_i(&old);
    b.ir_expose_public_i(&new);
    let graph = b.export_ir_graph();
    let trees = super::versions::trees(&graph.stmts);
    assert!(trees.values().next().expect("one tree").is_chain_in_order());
    for target in [Target::Halo2Base, Target::Noir] {
        let ram: &'static dyn Construction = match target {
            Target::Halo2Base => &RamTrace,
            Target::Noir => &NativeRam,
        };
        let lowered = crate::lower::emit::lower(&graph, target, &[ram]).expect("lowered");
        assert!(lowered.stmts.iter().any(|s| matches!(s.ir, IR::RamInit { .. })), "{target:?}: a RAM");
        for (c, d) in [(1i64, 1i64), (1, 2), (3, 0)] {
            let (ok, outputs) = mock_run(&lowered, &[("c".into(), c), ("d".into(), d), ("x".into(), 7)]);
            assert!(ok);
            let row0 = if c == d { 7 } else { 10 * d };
            let row1 = if c == d { 7 } else { 10 * (4 + d) };
            assert_eq!(outputs, vec![row0, row1], "{target:?} c={c} d={d}");
        }
    }
}
