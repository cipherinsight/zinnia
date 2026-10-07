//! The mock's semantics of array, table and memory operations
//! (`crate::ir_array`): values, the predicate, positions outside their range
//! (no witness), and the RAM discipline.

use crate::circuit_input::{CircuitInputs, ResolvedWitness};
use crate::ir::{IRGraph, IRStatement};
use crate::ir_array::{AccessAxis, ArrayAccess, ArrayElem};
use crate::ir_defs::IR;
use crate::prove::mock::MockProverBackend;
use crate::prove::traits::ProverBackend;
use crate::prove::types::ProvingParams;

/// A graph of `(IR, operands)` in order (statement `k` is the `k`-th).
fn graph(stmts: Vec<(IR, Vec<u32>)>) -> IRGraph {
    IRGraph::new(
        stmts
            .into_iter()
            .enumerate()
            .map(|(i, (ir, args))| IRStatement::new(i as u32, crate::types::ValueId::next(), ir, args, vec![], None))
            .collect(),
    )
}

/// `Ok(true)` satisfied, `Ok(false)` unsatisfied, `Err` a synthesis error.
fn satisfied(g: &IRGraph) -> Result<bool, String> {
    let params = ProvingParams::default();
    let witness = ResolvedWitness::new(CircuitInputs::new(), params.precision_bits);
    let artifact = MockProverBackend.prove(g, &witness, &params).map_err(|e| e.to_string())?;
    Ok(artifact.proof_bytes == "mock_satisfied")
}

fn int(v: i64) -> (IR, Vec<u32>) {
    (IR::ConstantInt { value: v }, vec![])
}

/// Statements 0..4: the cells 10, 20, 30, 40; 4: `ArrayInit`.
fn four_cells() -> Vec<(IR, Vec<u32>)> {
    vec![int(10), int(20), int(30), int(40), (IR::ArrayInit { len: 4, elem: ArrayElem::Int }, vec![0, 1, 2, 3])]
}

/// `assert(eq(a, b))` appended to `stmts`.
fn assert_eq_stmts(stmts: &mut Vec<(IR, Vec<u32>)>, a: u32, b: u32) {
    let eq = stmts.len() as u32;
    stmts.push((IR::EqI, vec![a, b]));
    stmts.push((IR::Assert, vec![eq]));
}

fn flat4() -> ArrayAccess {
    ArrayAccess::new(4, ArrayElem::Int, 0, vec![AccessAxis { len: 4, stride: 1 }])
}

/// Write `value` at position `i` under predicate `p`, then read position
/// `j`, and assert the read equals `expected`.
fn write_then_read(i: i64, value: i64, p: i64, j: i64, expected: i64) -> Result<bool, String> {
    let mut s = four_cells();
    s.extend([int(i), int(value), int(p)]); // 5, 6, 7
    s.push((IR::ArrayWrite(flat4()), vec![4, 5, 6, 7])); // 8
    s.push(int(j)); // 9
    s.push((IR::ArrayRead(flat4()), vec![8, 9])); // 10
    s.push(int(expected)); // 11
    assert_eq_stmts(&mut s, 10, 11);
    satisfied(&graph(s))
}

#[test]
fn a_write_changes_its_cell_only_when_its_predicate_holds() {
    assert_eq!(write_then_read(2, 99, 1, 2, 99), Ok(true));
    assert_eq!(write_then_read(2, 99, 1, 1, 20), Ok(true));
    assert_eq!(write_then_read(2, 99, 0, 2, 30), Ok(true));
    assert_eq!(write_then_read(2, 99, 0, 2, 99), Ok(false));
}

#[test]
fn a_position_outside_its_axis_has_no_witness() {
    // A read at -1 or 4 of 4 cells, and a write at 4 even when not taken.
    assert_eq!(write_then_read(0, 1, 1, -1, 0), Ok(false));
    assert_eq!(write_then_read(0, 1, 1, 4, 0), Ok(false));
    assert_eq!(write_then_read(4, 1, 0, 0, 10), Ok(false));
}

#[test]
fn a_multi_axis_access_names_offset_plus_scaled_positions() {
    // 2x2 view of the 4 cells read at (1, 1) through access offset 0,
    // axes (2, 2), (2, 1): address 3.
    let access = ArrayAccess::new(4, ArrayElem::Int, 0, vec![AccessAxis { len: 2, stride: 2 }, AccessAxis { len: 2, stride: 1 }]);
    let mut s = four_cells();
    s.extend([int(1), int(1)]); // 5, 6
    s.push((IR::ArrayRead(access), vec![4, 5, 6])); // 7
    s.push(int(40)); // 8
    assert_eq_stmts(&mut s, 7, 8);
    assert_eq!(satisfied(&graph(s)), Ok(true));
}

#[test]
fn tables_and_memories_read_their_cells_and_reject_outside() {
    let rom = |address: i64, expected: i64| {
        let mut s = vec![int(5), int(6), (IR::RomInit { len: 2, elem: ArrayElem::Int }, vec![0, 1]), int(address)];
        s.push((IR::RomRead { len: 2, elem: ArrayElem::Int }, vec![2, 3])); // 4
        s.push(int(expected)); // 5
        assert_eq_stmts(&mut s, 4, 5);
        satisfied(&graph(s))
    };
    assert_eq!(rom(1, 6), Ok(true));
    assert_eq!(rom(2, 0), Ok(false));
    assert_eq!(rom(-1, 0), Ok(false));

    let ram = |address: i64, read_at: i64, expected: i64| {
        let mut s = vec![int(5), int(6), (IR::RamInit { len: 2, elem: ArrayElem::Int }, vec![0, 1]), int(address), int(9)];
        s.push((IR::RamWrite { len: 2, elem: ArrayElem::Int }, vec![2, 3, 4])); // 5
        s.push(int(read_at)); // 6
        s.push((IR::RamRead { len: 2, elem: ArrayElem::Int }, vec![5, 6])); // 7
        s.push(int(expected)); // 8
        assert_eq_stmts(&mut s, 7, 8);
        satisfied(&graph(s))
    };
    assert_eq!(ram(0, 0, 9), Ok(true));
    assert_eq!(ram(0, 1, 6), Ok(true));
    assert_eq!(ram(2, 0, 5), Ok(false));
}

#[test]
fn the_ram_discipline_is_enforced() {
    // Two writes of one version.
    let mut s = vec![int(5), (IR::RamInit { len: 1, elem: ArrayElem::Int }, vec![0]), int(0)];
    s.push((IR::RamWrite { len: 1, elem: ArrayElem::Int }, vec![1, 2, 0]));
    s.push((IR::RamWrite { len: 1, elem: ArrayElem::Int }, vec![1, 2, 0]));
    assert!(satisfied(&graph(s)).unwrap_err().contains("written twice"));
    // A read of a version after its write.
    let mut s = vec![int(5), (IR::RamInit { len: 1, elem: ArrayElem::Int }, vec![0]), int(0)];
    s.push((IR::RamWrite { len: 1, elem: ArrayElem::Int }, vec![1, 2, 0]));
    s.push((IR::RamRead { len: 1, elem: ArrayElem::Int }, vec![1, 2]));
    assert!(satisfied(&graph(s)).unwrap_err().contains("after its write"));
}
