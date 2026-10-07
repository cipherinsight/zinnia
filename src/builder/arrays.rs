//! Building array, table and memory operations (`crate::ir_array`).
//!
//! `ir_array_read` and `ir_array_write` apply the operations' definitions
//! where the operands decide them (`docs/array-ssa.md`, 2.4), so no pass
//! needs to forward memory:
//!
//! 1. `read(ArrayInit(c), k)` with a constant address `k` is `c[k]`;
//! 2. `read(write(v, j, x, p), k)` is `read(v, k)` when no cell `k` can
//!    name is one `j` can name (different constant addresses, or lines of
//!    cells that do not meet: `a[r, i]` and `a[s, j]` for rows `r != s`);
//! 3. `read(write(v, j, x, true), j)` (the same constant address, or the
//!    same access map at the same index statements) is `x` (under a runtime
//!    predicate the read stays: `select(p, x, read(v, j))` would read an
//!    older version after the write, which one memory cannot);
//! 4. `write(v, j, x, false)` is `v`.
//!
//! Each is the operations' semantics, valid on every path (both sides have
//! no witness for the same out-of-range positions: a read of a constant
//! address outside the array is never folded).

use crate::ir_array::{ArrayAccess, ArrayElem, WriteOperands};
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRBuilder;

/// The constant truth value of a predicate, if it has one.
fn constant_truth(p: &Value) -> Option<bool> {
    p.bool_val().or_else(|| p.int_val().map(|v| v != 0))
}

/// The address an access names when every position is a constant (`None`
/// otherwise, and for constant positions outside their axes).
fn constant_address(access: &ArrayAccess, positions: &[Value]) -> Option<usize> {
    let constants: Option<Vec<i64>> = positions.iter().map(Value::int_val).collect();
    access.address(&constants?)
}

/// Whether two accesses name the same cell on every path: the same
/// constant address, or the same access map at the same index statements.
fn same_cell(a: &ArrayAccess, a_positions: &[Value], b: &ArrayAccess, b_positions: &[Value]) -> bool {
    match (constant_address(a, a_positions), constant_address(b, b_positions)) {
        (Some(x), Some(y)) => x == y,
        _ => {
            a.offset == b.offset
                && a.axes == b.axes
                && a_positions.iter().zip(b_positions).all(|(p, q)| p.stmt_id().is_some() && p.stmt_id() == q.stmt_id())
        }
    }
}

/// Whether no cell one access can name is a cell the other can name
/// (`ArrayAccess::disjoint_from`, constant positions by their values).
fn disjoint(a: &ArrayAccess, a_positions: &[Value], b: &ArrayAccess, b_positions: &[Value]) -> bool {
    let constants = |positions: &[Value]| positions.iter().map(Value::int_val).collect::<Vec<_>>();
    a.disjoint_from(&constants(a_positions), b, &constants(b_positions))
}

/// What a read learns from the version it reads.
enum Step {
    /// The value read.
    Value(Value),
    /// Read this (earlier) version instead.
    Earlier(Value),
    /// Nothing: build the read.
    Opaque,
}

impl IRBuilder {
    /// `ArrayInit`: a version holding `cells`.
    pub fn ir_array_init(&mut self, cells: &[Value], elem: ArrayElem) -> Value {
        self.create_ir(&IR::ArrayInit { len: cells.len() as u32, elem }, cells)
    }

    /// An `ArrayInit` of `cells`, one per list of cells: reads of the same
    /// cells share it (and so form one element group in the layer).
    pub fn array_of_cells(&mut self, cells: &[Value], elem: ArrayElem) -> Value {
        let cells: Vec<Value> = cells.iter().map(|c| self.ensure_ptr(c)).collect();
        let key: Vec<StmtId> = cells.iter().map(|c| c.stmt_id().expect("a materialized cell")).collect();
        if let Some(array) = self.array_inits.get(&(elem, key.clone())) {
            return array.clone();
        }
        let array = self.ir_array_init(&cells, elem);
        self.array_inits.insert((elem, key), array.clone());
        array
    }

    /// `ArrayWrite`: `version` with the cell `access` names at `positions`
    /// set to `value` when `predicate` holds (fold 4 for a constant false
    /// predicate).
    pub fn ir_array_write(
        &mut self,
        version: &Value,
        access: ArrayAccess,
        positions: &[Value],
        value: &Value,
        predicate: &Value,
    ) -> Value {
        assert_eq!(positions.len(), access.rank(), "one position per axis");
        if constant_truth(predicate) == Some(false) {
            return version.clone();
        }
        let mut operands = Vec::with_capacity(positions.len() + 3);
        operands.push(version.clone());
        operands.extend_from_slice(positions);
        operands.push(value.clone());
        operands.push(predicate.clone());
        self.create_ir(&IR::ArrayWrite(access), &operands)
    }

    /// `ArrayRead`: the cell of `version` that `access` names at
    /// `positions`, with the folds of the module doc.
    pub fn ir_array_read(&mut self, version: &Value, access: ArrayAccess, positions: &[Value]) -> Value {
        assert_eq!(positions.len(), access.rank(), "one position per axis");
        let mut version = version.clone();
        loop {
            match self.read_step(&version, &access, positions) {
                Step::Value(v) => return v,
                Step::Earlier(earlier) => version = earlier,
                Step::Opaque => break,
            }
        }
        let mut operands = Vec::with_capacity(positions.len() + 1);
        operands.push(version);
        operands.extend_from_slice(positions);
        self.create_ir(&IR::ArrayRead(access), &operands)
    }

    /// One fold of a read of `version` (the module doc's rules 1-3).
    fn read_step(&self, version: &Value, access: &ArrayAccess, positions: &[Value]) -> Step {
        let Some(stmt) = version.stmt_id() else { return Step::Opaque };
        let (Some(def), Some(operands)) = (self.stmts.get(stmt as usize), self.array_defs.get(&stmt)) else {
            return Step::Opaque;
        };
        match &def.ir {
            IR::ArrayInit { .. } => match constant_address(access, positions) {
                Some(k) => Step::Value(operands[k].clone()),
                None => Step::Opaque,
            },
            IR::ArrayWrite(w_access) => {
                let w = WriteOperands::split(operands, w_access.rank());
                if same_cell(access, positions, w_access, w.positions) {
                    return match constant_truth(w.predicate) {
                        Some(true) => Step::Value(w.value.clone()),
                        _ => Step::Opaque,
                    };
                }
                if disjoint(access, positions, w_access, w.positions) {
                    Step::Earlier(w.version.clone())
                } else {
                    Step::Opaque
                }
            }
            _ => Step::Opaque,
        }
    }

    /// `RomInit`: a read-only table holding `cells`.
    pub fn ir_rom_init(&mut self, cells: &[Value], elem: ArrayElem) -> Value {
        self.create_ir(&IR::RomInit { len: cells.len() as u32, elem }, cells)
    }

    /// `RomRead`: the cell of `table` at `address` (no witness outside it).
    pub fn ir_rom_read(&mut self, table: &Value, len: u32, elem: ArrayElem, address: &Value) -> Value {
        self.create_ir(&IR::RomRead { len, elem }, &[table.clone(), address.clone()])
    }

    /// `RamInit`: a memory holding `cells` (its first version).
    pub fn ir_ram_init(&mut self, cells: &[Value], elem: ArrayElem) -> Value {
        self.create_ir(&IR::RamInit { len: cells.len() as u32, elem }, cells)
    }

    /// `RamWrite`: the memory's next version, `value` at `address`.
    pub fn ir_ram_write(&mut self, memory: &Value, len: u32, elem: ArrayElem, address: &Value, value: &Value) -> Value {
        self.create_ir(&IR::RamWrite { len, elem }, &[memory.clone(), address.clone(), value.clone()])
    }

    /// `RamRead`: the cell of the memory version at `address`.
    pub fn ir_ram_read(&mut self, memory: &Value, len: u32, elem: ArrayElem, address: &Value) -> Value {
        self.create_ir(&IR::RamRead { len, elem }, &[memory.clone(), address.clone()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_array::AccessAxis;

    fn int_cells(b: &mut IRBuilder, n: usize) -> Vec<Value> {
        (0..n).map(|k| b.ir_constant_int(10 * k as i64)).collect()
    }

    #[test]
    fn a_constant_read_of_an_init_is_its_cell() {
        let mut b = IRBuilder::new();
        let cells = int_cells(&mut b, 4);
        let v = b.ir_array_init(&cells, ArrayElem::Int);
        let y = b.ir_array_read(&v, ArrayAccess::constant(4, ArrayElem::Int, 2), &[]);
        assert_eq!(y.int_val(), Some(20));
        assert_eq!(y.stmt_id(), cells[2].stmt_id());
    }

    #[test]
    fn a_read_skips_writes_to_other_constant_cells_and_sees_its_own() {
        let mut b = IRBuilder::new();
        let cells = int_cells(&mut b, 4);
        let v0 = b.ir_array_init(&cells, ArrayElem::Int);
        let x = b.ir_read_integer(crate::circuit_input::InputPath::new("x", vec![]), false);
        let yes = b.ir_constant_bool(true);
        let at = |k| ArrayAccess::constant(4, ArrayElem::Int, k);
        let v1 = b.ir_array_write(&v0, at(1), &[], &x, &yes);
        let r3 = b.ir_array_read(&v1, at(3), &[]);
        assert_eq!(r3.int_val(), Some(30));
        let r1 = b.ir_array_read(&v1, at(1), &[]);
        assert_eq!(r1.stmt_id(), x.stmt_id());
    }

    #[test]
    fn a_read_at_the_index_of_a_predicated_write_stays_a_read() {
        let mut b = IRBuilder::new();
        let cells = int_cells(&mut b, 4);
        let v0 = b.ir_array_init(&cells, ArrayElem::Int);
        let i = b.ir_read_integer(crate::circuit_input::InputPath::new("i", vec![]), false);
        let x = b.ir_read_integer(crate::circuit_input::InputPath::new("x", vec![]), false);
        let p = b.ir_read_integer(crate::circuit_input::InputPath::new("p", vec![]), false);
        let flat = ArrayAccess::new(4, ArrayElem::Int, 0, vec![AccessAxis { len: 4, stride: 1 }]);
        let v1 = b.ir_array_write(&v0, flat.clone(), std::slice::from_ref(&i), &x, &p);
        let y = b.ir_array_read(&v1, flat.clone(), std::slice::from_ref(&i));
        // A read of `v1` (not of `v0` behind a select: one memory could not
        // read `v0` after the write).
        assert_eq!(b.stmts[y.stmt_id().unwrap() as usize].arguments[0], v1.stmt_id().unwrap());
        // Under a constant true predicate it is the value written.
        let yes = b.ir_constant_bool(true);
        let v2 = b.ir_array_write(&v1, flat.clone(), std::slice::from_ref(&i), &x, &yes);
        assert_eq!(b.ir_array_read(&v2, flat, std::slice::from_ref(&i)).stmt_id(), x.stmt_id());
    }

    #[test]
    fn a_write_under_a_false_predicate_is_no_write() {
        let mut b = IRBuilder::new();
        let cells = int_cells(&mut b, 2);
        let v0 = b.ir_array_init(&cells, ArrayElem::Int);
        let no = b.ir_constant_bool(false);
        let x = b.ir_constant_int(7);
        let v1 = b.ir_array_write(&v0, ArrayAccess::constant(2, ArrayElem::Int, 0), &[], &x, &no);
        assert_eq!(v1.stmt_id(), v0.stmt_id());
    }

    #[test]
    fn a_read_skips_writes_to_lines_it_cannot_meet() {
        // A 3x4 array: writes through a runtime column `c` of rows 0 and 1
        // (lines {0..3}, {4..7}), and through a runtime row `r` of column 2
        // (the line {2, 6, 10}).
        let mut b = IRBuilder::new();
        let cells = int_cells(&mut b, 12);
        let v0 = b.ir_array_init(&cells, ArrayElem::Int);
        let input = |b: &mut IRBuilder, name: &str| b.ir_read_integer(crate::circuit_input::InputPath::new(name, vec![]), false);
        let (c, r, x, y) = (input(&mut b, "c"), input(&mut b, "r"), input(&mut b, "x"), input(&mut b, "y"));
        let yes = b.ir_constant_bool(true);
        let row = |k: u32| ArrayAccess::new(12, ArrayElem::Int, 4 * k, vec![AccessAxis { len: 4, stride: 1 }]);
        let column = |k: u32| ArrayAccess::new(12, ArrayElem::Int, k, vec![AccessAxis { len: 3, stride: 4 }]);
        let v1 = b.ir_array_write(&v0, row(0), std::slice::from_ref(&c), &x, &yes);
        let v2 = b.ir_array_write(&v1, row(1), std::slice::from_ref(&c), &y, &yes);
        // Row 0 at `c` after the write to row 1: the write to row 0.
        let read = b.ir_array_read(&v2, row(0), std::slice::from_ref(&c));
        assert_eq!(read.stmt_id(), x.stmt_id());
        // Row 2 never met a write: a read of the init.
        let read = b.ir_array_read(&v2, row(2), std::slice::from_ref(&c));
        assert_eq!(b.stmts[read.stmt_id().unwrap() as usize].arguments[0], v0.stmt_id().unwrap());
        // Column 2 meets both rows: no fold.
        let read = b.ir_array_read(&v2, column(2), std::slice::from_ref(&r));
        assert_eq!(b.stmts[read.stmt_id().unwrap() as usize].arguments[0], v2.stmt_id().unwrap());
        // Columns 1 and 3 of one stride: a column-1 write is skipped by a
        // column-3 read and by the constant cell 11, not by the cell 5.
        let v3 = b.ir_array_write(&v2, column(1), std::slice::from_ref(&r), &x, &yes);
        let read = b.ir_array_read(&v3, column(3), std::slice::from_ref(&r));
        assert_eq!(b.stmts[read.stmt_id().unwrap() as usize].arguments[0], v2.stmt_id().unwrap());
        let read = b.ir_array_read(&v3, ArrayAccess::constant(12, ArrayElem::Int, 11), &[]);
        assert_eq!(read.int_val(), Some(110));
        let read = b.ir_array_read(&v3, ArrayAccess::constant(12, ArrayElem::Int, 5), &[]);
        assert_eq!(b.stmts[read.stmt_id().unwrap() as usize].arguments[0], v3.stmt_id().unwrap());
    }
}
