//! Arrays as values of the IR (`docs/array-ssa.md`).
//!
//! An **array version** is an IR statement whose value is a whole vector of
//! `len` cells. A write does not change a version; it defines a new one:
//!
//! ```text
//! v  = ArrayInit  { len, elem }      (c_0, ..., c_{len-1})
//! v' = ArrayWrite (access)           (v, i_0, ..., i_{r-1}, x, p)
//! y  = ArrayRead  (access)           (v, i_0, ..., i_{r-1})
//! ```
//!
//! The frontend emits these three; the array lowering layer
//! (`lower::ArrayLowering`) decides how each version tree is built (an
//! overlay of selects, a RAM trace, a table read, ...) and replaces them with
//! scalar statements and the backend-level table and memory operations:
//!
//! ```text
//! t  = RomInit  { len }   (c_0, ..., c_{len-1})    a read-only table
//! y  = RomRead  { len }   (t, a)                   t[a]
//! m  = RamInit  { len }   (c_0, ..., c_{len-1})    a read-write memory
//! m' = RamWrite { len }   (m, a, x)                m with m[a] := x
//! y  = RamRead  { len }   (m, a)                   m[a]
//! ```
//!
//! # Semantics (exact, on field elements; floats are Q32 like everywhere)
//!
//! * An [`ArrayAccess`] names the cell `offset + Σ_k i_k · stride_k` for the
//!   runtime positions `i_k` (its index operands). Every `i_k` must lie in
//!   `[0, len_k)`; otherwise there is **no witness** (no proof) on every
//!   backend and on the mock. The frontend passes positions after NumPy's
//!   index wrap, checked where the access runs and guarded where it does
//!   not, so this rule only rejects programs NumPy rejects.
//! * `ArrayWrite(v, i, x, p)`: `v` with the named cell set to `x` when the
//!   predicate `p` is nonzero; `v` itself when `p` is zero (a write in a
//!   branch not taken changes nothing). The index rule holds either way.
//! * `RomRead`, `RamRead`, `RamWrite` take a flat address `a`; one outside
//!   `[0, len)` has no witness.
//! * RAM discipline: every RAM version is consumed by at most one
//!   `RamWrite`, and every `RamRead` of a version lies (in statement order)
//!   after its definition and before the `RamWrite` that consumes it. So the
//!   statement order is a valid order of the memory's accesses.
//!
//! Every operation here is pure: unused, it is dead code; equal, it is one
//! value. None is ever the only proof of an index bound: the bound is a fact
//! the program's check or the lowering's construction establishes.

use serde::{Deserialize, Serialize};

/// The number type of an array's cells (and of a read).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArrayElem {
    Int,
    Float,
}

/// One runtime index of an [`ArrayAccess`]: a position in `[0, len)`,
/// scaled by `stride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AccessAxis {
    pub len: u32,
    pub stride: u32,
}

/// The cell an `ArrayRead` / `ArrayWrite` names in an array of `len` cells
/// of type `elem`: `offset + Σ_k i_k · axes[k].stride` for the access's
/// runtime positions `i_k` (in operand order). Compile-time indices are
/// folded into `offset`; an access without axes names one constant cell.
///
/// Invariant (checked by [`ArrayAccess::new`]): every position in range
/// names a cell of the array.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArrayAccess {
    pub len: u32,
    pub elem: ArrayElem,
    pub offset: u32,
    pub axes: Vec<AccessAxis>,
}

impl ArrayAccess {
    /// An access whose every in-range position names a cell of the array.
    ///
    /// # Panics
    /// When the largest address it can name is outside `[0, len)` or an axis
    /// is empty (a compiler error: the frontend builds accesses from the
    /// array's shape and strides).
    pub fn new(len: u32, elem: ArrayElem, offset: u32, axes: Vec<AccessAxis>) -> ArrayAccess {
        let access = ArrayAccess { len, elem, offset, axes };
        assert!(
            access.axes.iter().all(|a| a.len > 0),
            "array access with an empty axis: {access:?}"
        );
        assert!(
            access.last_address() < u64::from(len),
            "array access reaches past the array: {access:?}"
        );
        access
    }

    /// The access of one constant cell.
    pub fn constant(len: u32, elem: ArrayElem, address: u32) -> ArrayAccess {
        ArrayAccess::new(len, elem, address, Vec::new())
    }

    /// The access of a flat runtime address over the whole array.
    pub fn flat(len: u32, elem: ArrayElem) -> ArrayAccess {
        ArrayAccess::new(len, elem, 0, vec![AccessAxis { len, stride: 1 }])
    }

    /// The number of index operands.
    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    /// The largest address the access names (every position at its end).
    fn last_address(&self) -> u64 {
        u64::from(self.offset)
            + self.axes.iter().map(|a| u64::from(a.len - 1) * u64::from(a.stride)).sum::<u64>()
    }

    /// The address of the positions `positions` (one per axis), or `None`
    /// when one is outside its axis (no witness).
    pub fn address(&self, positions: &[i64]) -> Option<usize> {
        assert_eq!(positions.len(), self.axes.len(), "one position per axis");
        let mut address = u64::from(self.offset);
        for (axis, &p) in self.axes.iter().zip(positions) {
            let p = u64::try_from(p).ok().filter(|&p| p < u64::from(axis.len))?;
            address += p * u64::from(axis.stride);
        }
        Some(address as usize)
    }

    /// Every address the access can name, by its positions in row-major
    /// order of the axes (the first axis outermost). Its length is the
    /// product of the axes' lengths; an access without axes names one cell.
    pub fn line(&self) -> Vec<usize> {
        let mut addresses = vec![self.offset as usize];
        for axis in &self.axes {
            addresses = addresses
                .iter()
                .flat_map(|&base| (0..axis.len as usize).map(move |p| base + p * axis.stride as usize))
                .collect();
        }
        addresses
    }

    /// The constant address of an access without axes.
    pub fn constant_address(&self) -> Option<usize> {
        self.axes.is_empty().then_some(self.offset as usize)
    }

    pub fn describe(&self) -> String {
        let axes: Vec<String> = self.axes.iter().map(|a| format!("{}x{}", a.len, a.stride)).collect();
        format!("{}][{:?}][{}][{}", self.len, self.elem, self.offset, axes.join(","))
    }
}

/// The operands of an `ArrayWrite`: `[version, positions..., value, predicate]`.
pub struct WriteOperands<'a, T> {
    pub version: &'a T,
    pub positions: &'a [T],
    pub value: &'a T,
    pub predicate: &'a T,
}

impl<'a, T> WriteOperands<'a, T> {
    /// Split the operands of an `ArrayWrite` with `rank` index operands.
    pub fn split(operands: &'a [T], rank: usize) -> WriteOperands<'a, T> {
        assert_eq!(operands.len(), rank + 3, "ArrayWrite operands: version, {rank} positions, value, predicate");
        WriteOperands {
            version: &operands[0],
            positions: &operands[1..1 + rank],
            value: &operands[1 + rank],
            predicate: &operands[2 + rank],
        }
    }
}

/// The operands of an `ArrayRead`: `[version, positions...]`.
pub struct ReadOperands<'a, T> {
    pub version: &'a T,
    pub positions: &'a [T],
}

impl<'a, T> ReadOperands<'a, T> {
    pub fn split(operands: &'a [T], rank: usize) -> ReadOperands<'a, T> {
        assert_eq!(operands.len(), rank + 1, "ArrayRead operands: version, {rank} positions");
        ReadOperands { version: &operands[0], positions: &operands[1..] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_access_names_offset_plus_scaled_positions() {
        // a[1, :, j] of a 3x4x5 array: offset 20, axes (4, 5) and (5, 1).
        let access = ArrayAccess::new(60, ArrayElem::Int, 20, vec![AccessAxis { len: 4, stride: 5 }, AccessAxis { len: 5, stride: 1 }]);
        assert_eq!(access.address(&[0, 0]), Some(20));
        assert_eq!(access.address(&[3, 4]), Some(39));
        assert_eq!(access.address(&[4, 0]), None);
        assert_eq!(access.address(&[-1, 0]), None);
        assert_eq!(access.line().len(), 20);
        assert_eq!(access.line()[..6], [20, 21, 22, 23, 24, 25]);
        assert_eq!(ArrayAccess::constant(60, ArrayElem::Int, 7).line(), vec![7]);
    }

    #[test]
    #[should_panic(expected = "reaches past the array")]
    fn an_access_past_the_array_is_refused() {
        ArrayAccess::new(10, ArrayElem::Int, 5, vec![AccessAxis { len: 3, stride: 3 }]);
    }

    #[test]
    fn operands_split_by_rank() {
        let ops = [10, 11, 12, 13, 14];
        let w = WriteOperands::split(&ops, 2);
        assert_eq!((*w.version, w.positions, *w.value, *w.predicate), (10, &[11, 12][..], 13, 14));
        let r = ReadOperands::split(&ops[..3], 2);
        assert_eq!((*r.version, r.positions), (10, &[11, 12][..]));
    }
}
