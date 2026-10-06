//! Assertions that a later operation's own constraints imply, read off the
//! IR (statements only, never witness values), shared by the backends.
//!
//! **Root guards.** The frontend asserts `np.sqrt`'s precondition `x >= 0`
//! before the root. Both backends lower the root of a value known to be
//! non-negative as `x·2^P = y^2 + d` with `y` and `d` range-checked far
//! below the field size; for an `x` already constrained into the signed
//! 64-bit window that identity is one of integers and has no solution
//! for a negative `x`. So when the backend knows `x` is in the window, the
//! assertion needs no constraint of its own: it holds for every witness
//! the root accepts (a circuit has no control flow, so the root's
//! constraints apply wherever they are emitted).
//!
//! **Divisor guards.** The frontend asserts a division's precondition
//! `d != 0` before it. Both backends divide by a divisor known to be
//! non-zero with a remainder bound against `|d|` (`0 <= |r| <= |d| - 1`,
//! both sides range-checked), which has no solution for `d = 0`. So the
//! assertion needs no constraint of its own when a later division by `d`
//! takes that path; the backend records `d` as non-zero instead (which is
//! what selects that path).

use std::collections::{HashMap, HashSet};

use crate::ir::IRGraph;
use crate::ir_defs::IR;

/// Statement ids of asserted `x >= 0` comparisons (`GteI`/`GteF(x, 0)` or
/// `LteI`/`LteF(0, x)`, used only by `Assert`) where a later statement is
/// `SqrtF(x)`.
pub fn root_guards(ir: &IRGraph) -> HashSet<u32> {
    let mut uses: HashMap<u32, Vec<u32>> = HashMap::new();
    for stmt in &ir.stmts {
        for arg in &stmt.arguments {
            uses.entry(*arg).or_default().push(stmt.stmt_id);
        }
    }
    let stmt = |id: u32| ir.stmts.get(id as usize);
    let is_zero = |id: u32| match stmt(id).map(|s| &s.ir) {
        Some(IR::ConstantInt { value }) => *value == 0,
        Some(IR::ConstantFloat { value }) => *value == 0.0,
        _ => false,
    };
    let only_asserted = |id: u32| {
        uses.get(&id).is_some_and(|users| !users.is_empty() && users.iter().all(|u| stmt(*u).is_some_and(|s| matches!(s.ir, IR::Assert))))
    };
    let root_after = |x: u32, after: u32| {
        uses.get(&x).is_some_and(|users| users.iter().any(|u| *u > after && stmt(*u).is_some_and(|s| matches!(s.ir, IR::SqrtF))))
    };
    ir.stmts
        .iter()
        .filter(|s| only_asserted(s.stmt_id))
        .filter_map(|s| {
            let x = match (&s.ir, s.arguments.as_slice()) {
                (IR::GteI | IR::GteF, [x, zero]) if is_zero(*zero) => *x,
                (IR::LteI | IR::LteF, [zero, x]) if is_zero(*zero) => *x,
                _ => return None,
            };
            root_after(x, s.stmt_id).then_some(s.stmt_id)
        })
        .collect()
}

/// The divisions whose lowering, for a divisor recorded as non-zero, bounds
/// the remainder by the divisor's magnitude (and so proves it non-zero):
/// `x / d`, `x // d` and `x % d` of floats, `x // d` and `x % d` of ints.
fn divides_with_remainder_bound(ir: &IR) -> bool {
    matches!(ir, IR::DivF | IR::FloorDivF | IR::ModF | IR::FloorDivI | IR::ModI)
}

/// Statement ids of asserted `d != 0` comparisons (`NeI`/`NeF` of `d` and
/// the constant 0, used only by `Assert`) where a later statement divides
/// by `d` ([`divides_with_remainder_bound`], `d` its second operand).
pub fn divisor_guards(ir: &IRGraph) -> HashSet<u32> {
    let mut uses: HashMap<u32, Vec<u32>> = HashMap::new();
    for stmt in &ir.stmts {
        for arg in &stmt.arguments {
            uses.entry(*arg).or_default().push(stmt.stmt_id);
        }
    }
    let stmt = |id: u32| ir.stmts.get(id as usize);
    let is_zero = |id: u32| match stmt(id).map(|s| &s.ir) {
        Some(IR::ConstantInt { value }) => *value == 0,
        Some(IR::ConstantFloat { value }) => *value == 0.0,
        _ => false,
    };
    let only_asserted = |id: u32| {
        uses.get(&id).is_some_and(|users| !users.is_empty() && users.iter().all(|u| stmt(*u).is_some_and(|s| matches!(s.ir, IR::Assert))))
    };
    let division_after = |d: u32, after: u32| {
        uses.get(&d).is_some_and(|users| {
            users.iter().any(|u| {
                *u > after
                    && stmt(*u).is_some_and(|s| divides_with_remainder_bound(&s.ir) && s.arguments.get(1) == Some(&d))
            })
        })
    };
    ir.stmts
        .iter()
        .filter(|s| matches!(s.ir, IR::NeI | IR::NeF) && only_asserted(s.stmt_id))
        .filter_map(|s| {
            let d = match s.arguments.as_slice() {
                [d, zero] if is_zero(*zero) && !is_zero(*d) => *d,
                [zero, d] if is_zero(*zero) && !is_zero(*d) => *d,
                _ => return None,
            };
            division_after(d, s.stmt_id).then_some(s.stmt_id)
        })
        .collect()
}
