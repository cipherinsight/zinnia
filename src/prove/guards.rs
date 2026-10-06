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
