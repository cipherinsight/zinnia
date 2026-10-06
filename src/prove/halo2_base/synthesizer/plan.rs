//! Facts about the whole IR graph that the per-statement lowering uses
//! before the statements that establish them are reached.
//!
//! Every fact here is read off the IR's statements, never off witness
//! values, so key generation and proving take the same decisions.
//!
//! * **Planned bounds.** `AssertU128Lt(x, hi)` with a constant `hi` is a
//!   constraint of the circuit wherever it appears in the statement list: a
//!   circuit has no control flow, and every IR statement is emitted. When an
//!   earlier statement needs `x` inside the 64-bit window (an input-domain
//!   check `x >= i64::MIN`, a comparison), the bound check can be emitted
//!   there instead: it proves `0 <= x < hi`, which lies inside the window,
//!   and the later `AssertU128Lt` then finds it proven and emits nothing.
//!   The set of accepted witnesses is unchanged.
//! * **Deferred products** (see `Halo2BaseSynthesizer::mul_i`, `add_i`).

use std::collections::{HashMap, HashSet};

use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::ValueId;

#[derive(Default)]
pub(super) struct Plan {
    /// `x -> hi` for every `AssertU128Lt(x, hi)` with a constant `hi > 0`
    /// (the smallest when there are several).
    pub bounds: HashMap<ValueId, i128>,
    /// Statement ids of the `MulI` products whose only use is one `AddI`
    /// or the subtrahend of one `SubI`: the sum is emitted as one
    /// multiply-add gate there, and the product is never a cell.
    pub deferred_products: HashSet<u32>,
}

impl Plan {
    pub fn of(ir: &IRGraph) -> Plan {
        Plan { bounds: planned_bounds(ir), deferred_products: deferred_products(ir) }
    }
}

fn constant_int(ir: &IRGraph, stmt: u32) -> Option<i64> {
    match ir.stmts.get(stmt as usize).map(|s| &s.ir) {
        Some(IR::ConstantInt { value }) => Some(*value),
        _ => None,
    }
}

fn planned_bounds(ir: &IRGraph) -> HashMap<ValueId, i128> {
    let mut bounds: HashMap<ValueId, i128> = HashMap::new();
    for stmt in &ir.stmts {
        if !matches!(stmt.ir, IR::AssertU128Lt) || stmt.arguments.len() != 2 {
            continue;
        }
        let Some(hi) = constant_int(ir, stmt.arguments[1]).filter(|hi| *hi > 0) else { continue };
        let Some(x) = ir.stmts.get(stmt.arguments[0] as usize) else { continue };
        let entry = bounds.entry(x.value_id).or_insert(hi as i128);
        *entry = (*entry).min(hi as i128);
    }
    bounds
}

/// A `MulI` qualifies when its value is an argument of exactly one
/// statement, an `AddI` (either operand, not both) or a `SubI` (the
/// subtrahend), and is not a root of any affine view in the provenance
/// table (memory reads translate roots to cells, so a root must be a cell).
fn deferred_products(ir: &IRGraph) -> HashSet<u32> {
    let mut uses: HashMap<u32, Vec<u32>> = HashMap::new();
    for stmt in &ir.stmts {
        for arg in &stmt.arguments {
            uses.entry(*arg).or_default().push(stmt.stmt_id);
        }
    }
    let roots: HashSet<ValueId> = ir
        .stmts
        .iter()
        .filter_map(|stmt| ir.provenance.affine_of(stmt.value_id))
        .flat_map(|view| view.terms.iter().map(|(root, _)| *root))
        .collect();
    ir.stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::MulI) && !roots.contains(&stmt.value_id))
        .filter(|stmt| {
            let Some([user]) = uses.get(&stmt.stmt_id).map(Vec::as_slice) else { return false };
            let Some(user) = ir.stmts.get(*user as usize) else { return false };
            match (&user.ir, user.arguments.as_slice()) {
                (IR::AddI, [a, b]) => a != b,
                (IR::SubI, [a, b]) => *b == stmt.stmt_id && a != b,
                _ => false,
            }
        })
        .map(|stmt| stmt.stmt_id)
        .collect()
}
