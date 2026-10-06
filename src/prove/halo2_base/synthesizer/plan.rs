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
//! * **Asserted comparisons.** A comparison whose only use is an `Assert`
//!   needs no result bit: an order is range-checked directly, `a != b` is an
//!   inverse (see `Halo2BaseSynthesizer::assert_relation`, `assert_distinct`).

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
    /// Statement ids of the comparisons (`<`, `<=`, `>`, `>=`, `!=`, key
    /// orders) whose only use is `Assert`.
    pub asserted_comparisons: HashSet<u32>,
    /// Statement ids of address values that never need a cell: constants and
    /// affine combinations (`AddI`, `SubI`, `MulI` by a constant) whose every
    /// use is a memory address or another such value. A memory access reads
    /// them through their affine view (roots and constant offsets), so the
    /// lowering assigns no cell unless an access needs the address itself
    /// (an overlay comparison, an address bound the roots do not imply), and
    /// then computes it from the view (see `Halo2BaseSynthesizer::materialize_address`).
    pub address_only: HashSet<u32>,
}

impl Plan {
    pub fn of(ir: &IRGraph) -> Plan {
        let uses = uses_of(ir);
        Plan {
            bounds: planned_bounds(ir),
            deferred_products: deferred_products(ir, &uses),
            asserted_comparisons: asserted_comparisons(ir, &uses),
            address_only: address_only(ir, &uses),
        }
    }
}

/// The statements that use each statement's value, by statement id.
fn uses_of(ir: &IRGraph) -> HashMap<u32, Vec<u32>> {
    let mut uses: HashMap<u32, Vec<u32>> = HashMap::new();
    for stmt in &ir.stmts {
        for arg in &stmt.arguments {
            uses.entry(*arg).or_default().push(stmt.stmt_id);
        }
    }
    uses
}

/// The one statement that uses `stmt`'s value, if exactly one does.
fn only_user<'a>(ir: &'a IRGraph, uses: &HashMap<u32, Vec<u32>>, stmt: u32) -> Option<&'a crate::ir::IRStatement> {
    match uses.get(&stmt).map(Vec::as_slice) {
        Some([user]) => ir.stmts.get(*user as usize),
        _ => None,
    }
}

fn asserted_comparisons(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    ir.stmts
        .iter()
        .filter(|stmt| {
            matches!(
                stmt.ir,
                IR::LtI
                    | IR::LteI
                    | IR::GtI
                    | IR::GteI
                    | IR::LtF
                    | IR::LteF
                    | IR::GtF
                    | IR::GteF
                    | IR::NeI
                    | IR::NeF
                    | IR::KeyLess { .. }
            )
        })
        .filter(|stmt| only_user(ir, uses, stmt.stmt_id).is_some_and(|user| matches!(user.ir, IR::Assert)))
        .map(|stmt| stmt.stmt_id)
        .collect()
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
fn deferred_products(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    // Only the views of memory addresses are read (memory.rs translates
    // their roots to cells); a root of any other view may be deferred.
    let value_of: HashMap<u32, ValueId> = ir.stmts.iter().map(|s| (s.stmt_id, s.value_id)).collect();
    let roots: HashSet<ValueId> = ir
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::ReadMemory { .. } | IR::WriteMemory { .. }))
        .filter_map(|stmt| stmt.arguments.first().and_then(|a| value_of.get(a)))
        .filter_map(|vid| ir.provenance.affine_of(*vid))
        .flat_map(|view| view.terms.iter().map(|(root, _)| *root))
        .collect();
    ir.stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::MulI) && !roots.contains(&stmt.value_id))
        .filter(|stmt| {
            let Some(user) = only_user(ir, uses, stmt.stmt_id) else { return false };
            match (&user.ir, user.arguments.as_slice()) {
                // `add_f`/`sub_f` are the field addition of `add_i`/`sub_i`.
                (IR::AddI | IR::AddF, [a, b]) => a != b,
                (IR::SubI | IR::SubF, [a, b]) => *b == stmt.stmt_id && a != b,
                _ => false,
            }
        })
        .map(|stmt| stmt.stmt_id)
        .collect()
}

/// See [`Plan::address_only`]. Candidates are constants used only as memory
/// addresses (or by other candidates) and integer `+`, `-`, `*` whose value
/// has an affine view with at least one root; a candidate is dropped when a
/// use is anything else or when its view has a candidate (or itself) as a
/// root, until no candidate changes.
fn address_only(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    let stmt_of_vid: HashMap<ValueId, u32> = ir.stmts.iter().map(|s| (s.value_id, s.stmt_id)).collect();
    let mut set: HashSet<u32> = ir
        .stmts
        .iter()
        .filter(|stmt| match stmt.ir {
            IR::ConstantInt { .. } => true,
            IR::AddI | IR::SubI | IR::MulI => ir
                .provenance
                .affine_of(stmt.value_id)
                .is_some_and(|view| !view.terms.is_empty() && view.terms.iter().all(|(root, _)| *root != stmt.value_id)),
            _ => false,
        })
        .filter(|stmt| uses.get(&stmt.stmt_id).is_some_and(|u| !u.is_empty()))
        .map(|stmt| stmt.stmt_id)
        .collect();
    loop {
        let keep: HashSet<u32> = set
            .iter()
            .copied()
            .filter(|&id| {
                let stmt = &ir.stmts[id as usize];
                let uses_ok = uses.get(&id).into_iter().flatten().all(|&user| {
                    let u = &ir.stmts[user as usize];
                    match u.ir {
                        IR::ReadMemory { .. } => true,
                        IR::WriteMemory { .. } => {
                            matches!(stmt.ir, IR::ConstantInt { .. }) && u.arguments.first() == Some(&id) && u.arguments.get(1) != Some(&id)
                        }
                        _ => set.contains(&user),
                    }
                });
                let roots_ok = match ir.provenance.affine_of(stmt.value_id) {
                    Some(view) if !matches!(stmt.ir, IR::ConstantInt { .. }) => view
                        .terms
                        .iter()
                        .all(|(root, _)| stmt_of_vid.get(root).is_some_and(|r| !set.contains(r))),
                    _ => true,
                };
                uses_ok && roots_ok
            })
            .collect();
        if keep.len() == set.len() {
            return keep;
        }
        set = keep;
    }
}
