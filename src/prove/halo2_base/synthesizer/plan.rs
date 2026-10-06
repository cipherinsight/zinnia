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
    /// Statement ids of the `BitAndI` with a constant single-bit mask `2^j`
    /// whose every use is a comparison with a constant: such a comparison
    /// is the bit (two-valued facts), so `x & 2^j = 2^j·bit` needs no cell.
    pub cellless_bit_masks: HashSet<u32>,
    /// Asserted `x >= 0` with a later `sqrt(x)` (`crate::prove::guards`;
    /// see `Halo2BaseSynthesizer::assert_nonneg_by_root`).
    pub root_guards: HashSet<u32>,
}

impl Plan {
    pub fn of(ir: &IRGraph) -> Plan {
        let uses = uses_of(ir);
        Plan {
            bounds: planned_bounds(ir),
            deferred_products: deferred_products(ir, &uses),
            asserted_comparisons: asserted_comparisons(ir, &uses),
            cellless_bit_masks: cellless_bit_masks(ir, &uses),
            root_guards: crate::prove::guards::root_guards(ir),
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
/// statement, an `AddI`/`AddF` (either operand, not both) or a
/// `SubI`/`SubF` (the subtrahend), and is not a root of any affine view in the provenance
/// table (memory reads translate roots to cells, so a root must be a cell).
fn deferred_products(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
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
            let Some(user) = only_user(ir, uses, stmt.stmt_id) else { return false };
            match (&user.ir, user.arguments.as_slice()) {
                // `add_f`/`sub_f` are `add_i`/`sub_i` on the Q encodings.
                (IR::AddI | IR::AddF, [a, b]) => a != b,
                (IR::SubI | IR::SubF, [a, b]) => *b == stmt.stmt_id && a != b,
                _ => false,
            }
        })
        .map(|stmt| stmt.stmt_id)
        .collect()
}

/// A `BitAndI` of a cell and a constant `2^j` (`j < 63`) whose every user is
/// a comparison (`==`, `!=`, `<`, `<=`, `>`, `>=`) of it with a constant,
/// and which no affine view reads.
fn cellless_bit_masks(ir: &IRGraph, uses: &HashMap<u32, Vec<u32>>) -> HashSet<u32> {
    let roots: HashSet<ValueId> = ir
        .stmts
        .iter()
        .filter_map(|stmt| ir.provenance.affine_of(stmt.value_id))
        .flat_map(|view| view.terms.iter().map(|(root, _)| *root))
        .collect();
    let single_bit = |c: Option<i64>| c.is_some_and(|c| c > 0 && c & (c - 1) == 0);
    ir.stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::BitAndI) && !roots.contains(&stmt.value_id))
        .filter(|stmt| match stmt.arguments.as_slice() {
            [a, b] => single_bit(constant_int(ir, *a)) != single_bit(constant_int(ir, *b))
                && constant_int(ir, *a).is_none() != constant_int(ir, *b).is_none(),
            _ => false,
        })
        .filter(|stmt| {
            let Some(users) = uses.get(&stmt.stmt_id).filter(|u| !u.is_empty()) else { return false };
            users.iter().all(|user| {
                let Some(user) = ir.stmts.get(*user as usize) else { return false };
                let comparison = matches!(user.ir, IR::EqI | IR::NeI | IR::LtI | IR::LteI | IR::GtI | IR::GteI);
                comparison
                    && user.arguments.len() == 2
                    && user.arguments.iter().any(|a| *a != stmt.stmt_id && constant_int(ir, *a).is_some())
            })
        })
        .map(|stmt| stmt.stmt_id)
        .collect()
}
