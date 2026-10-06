//! Which values a flag being 0 forces to 0 (a free fact, read off the IR).
//!
//! A bounding-box crop takes its window from flags like
//! `has[i] = count_nonzero(x[i, :]) != 0`. Every position past the last set
//! flag then holds only zeros, so a window ending there needs no mask on its
//! padding: the shifted-in cells are 0 already (`window.rs`). This module
//! proves the implication `flag == 0  =>  c == 0` for each cell `c` it
//! returns, by structure alone:
//!
//! * `a != 0`, `0 < a` (with `a >= 0`), `not(a == 0)`, `bool(a)`: the flag is
//!   0 only when `a == 0`;
//! * `a == 0` with `a = s + t` and `s, t >= 0`: then `s == 0` and `t == 0`;
//! * `a == 0` with `a` a 0/1 value (a comparison, `not`, `and`, `bool`): the
//!   flag rules above apply to `a`;
//! * any other `a == 0`: `a` itself is a cell forced to 0.
//!
//! Non-negativity is structural too: 0/1 values and sums of non-negative
//! values. Every step is an implication valid for all field values the
//! circuit accepts (comparison results are constrained bits), so the
//! returned set is sound; it may be smaller than the truth.

use std::collections::HashSet;

use crate::ir::IRStatement;
use crate::ir_defs::IR;
use crate::types::StmtId;

/// Statements whose value is 0 whenever the 0/1 value `flag` is 0; `b`
/// holds every statement, indexed by statement id (an `IRGraph`'s or an
/// `IRBuilder`'s statements).
pub fn forced_zero_by_unset(b: &[IRStatement], flag: StmtId) -> HashSet<StmtId> {
    let mut zeros = HashSet::new();
    unset(b, flag, &mut zeros, 0);
    zeros
}

const MAX_DEPTH: u32 = 64;

/// `flag == 0` (a 0/1 value): record what it forces to 0.
fn unset(b: &[IRStatement], flag: StmtId, zeros: &mut HashSet<StmtId>, depth: u32) {
    if depth > MAX_DEPTH {
        return;
    }
    let Some(stmt) = b.get(flag as usize) else { return };
    let args = stmt.arguments.as_slice();
    match (&stmt.ir, args) {
        // a != 0 is 0 only when a == 0.
        (IR::NeI | IR::NeF, [x, y]) if is_zero(b, *y) => is_zero_value(b, *x, zeros, depth + 1),
        (IR::NeI | IR::NeF, [x, y]) if is_zero(b, *x) => is_zero_value(b, *y, zeros, depth + 1),
        // 0 < a with a >= 0 is 0 only when a == 0.
        (IR::LtI | IR::LtIBounded, [z, a]) if is_zero(b, *z) && non_negative(b, *a, 0) => {
            is_zero_value(b, *a, zeros, depth + 1)
        }
        (IR::GtI | IR::GtIBounded, [a, z]) if is_zero(b, *z) && non_negative(b, *a, 0) => {
            is_zero_value(b, *a, zeros, depth + 1)
        }
        // not(a == 0) is 0 only when a == 0.
        (IR::LogicalNot, [inner]) => {
            if let Some(eq) = b.get(*inner as usize) {
                match (&eq.ir, eq.arguments.as_slice()) {
                    (IR::EqI | IR::EqF, [x, y]) if is_zero(b, *y) => is_zero_value(b, *x, zeros, depth + 1),
                    (IR::EqI | IR::EqF, [x, y]) if is_zero(b, *x) => is_zero_value(b, *y, zeros, depth + 1),
                    _ => {}
                }
            }
        }
        (IR::BoolCast, [x]) => is_zero_value(b, *x, zeros, depth + 1),
        _ => {}
    }
}

/// `a == 0`: record what it forces to 0.
fn is_zero_value(b: &[IRStatement], a: StmtId, zeros: &mut HashSet<StmtId>, depth: u32) {
    if depth > MAX_DEPTH {
        return;
    }
    zeros.insert(a);
    let Some(stmt) = b.get(a as usize) else { return };
    match (&stmt.ir, stmt.arguments.as_slice()) {
        (IR::AddI, [s, t]) if non_negative(b, *s, 0) && non_negative(b, *t, 0) => {
            is_zero_value(b, *s, zeros, depth + 1);
            is_zero_value(b, *t, zeros, depth + 1);
        }
        _ if is_bit(&stmt.ir) => unset(b, a, zeros, depth + 1),
        _ => {}
    }
}

/// Whether `a >= 0` on every accepted witness, by structure.
fn non_negative(b: &[IRStatement], a: StmtId, depth: u32) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    let Some(stmt) = b.get(a as usize) else { return false };
    match (&stmt.ir, stmt.arguments.as_slice()) {
        (IR::ConstantInt { value }, _) => *value >= 0,
        (IR::AddI, [s, t]) => non_negative(b, *s, depth + 1) && non_negative(b, *t, depth + 1),
        (ir, _) => is_bit(ir),
    }
}

/// Operations whose result is a constrained 0/1 value.
fn is_bit(ir: &IR) -> bool {
    matches!(
        ir,
        IR::EqI
            | IR::NeI
            | IR::LtI
            | IR::LteI
            | IR::GtI
            | IR::GteI
            | IR::LtIBounded
            | IR::GtIBounded
            | IR::EqF
            | IR::NeF
            | IR::LtF
            | IR::LteF
            | IR::GtF
            | IR::GteF
            | IR::LogicalAnd
            | IR::LogicalOr
            | IR::LogicalNot
            | IR::BoolCast
            | IR::ConstantBool { .. }
    )
}

/// Whether `s` is the constant 0 (integer or float).
fn is_zero(b: &[IRStatement], s: StmtId) -> bool {
    matches!(
        b.get(s as usize).map(|st| &st.ir),
        Some(IR::ConstantInt { value: 0 })
    ) || matches!(b.get(s as usize).map(|st| &st.ir), Some(IR::ConstantFloat { value }) if *value == 0.0)
}
