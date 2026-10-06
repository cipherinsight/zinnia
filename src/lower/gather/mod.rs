//! Gather family of the array-operator lowering layer: reads of an array at
//! runtime indices (crops and slices with runtime bounds, element reads) and
//! the index computations that feed them (the first / last set position of
//! a flag vector: `min` / `max` of `flatnonzero` and `argwhere`, `argmax`
//! of booleans).
//!
//! The frontend emits placeholders through the functions below
//! ([`scan_lane`], [`window_lanes`], [`element_lane`]); after dead-code
//! elimination the layer lowers them by the constructions of
//! [`constructions`], which emit the generators:
//!
//! * [`scan`]: a flag position by running products (with its one-hot and
//!   thermometer, [`encoding`]) or by a select chain;
//! * [`window`]: a slice with runtime bounds as a shift by a shared one-hot
//!   per axis;
//! * [`element`]: a read at a scan's result as an inner product with the
//!   scan's one-hot;
//! * [`vanishing`]: which cells a flag being 0 forces to 0 (lets a window
//!   skip the mask on padding that is 0 anyway).
//!
//! Costs per target are in [`cost`]; what passes know about a lane before
//! lowering in [`semantics`].

pub mod constructions;
pub mod cost;
pub mod element;
pub mod encoding;
pub mod scan;
pub mod semantics;
pub mod vanishing;
pub mod window;

#[cfg(test)]
mod tests;

use std::sync::Arc;

pub use constructions::constructions;
pub use scan::ScanKind;
pub use semantics::GatherSemantics;

use crate::builder::IRBuilder;
use crate::ir_defs::IR;
use crate::lower::{ArrayOp, OpKind};
use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};
use crate::types::Value;

/// The `kind` position of the 0/1 values `flags` (not empty): a placeholder
/// the layer lowers by `scan`. The fact `0 <= v < n` is recorded (every
/// construction respects it). For `First` and `Last` the caller has
/// asserted that some flag is set.
pub fn scan_lane(b: &mut IRBuilder, kind: ScanKind, flags: &[Value]) -> Value {
    let n = flags.len();
    assert!(n > 0, "a scan over no flags");
    let shared: Arc<[Value]> = flags.iter().map(|f| b.ensure_ptr(f)).collect();
    let op = ArrayOp { kind: kind.op_kind(), shared: n as u32, output: 0, position: Some(0) };
    let lane = b.ir_array_op(op, &shared, &[]);
    record_range_fact(b, &lane, 0, n as i64 - 1);
    lane
}

/// The window `x[start_0:stop_0, start_1:stop_1, ...]` (step 1 on every
/// axis, each `start` in `[0, n]`) of the array whose row-major `cells`
/// have `shape`, as the cells of a buffer of `shape` with the window at its
/// top-left and 0 elsewhere: one placeholder per cell.
pub fn window_lanes(b: &mut IRBuilder, cells: &[Value], shape: &[usize], bounds: &[(Value, Value)]) -> Option<Vec<Value>> {
    if shape.is_empty() || shape.len() > 3 || bounds.len() != shape.len() || cells.is_empty() {
        return None;
    }
    let mut dims = [0u32; 3];
    for (d, &n) in dims.iter_mut().zip(shape) {
        *d = u32::try_from(n).ok()?;
    }
    let kind = OpKind::Window { rank: shape.len() as u8, dims };
    let mut operands: Vec<Value> = cells.iter().map(|c| b.ensure_ptr(c)).collect();
    for (start, stop) in bounds {
        operands.push(b.ensure_ptr(start));
        operands.push(b.ensure_ptr(stop));
    }
    let shared: Arc<[Value]> = operands.into();
    Some(
        (0..cells.len())
            .map(|p| {
                let op = ArrayOp { kind, shared: shared.len() as u32, output: 0, position: Some(p as u32) };
                b.ir_array_op(op, &shared, &[])
            })
            .collect(),
    )
}

/// `cells[index]` as a placeholder, when `index` is a scan's result plus a
/// constant (the layer can then read it with the scan's one-hot); `None`
/// otherwise. The caller has checked `0 <= index < cells.len()`.
pub fn element_lane(b: &mut IRBuilder, cells: &[Value], index: &Value) -> Option<Value> {
    if cells.is_empty() || !is_scan_result(b, index) {
        return None;
    }
    let shared: Arc<[Value]> = cells.iter().map(|c| b.ensure_ptr(c)).collect();
    let op = ArrayOp { kind: OpKind::Element, shared: shared.len() as u32, output: 0, position: None };
    Some(b.ir_array_op(op, &shared, std::slice::from_ref(index)))
}

/// Whether `v` is a scan placeholder's lane plus a constant.
pub fn is_scan_result(b: &IRBuilder, v: &Value) -> bool {
    let Some(mut stmt) = v.stmt_id() else { return false };
    for _ in 0..8 {
        let Some(s) = b.stmts.get(stmt as usize) else { return false };
        let constant = |id: u32| matches!(b.stmts.get(id as usize).map(|s| &s.ir), Some(IR::ConstantInt { .. }));
        stmt = match (&s.ir, s.arguments.as_slice()) {
            (IR::ArrayOp(op), _) => return ScanKind::of(op.kind).is_some(),
            (IR::AddI | IR::SubI, [x, c]) if constant(*c) => *x,
            (IR::AddI, [c, x]) if constant(*c) => *x,
            _ => return false,
        };
    }
    false
}

/// The fact `lo <= v <= hi`, which every construction's result satisfies.
fn record_range_fact(b: &mut IRBuilder, v: &Value, lo: i64, hi: i64) {
    let Some(vid) = v.value_id() else { return };
    let var = || Box::new(ContractTerm::Var(ContractVar::Value(vid)));
    let fact = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp { op: CmpOp::Ge, lhs: var(), rhs: Box::new(ContractTerm::LitInt(lo)) },
            ContractTerm::Cmp { op: CmpOp::Le, lhs: var(), rhs: Box::new(ContractTerm::LitInt(hi)) },
        ],
    };
    b.facts.insert_for(vid, fact);
}
