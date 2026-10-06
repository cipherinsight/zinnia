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
use crate::types::{StmtId, Value};

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

/// `c[p] = values` for `p` a recorded permutation of `0..n`
/// (`IRBuilder::record_permutation`) and `n` scalar values: the cells
/// `c[0..n]` as placeholders. `None` when `p` is not known to be a
/// permutation.
pub fn permutation_scatter_lanes(b: &mut IRBuilder, p: &[Value], values: &[Value]) -> Option<Vec<Value>> {
    let n = p.len();
    if n == 0 || values.len() != n || !b.is_known_permutation(p) || !values.iter().all(Value::is_number) {
        return None;
    }
    let operands: Vec<Value> = p.iter().chain(values).map(|v| b.ensure_ptr(v)).collect();
    let shared: Arc<[Value]> = operands.into();
    Some(
        (0..n)
            .map(|j| {
                let op = ArrayOp { kind: OpKind::PermutationScatter, shared: 2 * n as u32, output: 0, position: Some(j as u32) };
                b.ir_array_op(op, &shared, &[])
            })
            .collect(),
    )
}

/// `cells[index]` as a placeholder, when `index` has a known one-hot (a
/// scan's result, or a cell of a permutation scatter of `k + c`, plus a
/// constant); `None` otherwise. The caller has checked `0 <= index <
/// cells.len()`.
pub fn element_lane(b: &mut IRBuilder, cells: &[Value], index: &Value) -> Option<Value> {
    if cells.is_empty() || !has_encoding(b, index) {
        return None;
    }
    let shared: Arc<[Value]> = cells.iter().map(|c| b.ensure_ptr(c)).collect();
    let op = ArrayOp { kind: OpKind::Element, shared: shared.len() as u32, output: 0, position: None };
    Some(b.ir_array_op(op, &shared, std::slice::from_ref(index)))
}

/// Whether `v` is, plus a constant, a lane whose one-hot a construction can
/// provide: a scan's result, or a cell of a permutation scatter of the
/// values `k + c` (`encoded_values`).
pub fn has_encoding(b: &IRBuilder, v: &Value) -> bool {
    let Some((lane, _)) = lane_source(b, v) else { return false };
    let stmt = &b.stmts[lane as usize];
    match &stmt.ir {
        IR::ArrayOp(op) if op.kind == OpKind::PermutationScatter => {
            let values: Vec<Option<i64>> = stmt.arguments[stmt.arguments.len() / 2..]
                .iter()
                .map(|&a| match b.stmts[a as usize].ir {
                    IR::ConstantInt { value } => Some(value),
                    _ => None,
                })
                .collect();
            encoded_values(&values).is_some()
        }
        IR::ArrayOp(op) => ScanKind::of(op.kind).is_some(),
        _ => false,
    }
}

/// `c` when `values[k] = k + c` for every `k`: the scattered cells then
/// have the one-hot `[c[j] == k + c] = [p[k] == j]`.
pub fn encoded_values(values: &[Option<i64>]) -> Option<i64> {
    let c = values.first().copied()??;
    values.iter().enumerate().all(|(k, v)| *v == Some(k as i64 + c)).then_some(c)
}

/// `true` when `stop - start >= 0` wherever both are used: `start` is the
/// first set position of some flags plus `c1` and `stop` the last set
/// position of the same flags plus `c2 >= c1`. Both scans require a set
/// flag (asserted by the frontend where they are used), and then the first
/// set position is at most the last. A slice's extent `max(0, stop - start)`
/// is then `stop - start`.
pub fn scan_bounds_ordered(b: &IRBuilder, start: &Value, stop: &Value) -> bool {
    let (Some((first, c1)), Some((last, c2))) = (lane_source(b, start), lane_source(b, stop)) else {
        return false;
    };
    let lane = |s: StmtId| match &b.stmts[s as usize].ir {
        IR::ArrayOp(op) => Some((op.kind, b.stmts[s as usize].arguments.clone())),
        _ => None,
    };
    match (lane(first), lane(last)) {
        (Some((OpKind::FirstSet, flags)), Some((OpKind::LastSet, same))) => flags == same && c2 >= c1,
        _ => false,
    }
}

/// `(lane, c)` when `v` is a gather placeholder's lane `lane` plus the
/// constant `c` (through `+`/`-` of constants).
fn lane_source(b: &IRBuilder, v: &Value) -> Option<(StmtId, i64)> {
    let mut stmt = v.stmt_id()?;
    let mut offset = 0i64;
    for _ in 0..8 {
        let s = b.stmts.get(stmt as usize)?;
        let constant = |id: StmtId| match b.stmts.get(id as usize).map(|s| &s.ir) {
            Some(IR::ConstantInt { value }) => Some(*value),
            _ => None,
        };
        (stmt, offset) = match (&s.ir, s.arguments.as_slice()) {
            (IR::ArrayOp(_), _) => return Some((stmt, offset)),
            (IR::AddI, [x, c]) if constant(*c).is_some() => (*x, offset + constant(*c)?),
            (IR::AddI, [c, x]) if constant(*c).is_some() => (*x, offset + constant(*c)?),
            (IR::SubI, [x, c]) if constant(*c).is_some() => (*x, offset - constant(*c)?),
            _ => return None,
        };
    }
    None
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
