//! Collapse redundant `AssertU128Lt(stride_k + base, hi)` chains.
//!
//! Pattern: when Zinnia lowers a runtime-indexed access across an unrolled
//! loop (e.g. `a[i*COLS + col_idx]` for `i in 0..ROWS`), the IR builder
//! emits one `discharge_index_in_range(addr_i, axis_len)` per row. Each
//! `addr_i` has the form `stride_i + base` for a different compile-time
//! `stride_i` and the same runtime `base = col_idx`. The result is N
//! distinct asserts of the form `(stride_i + base) < hi`.
//!
//! All N asserts are logically subsumed by the one with the **largest**
//! `stride_i`: if `(stride_max + base) < hi` then `(stride_k + base) <
//! hi` follows for every smaller `stride_k`. Keeping only the tightest
//! assert drops the (N - 1) redundant ones and the per-cmp u128 cast +
//! lookup-table hit each carried (~50 ACIR gates on UltraHonk).
//!
//! Live cases (DS-1000): 0345 / 0346 / 0347 (cumsum/product over
//! runtime-indexed column/row), 0363 / 0364 / 0365 (`np.insert` at a
//! runtime position) — ~1,300 gates of residual ZN > HW excess.

use std::collections::HashMap;

use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::types::StmtId;

use super::IRPass;

pub struct AssertSubsumption;

/// Normalised form of an `AssertU128Lt(idx, hi)` operand: a runtime base
/// plus a non-negative compile-time stride. `base` is the StmtId of the
/// runtime value; `stride` is 0 if `idx` is the bare base.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct AssertIdxForm {
    base: StmtId,
    stride: i64,
}

/// Try to express `idx_stmt` as `stride + base_stmt` where `stride` is a
/// non-negative i64 compile-time constant. Returns `None` if the form
/// doesn't match (e.g. negative stride, both operands runtime, idx is a
/// negative constant).
fn normalise_idx(ir_graph: &IRGraph, idx_stmt_id: StmtId) -> Option<AssertIdxForm> {
    let stmt = &ir_graph.stmts[idx_stmt_id as usize];
    // `idx = AddI(a, b)` where one of {a, b} is `ConstantInt { value: c >= 0 }`.
    if matches!(stmt.ir, IR::AddI) && stmt.arguments.len() == 2 {
        let a_id = stmt.arguments[0];
        let b_id = stmt.arguments[1];
        let a_const = const_int(ir_graph, a_id);
        let b_const = const_int(ir_graph, b_id);
        match (a_const, b_const) {
            (Some(c), None) if c >= 0 => {
                return Some(AssertIdxForm {
                    base: b_id,
                    stride: c,
                })
            }
            (None, Some(c)) if c >= 0 => {
                return Some(AssertIdxForm {
                    base: a_id,
                    stride: c,
                })
            }
            _ => {}
        }
    }
    // Bare base: `idx` is the runtime value itself, stride 0.
    if !matches!(stmt.ir, IR::ConstantInt { .. }) {
        return Some(AssertIdxForm {
            base: idx_stmt_id,
            stride: 0,
        });
    }
    None
}

fn const_int(ir_graph: &IRGraph, stmt_id: StmtId) -> Option<i64> {
    match ir_graph.stmts[stmt_id as usize].ir {
        IR::ConstantInt { value } => Some(value),
        _ => None,
    }
}

impl IRPass for AssertSubsumption {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
        // Phase 1: for each (base, hi) group of `AssertU128Lt` stmts,
        // find the one with the maximum stride. That assert subsumes
        // the rest. Mark the others for removal.
        let mut groups: HashMap<(StmtId, StmtId), (i64, StmtId)> = HashMap::new();
        let mut redundant: Vec<StmtId> = Vec::new();
        for stmt in &ir_graph.stmts {
            if !matches!(stmt.ir, IR::AssertU128Lt) || stmt.arguments.len() != 2 {
                continue;
            }
            let idx_id = stmt.arguments[0];
            let hi_id = stmt.arguments[1];
            let form = match normalise_idx(&ir_graph, idx_id) {
                Some(f) => f,
                None => continue,
            };
            let key = (form.base, hi_id);
            match groups.get(&key).copied() {
                None => {
                    groups.insert(key, (form.stride, stmt.stmt_id));
                }
                Some((best_stride, best_id)) => {
                    if form.stride > best_stride {
                        // The new assert is tighter; the previous one is
                        // now subsumed.
                        redundant.push(best_id);
                        groups.insert(key, (form.stride, stmt.stmt_id));
                    } else {
                        // The existing assert is at least as tight; this
                        // new one is redundant.
                        redundant.push(stmt.stmt_id);
                    }
                }
            }
        }

        if redundant.is_empty() {
            return ir_graph;
        }

        // Phase 2: rebuild the graph with redundant asserts dropped.
        // Asserts have no `value_id` consumers (return Value::None), so
        // we never need to remap downstream references — we only need to
        // re-pack stmt_ids so they stay equal to vec indices.
        let drop: std::collections::HashSet<StmtId> = redundant.into_iter().collect();
        let mut new_id_for_old: HashMap<StmtId, StmtId> = HashMap::new();
        let mut new_stmts: Vec<IRStatement> = Vec::with_capacity(ir_graph.stmts.len());
        for stmt in &ir_graph.stmts {
            if drop.contains(&stmt.stmt_id) {
                continue;
            }
            let new_arguments: Vec<StmtId> = stmt
                .arguments
                .iter()
                .map(|&arg| new_id_for_old[&arg])
                .collect();
            let new_stmt_id = new_stmts.len() as StmtId;
            new_stmts.push(IRStatement::new(
                new_stmt_id,
                stmt.value_id,
                stmt.ir.clone(),
                new_arguments,
                Vec::new(),
                stmt.dbg.clone(),
            ));
            new_id_for_old.insert(stmt.stmt_id, new_stmt_id);
        }
        let mut new_graph = IRGraph::new(new_stmts);
        new_graph.resolver_mut().on_ir_mutated(&[]);
        new_graph
    }
}
