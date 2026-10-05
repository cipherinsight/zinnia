//! Collapse redundant `AssertU128Lt(base + stride_k, hi)` chains.
//!
//! Pattern: an access with a runtime base and several compile-time offsets
//! (for example `a[i*COLS + col]` unrolled over `i`) discharges one
//! `AssertU128Lt(base + stride_k, hi)` per offset, all with the same
//! runtime `base` and the same `hi`. `AssertU128Lt(x, hi)` means
//! `0 <= x < hi`, so assert `k` says `base` lies in
//! `[-stride_k, hi - stride_k)`.
//!
//! **Soundness.** The conjunction of the group is the intersection of those
//! windows, `[-stride_min, hi - stride_max)`. The member with the largest
//! stride gives the upper end and the member with the smallest stride gives
//! the lower end; every other member's window contains the intersection
//! (for `stride_min <= s <= stride_max`, `base + s` then lies in
//! `[s - stride_min, hi - stride_max + s) ⊆ [0, hi)`). So keeping the two
//! extreme members and dropping the rest preserves the conjunction exactly.
//! The values are field elements; the argument holds because every window
//! is a contiguous arc of length `hi` far smaller than the field, and when
//! `stride_max - stride_min >= hi` the two kept windows are disjoint, so the
//! kept asserts are unsatisfiable just like the original group.
//!
//! An earlier version kept only the largest-stride member, which drops the
//! `base >= -stride_min` half and admits `base` down to `-stride_max`.

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

/// The smallest- and largest-stride members of one `(base, hi)` group.
#[derive(Clone, Copy)]
struct GroupExtremes {
    min: (i64, StmtId),
    max: (i64, StmtId),
}

impl IRPass for AssertSubsumption {
    fn exec(&self, mut ir_graph: IRGraph) -> IRGraph {
        // Phase 1: for each (base, hi) group of `AssertU128Lt` stmts, find
        // the smallest- and largest-stride members (see the module docs for
        // why these two imply the rest). Every other member is redundant.
        let mut groups: HashMap<(StmtId, StmtId), GroupExtremes> = HashMap::new();
        let mut members: Vec<StmtId> = Vec::new();
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
            let entry = (form.stride, stmt.stmt_id);
            members.push(stmt.stmt_id);
            groups
                .entry((form.base, hi_id))
                .and_modify(|g| {
                    if entry.0 < g.min.0 {
                        g.min = entry;
                    }
                    if entry.0 > g.max.0 {
                        g.max = entry;
                    }
                })
                .or_insert(GroupExtremes {
                    min: entry,
                    max: entry,
                });
        }
        let kept: std::collections::HashSet<StmtId> = groups
            .values()
            .flat_map(|g| [g.min.1, g.max.1])
            .collect();
        let redundant: Vec<StmtId> = members
            .into_iter()
            .filter(|id| !kept.contains(id))
            .collect();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::{CircuitInputs, ResolvedWitness};
    use crate::prove::mock::MockProverBackend;
    use crate::prove::traits::ProverBackend;
    use crate::prove::types::ProvingParams;

    /// `base = lo - 3` (a runtime value, not a constant), then
    /// `AssertU128Lt(base + s, 6)` for each stride `s`.
    fn strided_asserts(lo: i64, strides: &[i64]) -> IRGraph {
        let mut stmts: Vec<(IR, Vec<StmtId>)> = vec![
            (IR::ConstantInt { value: lo }, vec![]),
            (IR::ConstantInt { value: 3 }, vec![]),
            (IR::SubI, vec![0, 1]), // base, stmt 2
            (IR::ConstantInt { value: 6 }, vec![]), // hi, stmt 3
        ];
        for &s in strides {
            let c = stmts.len() as StmtId;
            stmts.push((IR::ConstantInt { value: s }, vec![]));
            stmts.push((IR::AddI, vec![2, c]));
            stmts.push((IR::AssertU128Lt, vec![c + 1, 3]));
        }
        let ir_stmts = stmts
            .into_iter()
            .enumerate()
            .map(|(i, (ir, args))| {
                IRStatement::new(i as StmtId, crate::types::ValueId::next(), ir, args, vec![], None)
            })
            .collect();
        IRGraph::new(ir_stmts)
    }

    fn kept_strides(graph: &IRGraph) -> Vec<i64> {
        graph
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::AssertU128Lt))
            .map(|s| normalise_idx(graph, s.arguments[0]).unwrap().stride)
            .collect()
    }

    fn mock_satisfied(graph: &IRGraph) -> bool {
        let params = ProvingParams::default();
        let witness = ResolvedWitness::new(CircuitInputs::new(), params.precision_bits);
        let artifact = MockProverBackend.prove(graph, &witness, &params).unwrap();
        artifact.proof_bytes == "mock_satisfied"
    }

    #[test]
    fn keeps_smallest_and_largest_stride() {
        let out = AssertSubsumption.exec(strided_asserts(3, &[2, 0, 5, 3]));
        let mut kept = kept_strides(&out);
        kept.sort();
        assert_eq!(kept, vec![0, 5]);
    }

    #[test]
    fn negative_base_still_rejected_after_subsumption() {
        // base = -1: `base + 5 = 4 < 6` holds, but `base + 0 = -1` violates
        // the lower bound. The largest-stride assert alone would accept it.
        let graph = strided_asserts(2, &[0, 2, 5]);
        assert!(!mock_satisfied(&graph));
        assert!(!mock_satisfied(&AssertSubsumption.exec(graph)));
    }

    #[test]
    fn honest_base_accepted_after_subsumption() {
        // base = 0: every `base + s` with s in {0, 2, 5} lies in [0, 6).
        let out = AssertSubsumption.exec(strided_asserts(3, &[0, 2, 5]));
        assert!(mock_satisfied(&out));
    }
}
