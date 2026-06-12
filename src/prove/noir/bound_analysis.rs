//! Forward-dataflow `fits-u32` bound analysis over `IRGraph`.
//!
//! Walks IR statements in order and, for each result-producing statement,
//! records a tight upper bound on the (non-negative) value the result can
//! take. The Noir emitter consults this map to decide whether to bind a
//! cell as `u32` (saving a `BLACKBOX::RANGE` per `Field as u32` cast) or
//! fall back to `Field`.
//!
//! Conservative by construction:
//! - Anything we can't bound returns `None` (Field path).
//! - Arithmetic is bounded via `checked_add` / `checked_mul`; overflow
//!   collapses to `None`.
//! - Float ops, Q-function results, hashes, dynamic-NDArray reads, and
//!   any path with an unknown-bounded ancestor are unbounded.
//! - Compare / boolean results are bounded by 1.
//!
//! Returned bounds use the canonical statement id (`stmt.stmt_id`) as the
//! key — same key the interpreter uses to slot results into its `cells`
//! vector. The emitter receives the stmt_id via the new
//! `Synthesizer::on_stmt_begin` hook.

use std::collections::HashMap;

use crate::ir::IRGraph;
use crate::ir_defs::IR;

/// Map from `stmt_id` to an inclusive upper bound on the (non-negative)
/// integer value of that statement's result. Absence means "no provable
/// bound" — the emitter should default to `Field`.
pub type BoundMap = HashMap<u32, u64>;

/// Count of operations that *natively* lower to a u32/u128 RANGE check
/// in the Noir emit (`AssertU128Lt`, `LtIBounded`, `GtIBounded`,
/// `IndexInBox`). The Noir emitter uses this as a "is the u32 lookup
/// table going to be paid for anyway?" proxy: a high count means
/// eagerly promoting bool/add chains is free, a low count means the
/// table overhead would dominate.
///
/// The threshold lives in `NoirEmitter::new`. Below it, kernels stay
/// on the Field path (cf. 0503/0504/0505 with 1 intrinsic op each,
/// where promoting cost +3500 each; 0355/0356/0357 with a handful,
/// +707 each).
pub fn count_u32_intrinsic_ops(graph: &IRGraph) -> usize {
    graph
        .stmts
        .iter()
        .filter(|s| {
            matches!(
                &s.ir,
                IR::AssertU128Lt
                    | IR::AssertU128LtBit { .. }
                    | IR::LtIBounded
                    | IR::GtIBounded
                    | IR::IndexInBox { .. }
            )
        })
        .count()
}

const U32_MAX: u64 = (1u64 << 32) - 1;

/// Run the forward pass. Only bounds that provably fit `u32` are kept;
/// anything wider is treated as unbounded (returns None implicitly).
pub fn run(graph: &IRGraph) -> BoundMap {
    let mut bounds: BoundMap = HashMap::new();

    for stmt in &graph.stmts {
        let id = stmt.stmt_id;
        let arg_bounds: Vec<Option<u64>> = stmt
            .arguments
            .iter()
            .map(|a| bounds.get(a).copied())
            .collect();

        let result: Option<u64> = match &stmt.ir {
            // ── Constants ─────────────────────────────────────────────
            IR::ConstantInt { value } => {
                if *value >= 0 {
                    Some(*value as u64)
                } else {
                    None
                }
            }
            IR::ConstantBool { .. } => Some(1),
            IR::ConstantFloat { .. } | IR::ConstantStr { .. } => None,

            // ── Integer arithmetic ────────────────────────────────────
            IR::AddI => arg_bounds.get(0).copied().flatten().and_then(|a| {
                arg_bounds
                    .get(1)
                    .copied()
                    .flatten()
                    .and_then(|b| a.checked_add(b))
            }),
            IR::SubI => {
                // result ∈ [-b_max, a_max]. We only return a non-negative
                // bound; sub can produce negatives, so we have to give up
                // unless we can prove b == 0 (rare). Be conservative.
                None
            }
            IR::MulI => arg_bounds.get(0).copied().flatten().and_then(|a| {
                arg_bounds
                    .get(1)
                    .copied()
                    .flatten()
                    .and_then(|b| a.checked_mul(b))
            }),
            IR::DivI | IR::FloorDivI => arg_bounds.get(0).copied().flatten(),
            IR::ModI => {
                // result < |b|. If b is bounded, result <= b - 1; if not,
                // fall back to a's bound.
                match arg_bounds.get(1).copied().flatten() {
                    Some(b) if b > 0 => Some(b - 1),
                    _ => arg_bounds.get(0).copied().flatten(),
                }
            }
            IR::PowI | IR::AbsI | IR::SignI | IR::InvI => None,

            // ── Integer bitwise — narrow bound to min(operand bounds) ─
            IR::BitAndI => match (
                arg_bounds.get(0).copied().flatten(),
                arg_bounds.get(1).copied().flatten(),
            ) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                _ => None,
            },
            IR::BitOrI | IR::BitXorI | IR::ShlI | IR::ShrI | IR::BitNotI => None,

            // ── Float ops — Field path ────────────────────────────────
            IR::AddF
            | IR::SumF { .. }
            | IR::SubF
            | IR::MulF
            | IR::ProductSumF { .. }
            | IR::DivF
            | IR::FloorDivF
            | IR::ModF
            | IR::PowF
            | IR::PowFI
            | IR::AbsF
            | IR::SignF
            | IR::SinF
            | IR::SinHF
            | IR::CosF
            | IR::CosHF
            | IR::TanF
            | IR::TanHF
            | IR::SqrtF
            | IR::ExpF
            | IR::LogF
            | IR::ArcSinF
            | IR::ArcCosF
            | IR::ArcTan2F => None,

            // ── Integer comparisons (result ∈ {0,1}) ──────────────────
            IR::EqI
            | IR::NeI
            | IR::LtI
            | IR::LteI
            | IR::GtI
            | IR::GteI
            | IR::LtIBounded
            | IR::GtIBounded => Some(1),
            IR::IndexInBox { .. } => Some(1),

            // ── Float comparisons ─────────────────────────────────────
            IR::EqF | IR::NeF | IR::LtF | IR::LteF | IR::GtF | IR::GteF => Some(1),

            // ── Boolean logic ─────────────────────────────────────────
            IR::LogicalAnd | IR::LogicalOr | IR::LogicalNot => Some(1),

            // ── Selection — max of branches (cond is args[0]) ─────────
            IR::SelectI | IR::SelectB | IR::SelectBitI => match (
                arg_bounds.get(1).copied().flatten(),
                arg_bounds.get(2).copied().flatten(),
            ) {
                (Some(t), Some(f)) => Some(t.max(f)),
                _ => None,
            },
            IR::SelectManyI { len, lane } => {
                let len = *len as usize;
                let lane = *lane as usize;
                match (
                    arg_bounds.get(1 + lane).copied().flatten(),
                    arg_bounds.get(1 + len + lane).copied().flatten(),
                ) {
                    (Some(t), Some(f)) => Some(t.max(f)),
                    _ => None,
                }
            }
            IR::SelectF | IR::SelectBitF | IR::SelectManyF { .. } => None,

            // ── Casts ─────────────────────────────────────────────────
            IR::BoolCast | IR::AssertU128LtBit { .. } => Some(1),
            IR::IntCast => arg_bounds.get(0).copied().flatten(),
            IR::FloatCast => None,

            // ── I/O — unbounded (witness values, could be anything) ───
            IR::ReadInteger { .. }
            | IR::ReadFloat { .. }
            | IR::ReadHash { .. }
            | IR::ReadExternalResult { .. } => None,

            // ── Inert / structural — no result on the cell stack ──────
            IR::AddStr
            | IR::StrI
            | IR::StrF
            | IR::Print
            | IR::Assert
            | IR::AssertBit
            | IR::AssertNonNegI
            | IR::AssertU128Lt
            | IR::ExposePublicI
            | IR::ExposePublicF
            | IR::StructuralPredicate { .. }
            | IR::ScalarPrecondition { .. }
            | IR::AllocateMemory { .. }
            | IR::WriteMemory { .. }
            | IR::MemoryTraceEmit { .. }
            | IR::MemoryTraceSeal
            | IR::AllocateDynamicNDArrayMeta { .. }
            | IR::WitnessDynamicNDArrayMeta { .. }
            | IR::AssertDynamicNDArrayMeta { .. }
            | IR::DynamicNDArraySetItem { .. }
            | IR::InvokeExternal { .. }
            | IR::ExportExternalI { .. }
            | IR::ExportExternalF { .. } => None,

            // Memory / dynamic-NDArray reads return witness values — unknown bound.
            IR::ReadMemory { .. } | IR::DynamicNDArrayGetItem { .. } => None,

            // ── Hash ──────────────────────────────────────────────────
            IR::PoseidonHash | IR::EqHash => None,
        };

        // Provenance fallback / tightening: the ProvenanceTable carries
        // structural intervals the local dataflow can't see — segment
        // hulls on ReadMemory outputs, the SelectI min/max recognizer,
        // AssertU128Lt / AssertNonNegI hooks (the assert constraint is
        // itself part of the emitted circuit, so the bound holds for any
        // accepted witness). Intervals are deposited at IR-build time
        // from structure, never from witness evaluation, so consuming
        // them here cannot specialize the circuit; a wrong bound could
        // only over-constrain (Noir u32 ops are overflow-checked), never
        // accept a bad witness.
        let prov_bound = graph
            .provenance
            .interval_of(stmt.value_id)
            .filter(|iv| iv.lo >= 0)
            .and_then(|iv| u64::try_from(iv.hi).ok());
        let combined = match (result, prov_bound) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };

        if let Some(b) = combined {
            if b <= U32_MAX {
                bounds.insert(id, b);
            }
        }
    }

    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_yields_empty_map() {
        let graph = IRGraph { stmts: Vec::new() };
        assert!(run(&graph).is_empty());
    }
}
