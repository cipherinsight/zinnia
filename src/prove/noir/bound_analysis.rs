//! Forward-dataflow bound analysis over `IRGraph`.
//!
//! Walks IR statements in order and, for each result-producing statement,
//! records a tight upper bound (below 2^32) on the (non-negative) value the
//! result can take. The Noir emitter reads these bounds as intervals
//! (`NoirEmitter::ival`): they choose the width of a comparison's range
//! check, the slots a dynamic gather enumerates (whose one-hot membership
//! assert enforces them), and which cells are 0/1.
//!
//! Conservative by construction:
//! - Anything we can't bound returns `None`.
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
            // `DivI` is field division (a * b^-1): no integer bound.
            IR::DivI => None,
            // a // b with a in [0, A] and a bounded (so non-negative) divisor
            // lies in [0, A] (a zero divisor gives 0). A divisor that may be
            // negative gives a negative quotient: no bound.
            IR::FloorDivI => match (
                arg_bounds.get(0).copied().flatten(),
                arg_bounds.get(1).copied().flatten(),
            ) {
                (Some(a), Some(_)) => Some(a),
                _ => None,
            },
            // The producer promises non-negative operands, and the Noir
            // lowering (`q_floor_div_bounded`) enforces it: q <= a.
            IR::FloorDivIPinned { .. } => arg_bounds.get(0).copied().flatten(),
            // a % b lies in [0, b - 1] for a bounded (non-negative) divisor b
            // (0 for b == 0); a negative divisor gives a non-positive
            // remainder, so an unbounded divisor gives no bound.
            IR::ModI => match arg_bounds.get(1).copied().flatten() {
                Some(b) if b > 0 => Some(b - 1),
                _ => None,
            },
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

            // ── Prover hints — any value the prover chooses ───────────
            IR::Hint { .. } | IR::ArrayOp(_) => None,
            IR::KeyLess { .. } | IR::PackedKeyLess { .. } => Some(1),
            IR::PackKey { .. } | IR::PackedKeyIndex { .. } => None,
        };

        // Provenance fallback / tightening: the ProvenanceTable carries
        // structural intervals the local dataflow can't see — segment
        // hulls on ReadMemory outputs, the SelectI min/max recognizer, and
        // the intervals AssertU128Lt / AssertNonNegI establish. The Noir
        // lowering of those asserts consults no bound (see
        // `assert_u128_lt`), so the facts hold for every accepted witness.
        // Intervals are deposited at IR-build time from structure, never
        // from witness values, so they cannot specialize the circuit. The
        // emitter uses a bound only to choose a cheaper form whose own
        // checks enforce it (a comparison's range-check width, a gather's
        // membership assert), so a wrong bound fails the proof rather than
        // accepting a wrong value; 0/1 facts come from lowerings that
        // guarantee them.
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
    use crate::ir::IRStatement;
    use crate::provenance::Interval;
    use crate::types::ValueId;

    /// Build a statement whose `stmt_id == index` (the graph invariant) and
    /// whose `value_id == ValueId(index)`, so the `bounds` map (keyed by
    /// stmt_id, a u32) and `interval_of` (keyed by value_id) line up trivially
    /// in these tests. `arguments` are stmt_ids; `arg_values` are the matching
    /// ValueIds.
    fn stmt(index: u32, ir: IR, args: Vec<u32>) -> IRStatement {
        let arg_values = args.iter().map(|a| ValueId(*a as u64)).collect();
        IRStatement::new(index, ValueId(index as u64), ir, args, arg_values, None)
    }

    fn graph_of(stmts: Vec<IRStatement>) -> IRGraph {
        IRGraph::new(stmts)
    }

    #[test]
    fn empty_graph_yields_empty_map() {
        let graph = IRGraph::empty_for_tests();
        assert!(run(&graph).is_empty());
    }

    #[test]
    fn add_i_of_small_constants_is_bounded_by_their_sum() {
        // 100 + 200 = 300, well under U32_MAX → bound recorded.
        let graph = graph_of(vec![
            stmt(0, IR::ConstantInt { value: 100 }, vec![]),
            stmt(1, IR::ConstantInt { value: 200 }, vec![]),
            stmt(2, IR::AddI, vec![0, 1]),
        ]);
        let bounds = run(&graph);
        assert_eq!(bounds.get(&2).copied(), Some(300));
    }

    #[test]
    fn add_i_bound_exceeding_u32_collapses_to_none() {
        // Both operands fit u32 individually, but their sum exceeds U32_MAX
        // (4_294_967_295), so AddI can no longer be proven to fit u32 and is
        // dropped from the bound map (the emitter falls back to the Field
        // path). The operand constants themselves remain bounded.
        let big = 3_000_000_000i64; // < U32_MAX, > U32_MAX/2
        let graph = graph_of(vec![
            stmt(0, IR::ConstantInt { value: big }, vec![]),
            stmt(1, IR::ConstantInt { value: big }, vec![]),
            stmt(2, IR::AddI, vec![0, 1]),
        ]);
        let bounds = run(&graph);
        assert_eq!(bounds.get(&0).copied(), Some(big as u64));
        assert_eq!(bounds.get(&1).copied(), Some(big as u64));
        // 6_000_000_000 > U32_MAX → no bound kept.
        assert_eq!(bounds.get(&2).copied(), None);
    }

    #[test]
    fn mul_i_checked_overflow_collapses_to_none() {
        // u64 checked_mul genuinely overflows here: (1<<40) * (1<<40) = 1<<80.
        // The dataflow's checked_mul returns None, so even before the U32
        // gate the product is unbounded.
        let huge = 1i64 << 40;
        let graph = graph_of(vec![
            stmt(0, IR::ConstantInt { value: huge }, vec![]),
            stmt(1, IR::ConstantInt { value: huge }, vec![]),
            stmt(2, IR::MulI, vec![0, 1]),
        ]);
        let bounds = run(&graph);
        // Operands themselves exceed U32_MAX so they are also dropped.
        assert_eq!(bounds.get(&2).copied(), None);
    }

    #[test]
    fn division_by_a_possibly_negative_divisor_is_unbounded() {
        // 7 // y with y unbounded can be negative (y = -2 gives -4, as in
        // Python and the kernel): the dividend's bound must not be copied to
        // the quotient. Field division (DivI) has no bound.
        let graph = graph_of(vec![
            stmt(0, IR::ConstantInt { value: 7 }, vec![]),
            stmt(
                1,
                IR::ReadInteger {
                    path: crate::circuit_input::InputPath::new("y", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            stmt(2, IR::FloorDivI, vec![0, 1]),
            stmt(3, IR::DivI, vec![0, 1]),
            stmt(4, IR::ModI, vec![0, 1]),
            stmt(5, IR::ConstantInt { value: 4 }, vec![]),
            stmt(6, IR::FloorDivI, vec![0, 5]),
            stmt(7, IR::DivI, vec![0, 5]),
            stmt(8, IR::ModI, vec![0, 5]),
        ]);
        let bounds = run(&graph);
        assert_eq!(bounds.get(&2), None);
        assert_eq!(bounds.get(&3), None);
        assert_eq!(bounds.get(&4), None);
        assert_eq!(bounds.get(&6).copied(), Some(7));
        assert_eq!(bounds.get(&7), None);
        assert_eq!(bounds.get(&8).copied(), Some(3));
    }

    #[test]
    fn provenance_interval_min_merges_with_local_bound() {
        // A ModI by a large constant gives a loose local bound (modulus-1),
        // but the ProvenanceTable carries a tighter structural interval on
        // the result value. The two must min-merge to the tighter bound.
        //
        //   v0 = read (unbounded witness)
        //   v1 = const 1000
        //   v2 = v0 % v1     → local bound 999
        // Provenance says v2 ∈ [0, 7]  → merged bound = min(999, 7) = 7.
        let mut graph = graph_of(vec![
            stmt(0, IR::ConstantFloat { value: 0.0 }, vec![]),
            stmt(1, IR::ConstantInt { value: 1000 }, vec![]),
            stmt(2, IR::ModI, vec![0, 1]),
        ]);
        // Statement 2's value_id is 2 (set by `stmt`).
        graph
            .provenance
            .set_interval(ValueId(2), Interval { lo: 0, hi: 7 });

        let bounds = run(&graph);
        assert_eq!(bounds.get(&2).copied(), Some(7));

        // Sanity: without the provenance interval the bound is the loose 999.
        let plain = graph_of(vec![
            stmt(0, IR::ConstantFloat { value: 0.0 }, vec![]),
            stmt(1, IR::ConstantInt { value: 1000 }, vec![]),
            stmt(2, IR::ModI, vec![0, 1]),
        ]);
        assert_eq!(run(&plain).get(&2).copied(), Some(999));
    }

    #[test]
    fn provenance_interval_with_negative_lo_is_ignored() {
        // `interval_of` is only consulted when lo >= 0 (the bound must be a
        // non-negative upper bound). A negative-lo interval is discarded and
        // the local bound stands.
        let mut graph = graph_of(vec![
            stmt(0, IR::ConstantFloat { value: 0.0 }, vec![]),
            stmt(1, IR::ConstantInt { value: 1000 }, vec![]),
            stmt(2, IR::ModI, vec![0, 1]),
        ]);
        graph
            .provenance
            .set_interval(ValueId(2), Interval { lo: -5, hi: 7 });
        assert_eq!(run(&graph).get(&2).copied(), Some(999));
    }
}
