use std::collections::HashMap;
use std::mem::{discriminant, Discriminant};

use crate::circuit_input::InputPath;
use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::types::StmtId;

use super::IRPass;

pub struct DuplicateCodeElimination;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum FastDceKey {
    ConstantInt(i64),
    ConstantBool(bool),
    Commutative(Discriminant<IR>, StmtId, StmtId),
    SumF(u32, Vec<StmtId>),
    ProductSumF(u32, Vec<ProductKey>),
    ReadInteger(InputPath, bool),
    ReadFloat(InputPath, bool),
    ReadHash(InputPath, bool),
}

/// A square of an exact difference is independent of its orientation.
/// Keep it distinct from an ordinary product of the difference's operands.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ProductKey {
    Product(StmtId, StmtId),
    DifferenceSquare(StmtId, StmtId),
}

fn commutative_args(args: &[StmtId]) -> Option<(StmtId, StmtId)> {
    if args.len() != 2 {
        return None;
    }
    let a = args[0];
    let b = args[1];
    Some(if a <= b { (a, b) } else { (b, a) })
}

fn fast_dce_key(
    ir: &IR,
    args: &[StmtId],
    graph: &IRGraph,
    replacements: &HashMap<StmtId, StmtId>,
) -> Option<FastDceKey> {
    match ir {
        IR::ConstantInt { value } => Some(FastDceKey::ConstantInt(*value)),
        IR::ConstantBool { value } => Some(FastDceKey::ConstantBool(*value)),
        IR::AddI
        | IR::MulI
        | IR::BitAndI
        | IR::BitOrI
        | IR::AddF
        | IR::MulF
        | IR::EqI
        | IR::NeI
        | IR::EqF
        | IR::NeF => {
            let (a, b) = commutative_args(args)?;
            Some(FastDceKey::Commutative(discriminant(ir), a, b))
        }
        IR::SumF { len } => Some(FastDceKey::SumF(*len, args.to_vec())),
        IR::ProductSumF { len } if args.len() == 2 * (*len as usize) => {
            // Commuting the factors of each raw product preserves the
            // reduction schedule and its single final rescale. Do not
            // reorder pairs or merge independently rounded products.
            let pairs = args
                .chunks_exact(2)
                .map(|pair| {
                    if pair[0] == pair[1] {
                        let factor = &graph.stmts[pair[0] as usize];
                        if matches!(factor.ir, IR::SubF) && factor.arguments.len() == 2 {
                            let operands: Vec<_> = factor
                                .arguments
                                .iter()
                                .map(|a| replacements.get(a).copied().unwrap_or(*a))
                                .collect();
                            let (a, b) = commutative_args(&operands).unwrap();
                            // Both signs change together, before the one rescale.
                            // The raw sum, quotient window and root domain are
                            // identical; observable SubF users keep their sign.
                            return ProductKey::DifferenceSquare(a, b);
                        }
                    }
                    let (a, b) = commutative_args(pair).unwrap();
                    ProductKey::Product(a, b)
                })
                .collect();
            Some(FastDceKey::ProductSumF(*len, pairs))
        }
        IR::ReadInteger { path, is_public } => {
            Some(FastDceKey::ReadInteger(path.clone(), *is_public))
        }
        IR::ReadFloat { path, is_public } => Some(FastDceKey::ReadFloat(path.clone(), *is_public)),
        IR::ReadHash { path, is_public } => Some(FastDceKey::ReadHash(path.clone(), *is_public)),
        _ => None,
    }
}

impl IRPass for DuplicateCodeElimination {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        // Phase 1: identify duplicates
        //
        // `ReadMemory` and `WriteMemory` carry order dependence and must
        // not be deduplicated — two `ReadMemory` ops at the same address
        // are NOT equivalent if a `WriteMemory` to that segment happened
        // in between (read-after-write hazard exposed by P3 segarr-write
        // paths). Other fixed ops (e.g. ReadInteger reading a fixed input
        // path, Assert) remain dedupable: they're either pure-input or
        // are kept by `is_fixed` from being eliminated entirely.
        let mut to_be_replaced: HashMap<StmtId, StmtId> = HashMap::new();
        let mut fast_seen: HashMap<FastDceKey, StmtId> = HashMap::new();
        let mut seen: HashMap<(Discriminant<IR>, Vec<StmtId>), Vec<(IR, StmtId)>> = HashMap::new();

        for stmt in &ir_graph.stmts {
            // Canonicalize arguments through replacements found so far
            // (statements are in dependency order, so an argument's
            // replacement is already known). This lets duplicates cascade:
            // once two producers unify, their identical consumers unify too,
            // all in a single forward pass.
            let resolved_args: Vec<StmtId> = stmt
                .arguments
                .iter()
                .map(|a| to_be_replaced.get(a).copied().unwrap_or(*a))
                .collect();
            if let Some(key) = fast_dce_key(&stmt.ir, &resolved_args, &ir_graph, &to_be_replaced) {
                if let Some(existing_id) = fast_seen.get(&key).copied() {
                    to_be_replaced.insert(stmt.stmt_id, existing_id);
                } else {
                    fast_seen.insert(key, stmt.stmt_id);
                }
                continue;
            }
            // Side-effecting / order- or count-sensitive ops must not dedupe
            // (e.g. two ExposePublic of the same cell are two public slots).
            // Pure asserts are idempotent, so duplicates of those may fold.
            if stmt.ir.is_fixed()
                && !matches!(
                    stmt.ir,
                    IR::Assert | IR::AssertBit | IR::AssertU128Lt | IR::AssertU128LtBit { .. }
                )
            {
                continue;
            }
            let key = (discriminant(&stmt.ir), resolved_args);
            let bucket = seen.entry(key).or_default();
            let mut existing = None;
            for (ir, id) in bucket.iter() {
                if *ir == stmt.ir {
                    existing = Some(*id);
                    break;
                }
            }
            if let Some(existing_id) = existing {
                to_be_replaced.insert(stmt.stmt_id, existing_id);
            } else {
                bucket.push((stmt.ir.clone(), stmt.stmt_id));
            }
        }

        // Phase 2: rebuild graph, replacing duplicates.
        //
        // This pass used to recreate every surviving statement through
        // IRBuilder::create_ir. That is semantically unnecessary here: DCE is
        // only remapping statement ids, not changing IR constructors or
        // publishing new facts. Direct remapping avoids re-running builder-side
        // construction work on large ProductSumF graphs.
        let mut new_id_for_old: HashMap<StmtId, StmtId> = HashMap::new();
        let mut new_stmts: Vec<IRStatement> = Vec::with_capacity(ir_graph.stmts.len());

        for stmt in ir_graph.get_topological_order(false) {
            if let Some(&replacement) = to_be_replaced.get(&stmt.stmt_id) {
                // This stmt is a duplicate; point it at the canonical stmt's
                // new id so later users of the duplicate remap correctly.
                let replacement_new_id = new_id_for_old[&replacement];
                new_id_for_old.insert(stmt.stmt_id, replacement_new_id);
                continue;
            }

            let new_arguments: Vec<StmtId> = stmt
                .arguments
                .iter()
                .map(|&arg| {
                    let resolved = to_be_replaced.get(&arg).copied().unwrap_or(arg);
                    new_id_for_old[&resolved]
                })
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
        // Provenance: this pass replaces value identities — re-key every
        // record from the duplicate's ValueId to the canonical stmt's.
        // (`apply_pass` then enforces liveness for whatever remains.)
        let mut table = ir_graph.provenance.clone();
        // Build the vid→vid remap once and apply it in a single table pass
        // (`remap_values` is O(table + dups); per-pair `remap_value` was
        // O(table) each, i.e. quadratic on duplicate-heavy kernels).
        let mut vid_map: HashMap<crate::types::ValueId, crate::types::ValueId> =
            HashMap::with_capacity(to_be_replaced.len());
        for (&dup, &canon) in &to_be_replaced {
            let dup_vid = ir_graph.stmts.get(dup as usize).map(|s| s.value_id);
            let canon_vid = ir_graph.stmts.get(canon as usize).map(|s| s.value_id);
            if let (Some(d), Some(c)) = (dup_vid, canon_vid) {
                if d != c {
                    vid_map.insert(d, c);
                }
            }
        }
        table.remap_values(&vid_map);
        new_graph.provenance = table;
        // P0 seam: signal mutation to the resolver. No-op today via
        // StaticOnlyResolver; P1 uses this for cache invalidation.
        new_graph.resolver_mut().on_ir_mutated(&[]);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ValueId;

    #[test]
    fn difference_squares_merge_transitively_without_changing_other_products() {
        let nodes = vec![
            (
                IR::ReadFloat {
                    path: InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            (
                IR::ReadFloat {
                    path: InputPath::new("y", vec![]),
                    is_public: false,
                },
                vec![],
            ),
            (IR::SubF, vec![0, 1]),
            (IR::SubF, vec![1, 0]),
            (IR::ProductSumF { len: 1 }, vec![2, 2]),
            (IR::ProductSumF { len: 1 }, vec![3, 3]),
            (IR::ConstantFloat { value: 0.0 }, vec![]),
            (IR::GteF, vec![4, 6]),
            (IR::GteF, vec![5, 6]),
            (IR::Assert, vec![7]),
            (IR::Assert, vec![8]),
            (IR::SqrtF, vec![4]),
            (IR::SqrtF, vec![5]),
            (IR::ExposePublicF, vec![11]),
            (IR::ExposePublicF, vec![12]),
            (IR::ProductSumF { len: 1 }, vec![2, 3]), // one changed sign
            (IR::ProductSumF { len: 1 }, vec![0, 1]), // not a square
            (IR::ExposePublicF, vec![2]),             // observable orientation
            (IR::ExposePublicF, vec![3]),
            (IR::ProductSumF { len: 2 }, vec![2, 2, 0, 0]),
            (IR::ProductSumF { len: 2 }, vec![0, 0, 3, 3]), // retain pair order
        ];
        let graph = IRGraph::new(
            nodes
                .into_iter()
                .enumerate()
                .map(|(id, (ir, args))| {
                    IRStatement::new(id as StmtId, ValueId::next(), ir, args, vec![], None)
                })
                .collect(),
        );
        let square_vid = graph.stmts[4].value_id;
        let out = DuplicateCodeElimination.exec(graph);
        assert_eq!(
            out.stmts
                .iter()
                .filter(|s| matches!(s.ir, IR::ProductSumF { .. }))
                .count(),
            5
        );
        assert_eq!(
            out.stmts
                .iter()
                .filter(|s| matches!(s.ir, IR::Assert))
                .count(),
            1
        );
        assert_eq!(
            out.stmts
                .iter()
                .filter(|s| matches!(s.ir, IR::SqrtF))
                .count(),
            1
        );
        assert!(out.stmts.iter().any(|s| s.value_id == square_vid));
        let exposed: Vec<_> = out
            .stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::ExposePublicF))
            .collect();
        assert_eq!(exposed[0].arguments, exposed[1].arguments);
        assert_ne!(exposed[2].arguments, exposed[3].arguments);
    }
}
