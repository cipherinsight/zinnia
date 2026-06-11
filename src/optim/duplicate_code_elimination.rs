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
    ProductSumF(u32, Vec<StmtId>),
    ReadInteger(InputPath, bool),
    ReadFloat(InputPath, bool),
    ReadHash(InputPath, bool),
}

fn commutative_args(args: &[StmtId]) -> Option<(StmtId, StmtId)> {
    if args.len() != 2 {
        return None;
    }
    let a = args[0];
    let b = args[1];
    Some(if a <= b { (a, b) } else { (b, a) })
}

fn fast_dce_key(ir: &IR, args: &[StmtId]) -> Option<FastDceKey> {
    match ir {
        IR::ConstantInt { value } => Some(FastDceKey::ConstantInt(*value)),
        IR::ConstantBool { value } => Some(FastDceKey::ConstantBool(*value)),
        IR::AddI | IR::MulI | IR::BitAndI | IR::BitOrI | IR::AddF | IR::MulF => {
            let (a, b) = commutative_args(args)?;
            Some(FastDceKey::Commutative(discriminant(ir), a, b))
        }
        IR::SumF { len } => Some(FastDceKey::SumF(*len, args.to_vec())),
        IR::ProductSumF { len } => Some(FastDceKey::ProductSumF(*len, args.to_vec())),
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
            if matches!(stmt.ir, IR::ReadMemory { .. } | IR::WriteMemory { .. }) {
                // Memory ops never dedupe — order matters.
                continue;
            }
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
            if let Some(key) = fast_dce_key(&stmt.ir, &resolved_args) {
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
        for (&dup, &canon) in &to_be_replaced {
            let dup_vid = ir_graph.stmts.get(dup as usize).map(|s| s.value_id);
            let canon_vid = ir_graph.stmts.get(canon as usize).map(|s| s.value_id);
            if let (Some(d), Some(c)) = (dup_vid, canon_vid) {
                table.remap_value(d, c);
            }
        }
        new_graph.provenance = table;
        // P0 seam: signal mutation to the resolver. No-op today via
        // StaticOnlyResolver; P1 uses this for cache invalidation.
        new_graph.resolver_mut().on_ir_mutated(&[]);
        new_graph
    }
}
