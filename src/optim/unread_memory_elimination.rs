//! Unread-memory write elimination.
//!
//! Some IR-build sites materialise a memory segment that downstream code
//! never reads. The clearest example is `np.argwhere(A).min/max(axis=0)`:
//! the argwhere lowering writes a `(count, rank)` coord segment, but the
//! min/max(axis=0) bypass added in round 4 reads the per-cell `keep`
//! predicates directly from `IRBuilder::argwhere_keep_indicators` instead
//! of `ReadMemory`-ing the segment. The segment writes (and their input
//! `select_i` / `add_i` arithmetic) stay alive because `WriteMemory` and
//! `AllocateMemory` are `is_fixed()`, so `DeadCodeElimination` won't touch
//! them. The result is ~6 ops × N input cells of pure overhead — for the
//! nonzero-bbox kernels (0507/0508/0509/0510) this is the entire residual
//! 1.3-1.5× circuit_size gap vs the handwritten Noir baselines.
//!
//! This pass walks the IR, finds every segment whose `ReadMemory` use count
//! is zero, and rebuilds dropping the `WriteMemory` and `AllocateMemory`
//! ops on those segments. The arithmetic that fed `write_val` and `addr`
//! becomes unreferenced and `DeadCodeElimination` (which always runs
//! immediately after this pass in the pipeline) collects it. Soundness:
//! a segment that's never read can't affect any constraint or output;
//! removing its writes is observationally equivalent.

use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRPass;

pub struct UnreadMemoryEliminator;

impl IRPass for UnreadMemoryEliminator {
    fn exec(&self, ir_graph: IRGraph) -> IRGraph {
        // Iterate to fixpoint: eliding a segment's writes leaves the reads
        // that produced the written values unused; eliding those reads can
        // in turn leave further segments unread.
        let mut graph = ir_graph;
        loop {
            let (g, changed) = eliminate_round(graph);
            graph = g;
            if !changed {
                return graph;
            }
        }
    }
}

fn eliminate_round(ir_graph: IRGraph) -> (IRGraph, bool) {
    // Phase 1: tally per-segment Read/Write/Alloc activity and per-stmt
    // value usage.
    let mut read_count: HashMap<u32, usize> = HashMap::new();
    let mut has_writes: HashMap<u32, bool> = HashMap::new();
    let mut used: HashSet<StmtId> = HashSet::new();
    for stmt in &ir_graph.stmts {
        for arg in &stmt.arguments {
            used.insert(*arg);
        }
        match stmt.ir {
            IR::ReadMemory { segment_id } => {
                *read_count.entry(segment_id).or_insert(0) += 1;
            }
            IR::WriteMemory { segment_id } => {
                has_writes.insert(segment_id, true);
            }
            IR::AllocateMemory { segment_id, .. } => {
                // Note the segment exists even with no reads/writes.
                has_writes.entry(segment_id).or_insert(false);
            }
            _ => {}
        }
    }
    // Segments to elide: have at least one write or allocate but zero reads.
    let elide: HashSet<u32> = has_writes
        .keys()
        .filter(|&&seg| read_count.get(&seg).copied().unwrap_or(0) == 0)
        .copied()
        .collect();
    // Reads to elide: a ReadMemory whose value nothing consumes is pure
    // dead work. Its in-circuit bound checks belong to the read itself,
    // not to program semantics — soundness-relevant bounds are discharged
    // by explicit asserts at IR-gen time — so dropping it is sound.
    let dead_reads: HashSet<StmtId> = ir_graph
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::ReadMemory { .. }) && !used.contains(&stmt.stmt_id))
        .map(|stmt| stmt.stmt_id)
        .collect();
    if elide.is_empty() && dead_reads.is_empty() {
        return (ir_graph, false);
    }

    // Phase 2: rebuild dropping WriteMemory + AllocateMemory on elided
    // segments and value-unused ReadMemory ops.
    {
        let mut builder = IRBuilder::new();
        let mut value_lookup: HashMap<StmtId, Value> = HashMap::new();
        let true_const: Value = builder.ir_constant_bool(true);

        for stmt in ir_graph.get_topological_order(false) {
            let drop = match stmt.ir {
                IR::WriteMemory { segment_id } | IR::AllocateMemory { segment_id, .. } => {
                    elide.contains(&segment_id)
                }
                IR::ReadMemory { .. } => dead_reads.contains(&stmt.stmt_id),
                _ => false,
            };
            if drop {
                // Emit a sentinel `Value::None`-shaped placeholder so any
                // (unexpected) downstream lookup of this stmt_id resolves
                // to a benign constant rather than panicking. The rebuild
                // walks in topological order so dropped nodes' consumers
                // would not exist in a well-formed graph anyway, but
                // defensive: pair `value_lookup` with a true constant.
                value_lookup.insert(stmt.stmt_id, true_const.clone());
                continue;
            }
            let ir_args: Vec<Value> = stmt
                .arguments
                .iter()
                .map(|&arg| value_lookup[&arg].clone())
                .collect();
            let new_val = builder.create_ir(&stmt.ir, &ir_args);
            value_lookup.insert(stmt.stmt_id, new_val);
        }

        let mut new_graph = builder.export_ir_graph();
        new_graph.resolver_mut().on_ir_mutated(&[]);
        (new_graph, true)
    }
}
