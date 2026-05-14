use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ir_defs::IR;
use crate::optim::resolver::{Resolver, StaticOnlyResolver};
use crate::types::{StmtId, ValueId};

// ---------------------------------------------------------------------------
// DebugInfo
// ---------------------------------------------------------------------------

/// Debug information attached to IR statements for error reporting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DebugInfo {
    pub line: Option<u32>,
    pub col: Option<u32>,
    pub source: Option<String>,
}

// ---------------------------------------------------------------------------
// IRStatement
// ---------------------------------------------------------------------------

/// A single IR statement in the compilation pipeline.
/// Mirrors Python `IRStatement` from `zinnia/compile/ir/ir_stmt.py`.
///
/// `stmt_id` is the IR-layer position (== index in the `IRGraph::stmts`
/// vec). `value_id` is the identity of the `Value` this statement
/// produces (compiler.value-id-and-fact-leaves).
///
/// **Phase 3 — dual-view arguments.** `arguments` retains its `Vec<StmtId>`
/// shape so existing optim passes (constant-folding, DCE, duplicate
/// elimination, etc.) and the SMT walker can keep using direct
/// `stmts[arg as usize]` indexing. `arg_values` is the parallel
/// `Vec<ValueId>` view: the same arguments expressed in the
/// compilation-layer identity space. Fact-aware code (the resolver-prove
/// path, future refinement-type forward propagation) reads `arg_values`
/// so it never has to drop down to `stmt_id`. Each entry at index `i` in
/// `arg_values` is the `value_id` of the IR statement at
/// `arguments[i]`.
#[derive(Debug, Clone)]
pub struct IRStatement {
    pub stmt_id: StmtId,
    pub value_id: ValueId,
    pub ir: IR,
    pub arguments: Vec<StmtId>,
    pub arg_values: Vec<ValueId>,
    pub dbg: Option<DebugInfo>,
}

impl IRStatement {
    pub fn new(
        stmt_id: StmtId,
        value_id: ValueId,
        ir: IR,
        arguments: Vec<StmtId>,
        arg_values: Vec<ValueId>,
        dbg: Option<DebugInfo>,
    ) -> Self {
        Self {
            stmt_id,
            value_id,
            ir,
            arguments,
            arg_values,
            dbg,
        }
    }

    /// Serialize to the Python IRStatement.export() format.
    pub fn export(&self) -> serde_json::Value {
        serde_json::json!({
            "stmt_id": self.stmt_id,
            "ir_instance": self.ir.export(),
            "arguments": self.arguments,
        })
    }

    /// Deserialize from the Python IRStatement.export() format.
    /// Mints a fresh `value_id`; `arg_values` is filled later by
    /// `IRGraph::import_stmts` once each stmt's `value_id` is known.
    pub fn import_from(data: &serde_json::Value) -> Result<Self, String> {
        let stmt_id = data["stmt_id"]
            .as_u64()
            .ok_or("IRStatement: missing stmt_id")? as StmtId;
        let ir = IR::import_from(&data["ir_instance"])?;
        let arguments: Vec<StmtId> = data["arguments"]
            .as_array()
            .ok_or("IRStatement: missing arguments")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as StmtId)
            .collect();
        Ok(IRStatement {
            stmt_id,
            value_id: ValueId::next(),
            ir,
            arguments,
            arg_values: Vec::new(),
            dbg: None,
        })
    }
}

// ---------------------------------------------------------------------------
// IRGraph
// ---------------------------------------------------------------------------

/// A graph of IR statements with adjacency information.
/// Mirrors Python `IRGraph` from `zinnia/compile/ir/ir_graph.py`.
pub struct IRGraph {
    pub stmts: Vec<IRStatement>,
    /// `ValueId → StmtId` (= index in `stmts`) lookup. Built and
    /// maintained by `update_graph` after Phase 3
    /// (compiler.value-id-and-fact-leaves) switched `arguments` to
    /// reference Values via `ValueId`. Walkers consult this to translate
    /// each argument's ValueId back to an IR-layer position when they
    /// need direct indexed access.
    pub value_to_stmt: HashMap<ValueId, StmtId>,
    /// in_d[i] = number of arguments stmt i depends on
    pub in_d: Vec<u32>,
    /// out_d[i] = number of stmts that reference stmt i
    pub out_d: Vec<u32>,
    /// in_links[i] = arguments of stmt i (in stmt_id space, derived from
    /// `arguments` via `value_to_stmt`).
    pub in_links: Vec<Vec<StmtId>>,
    /// out_links[i] = list of stmt IDs that reference stmt i
    pub out_links: Vec<Vec<StmtId>>,
    /// P0 SMT-resolver seam: the active [`Resolver`] for "must be a
    /// compile-time constant" queries during the optim pipeline. Optim
    /// passes that mutate the IR call `resolver_mut().on_ir_mutated(&[])`
    /// at the end of `exec`. The default is [`StaticOnlyResolver`], whose
    /// `on_ir_mutated` is a no-op — so today's behaviour is unchanged.
    /// P1 swaps in `SmtResolver`, which uses the hook for cache
    /// invalidation. See `src/optim/resolver.rs`.
    resolver: Box<dyn Resolver>,
    /// P5: telemetry handle from the AST→IR phase, snapshotted before the
    /// builder was consumed. Lets `compile_circuit` log the AST→IR
    /// counters even though that phase's resolver no longer exists.
    /// `None` when the upstream resolver had no telemetry (e.g., the
    /// `StaticOnlyResolver` default with `smt_enable=false`).
    astgen_telemetry: Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>>,
}

impl IRGraph {
    /// Create a new IRGraph from a list of statements.
    /// Recomputes all adjacency information.
    pub fn new(stmts: Vec<IRStatement>) -> Self {
        let mut graph = IRGraph {
            stmts: Vec::new(),
            value_to_stmt: HashMap::new(),
            in_d: Vec::new(),
            out_d: Vec::new(),
            in_links: Vec::new(),
            out_links: Vec::new(),
            resolver: Box::new(StaticOnlyResolver::new()),
            astgen_telemetry: None,
        };
        graph.update_graph(stmts);
        graph
    }

    /// Stash an AST→IR-phase telemetry handle on the graph (P5). Read by
    /// `compile_circuit` for the end-of-compilation summary.
    pub fn set_astgen_telemetry(
        &mut self,
        t: std::sync::Arc<crate::optim::telemetry::SmtTelemetry>,
    ) {
        self.astgen_telemetry = Some(t);
    }

    /// Return the AST→IR-phase telemetry handle if one was attached.
    pub fn astgen_telemetry(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        self.astgen_telemetry.as_ref().map(std::sync::Arc::clone)
    }

    /// Optim-phase resolver telemetry, when the active resolver has one.
    pub fn resolver_telemetry(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        self.resolver.telemetry_handle()
    }

    /// Borrow the active [`Resolver`].
    pub fn resolver(&self) -> &dyn Resolver {
        &*self.resolver
    }

    /// Borrow the active [`Resolver`] mutably. Optim passes that mutate the
    /// IR call this and invoke `on_ir_mutated(&[])` at the end of `exec` so
    /// future stateful resolvers (P1's `SmtResolver`) can invalidate caches.
    pub fn resolver_mut(&mut self) -> &mut dyn Resolver {
        &mut *self.resolver
    }

    /// Hand out `&mut dyn Resolver` and `&[IRStatement]` simultaneously
    /// from the same `&mut IRGraph`. P1's `SmtResolver` walks the IR via
    /// the resolver's `_with_stmts` trait methods; this is the chokepoint
    /// optim passes use when they need full-power SMT discharge.
    pub fn split_resolver_and_stmts(
        &mut self,
    ) -> (&mut dyn Resolver, &[IRStatement]) {
        (&mut *self.resolver, &self.stmts)
    }

    /// Swap in a different [`Resolver`] implementation. Reserved for P1+.
    pub fn set_resolver(&mut self, r: Box<dyn Resolver>) {
        self.resolver = r;
    }

    /// Replace the statement list and recompute all adjacency information.
    /// Builds `value_to_stmt` from each statement's `value_id` and
    /// fills any empty `arg_values` by translating `arguments` (stmt_id
    /// space) via that map.
    pub fn update_graph(&mut self, mut stmts: Vec<IRStatement>) {
        let n = stmts.len();

        // Validate that stmt_id == index
        for (i, stmt) in stmts.iter().enumerate() {
            debug_assert_eq!(
                i as StmtId, stmt.stmt_id,
                "Statement at index {} has stmt_id {}",
                i, stmt.stmt_id
            );
        }

        // Build ValueId → StmtId map for downstream consumers and to
        // fill any imported statements' empty arg_values.
        let mut value_to_stmt: HashMap<ValueId, StmtId> = HashMap::with_capacity(n);
        for stmt in &stmts {
            value_to_stmt.insert(stmt.value_id, stmt.stmt_id);
        }

        // Backfill arg_values for statements loaded via import (where
        // only `arguments` was filled). At-construction-time statements
        // already have both fields set by `IRBuilder::create_ir`.
        let stmt_value_ids: Vec<ValueId> =
            stmts.iter().map(|s| s.value_id).collect();
        for stmt in stmts.iter_mut() {
            if stmt.arg_values.is_empty() && !stmt.arguments.is_empty() {
                stmt.arg_values = stmt
                    .arguments
                    .iter()
                    .filter_map(|sid| stmt_value_ids.get(*sid as usize).copied())
                    .collect();
            }
        }

        let mut in_d = vec![0u32; n];
        let mut out_d = vec![0u32; n];
        let mut out_links: Vec<Vec<StmtId>> = vec![Vec::new(); n];

        for (i, stmt) in stmts.iter().enumerate() {
            for &arg in &stmt.arguments {
                in_d[i] += 1;
                out_d[arg as usize] += 1;
                out_links[arg as usize].push(i as StmtId);
            }
        }

        let in_links: Vec<Vec<StmtId>> =
            stmts.iter().map(|s| s.arguments.clone()).collect();

        self.stmts = stmts;
        self.value_to_stmt = value_to_stmt;
        self.in_d = in_d;
        self.out_d = out_d;
        self.in_links = in_links;
        self.out_links = out_links;
    }

    /// Returns copies of the in/out degree vectors.
    pub fn get_io_degrees(&self) -> (Vec<u32>, Vec<u32>) {
        (self.in_d.clone(), self.out_d.clone())
    }

    /// Returns copies of the in/out link vectors.
    pub fn get_io_links(&self) -> (Vec<Vec<StmtId>>, Vec<Vec<StmtId>>) {
        (self.in_links.clone(), self.out_links.clone())
    }

    /// Returns statements in topological order (forward or reverse).
    /// Since statements are always stored in order, this is trivial.
    pub fn get_topological_order(&self, reverse: bool) -> Vec<&IRStatement> {
        if reverse {
            self.stmts.iter().rev().collect()
        } else {
            self.stmts.iter().collect()
        }
    }

    /// Retrieve a statement by its ID.
    pub fn retrieve_stmt_with_id(&self, idx: StmtId) -> &IRStatement {
        &self.stmts[idx as usize]
    }

    /// Remove a single statement and re-index all remaining statements.
    pub fn remove_stmt(&mut self, idx: StmtId) {
        self.remove_stmt_bunch(&[idx]);
    }

    /// Remove a batch of statements and re-index all remaining statements.
    /// `arguments` (stmt_id space) is remapped through `id_mapping`;
    /// `arg_values` (value_id space) is stable across stmt_id renumbering
    /// and needs no remapping.
    pub fn remove_stmt_bunch(&mut self, indices: &[StmtId]) {
        let index_set: std::collections::HashSet<StmtId> =
            indices.iter().copied().collect();

        let mut new_stmts: Vec<IRStatement> = Vec::new();
        let mut id_mapping: std::collections::HashMap<StmtId, StmtId> =
            std::collections::HashMap::new();

        for stmt in &self.stmts {
            if !index_set.contains(&stmt.stmt_id) {
                let new_id = new_stmts.len() as StmtId;
                id_mapping.insert(stmt.stmt_id, new_id);
                let mut new_stmt = stmt.clone();
                new_stmt.stmt_id = new_id;
                new_stmts.push(new_stmt);
            }
        }

        for stmt in &mut new_stmts {
            for arg in &mut stmt.arguments {
                *arg = id_mapping[arg];
            }
            // arg_values is in ValueId space — stable, no remap needed.
        }

        self.update_graph(new_stmts);
    }

    /// Export the statements list (for Python interop).
    pub fn export_stmts(&self) -> Vec<serde_json::Value> {
        self.stmts.iter().map(|s| s.export()).collect()
    }

    /// Import from a list of exported statement dicts. Imported
    /// statements have only `arguments` (stmt_ids) populated;
    /// `update_graph` fills `arg_values` and the value_to_stmt map.
    pub fn import_stmts(data: &[serde_json::Value]) -> Result<Self, String> {
        let mut stmts = Vec::with_capacity(data.len());
        for d in data {
            stmts.push(IRStatement::import_from(d)?);
        }
        Ok(IRGraph::new(stmts))
    }

    /// Returns the number of statements in the graph.
    pub fn len(&self) -> usize {
        self.stmts.len()
    }

    /// Returns true if the graph has no statements.
    pub fn is_empty(&self) -> bool {
        self.stmts.is_empty()
    }
}

impl Clone for IRGraph {
    fn clone(&self) -> Self {
        let mut g = IRGraph::new(self.stmts.clone());
        if let Some(t) = self.astgen_telemetry.as_ref() {
            g.astgen_telemetry = Some(std::sync::Arc::clone(t));
        }
        g
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_defs::IR;

    fn make_test_graph() -> IRGraph {
        // Build: stmt0 = constant_int(10), stmt1 = constant_int(20), stmt2 = add_i(stmt0, stmt1)
        let v0 = ValueId::next();
        let v1 = ValueId::next();
        let v2 = ValueId::next();
        let stmts = vec![
            IRStatement::new(0, v0, IR::ConstantInt { value: 10 }, vec![], vec![], None),
            IRStatement::new(1, v1, IR::ConstantInt { value: 20 }, vec![], vec![], None),
            IRStatement::new(2, v2, IR::AddI, vec![0, 1], vec![v0, v1], None),
        ];
        IRGraph::new(stmts)
    }

    #[test]
    fn test_graph_construction() {
        let graph = make_test_graph();
        assert_eq!(graph.len(), 3);

        // stmt0: in=0, out=1 (referenced by stmt2)
        assert_eq!(graph.in_d[0], 0);
        assert_eq!(graph.out_d[0], 1);

        // stmt1: in=0, out=1 (referenced by stmt2)
        assert_eq!(graph.in_d[1], 0);
        assert_eq!(graph.out_d[1], 1);

        // stmt2: in=2 (depends on stmt0, stmt1), out=0
        assert_eq!(graph.in_d[2], 2);
        assert_eq!(graph.out_d[2], 0);

        // out_links
        assert_eq!(graph.out_links[0], vec![2]);
        assert_eq!(graph.out_links[1], vec![2]);
        assert_eq!(graph.out_links[2], Vec::<StmtId>::new());
    }

    #[test]
    fn test_remove_stmt() {
        let v0 = ValueId::next();
        let v1 = ValueId::next();
        let v2 = ValueId::next();
        let v3 = ValueId::next();
        let stmts = vec![
            IRStatement::new(0, v0, IR::ConstantInt { value: 10 }, vec![], vec![], None),
            IRStatement::new(1, v1, IR::ConstantInt { value: 20 }, vec![], vec![], None),
            IRStatement::new(2, v2, IR::ConstantInt { value: 30 }, vec![], vec![], None),
            IRStatement::new(3, v3, IR::AddI, vec![0, 2], vec![v0, v2], None),
        ];
        let mut graph = IRGraph::new(stmts);

        graph.remove_stmt(1);

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.stmts[0].stmt_id, 0);
        assert_eq!(graph.stmts[0].ir, IR::ConstantInt { value: 10 });
        assert_eq!(graph.stmts[1].stmt_id, 1);
        assert_eq!(graph.stmts[1].ir, IR::ConstantInt { value: 30 });
        assert_eq!(graph.stmts[2].stmt_id, 2);
        assert_eq!(graph.stmts[2].ir, IR::AddI);
        // arguments (stmt_id space) remap to the new positions [0, 1].
        assert_eq!(graph.stmts[2].arguments, vec![0, 1]);
        // arg_values (value_id space) is stable — v0 and v2 unchanged.
        assert_eq!(graph.stmts[2].arg_values, vec![v0, v2]);
    }

    #[test]
    fn test_remove_unreferenced_stmt() {
        let v0 = ValueId::next();
        let v1 = ValueId::next();
        let v2 = ValueId::next();
        let v3 = ValueId::next();
        let stmts = vec![
            IRStatement::new(0, v0, IR::ConstantInt { value: 10 }, vec![], vec![], None),
            IRStatement::new(1, v1, IR::ConstantInt { value: 20 }, vec![], vec![], None),
            IRStatement::new(2, v2, IR::ConstantInt { value: 30 }, vec![], vec![], None),
            IRStatement::new(3, v3, IR::AddI, vec![0, 2], vec![v0, v2], None),
        ];
        let mut graph = IRGraph::new(stmts);

        graph.remove_stmt(1);

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.stmts[0].ir, IR::ConstantInt { value: 10 });
        assert_eq!(graph.stmts[1].ir, IR::ConstantInt { value: 30 });
        assert_eq!(graph.stmts[2].ir, IR::AddI);
        assert_eq!(graph.stmts[2].arguments, vec![0, 1]);
        assert_eq!(graph.stmts[2].arg_values, vec![v0, v2]);
    }

    #[test]
    fn test_remove_stmt_bunch() {
        let v0 = ValueId::next();
        let v1 = ValueId::next();
        let v2 = ValueId::next();
        let v3 = ValueId::next();
        let v4 = ValueId::next();
        let stmts = vec![
            IRStatement::new(0, v0, IR::ConstantInt { value: 1 }, vec![], vec![], None),
            IRStatement::new(1, v1, IR::ConstantInt { value: 2 }, vec![], vec![], None),
            IRStatement::new(2, v2, IR::ConstantInt { value: 3 }, vec![], vec![], None),
            IRStatement::new(3, v3, IR::ConstantInt { value: 4 }, vec![], vec![], None),
            IRStatement::new(4, v4, IR::AddI, vec![0, 3], vec![v0, v3], None),
        ];
        let mut graph = IRGraph::new(stmts);

        graph.remove_stmt_bunch(&[1, 2]);

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.stmts[0].ir, IR::ConstantInt { value: 1 });
        assert_eq!(graph.stmts[1].ir, IR::ConstantInt { value: 4 });
        assert_eq!(graph.stmts[2].ir, IR::AddI);
        assert_eq!(graph.stmts[2].arguments, vec![0, 1]);
        assert_eq!(graph.stmts[2].arg_values, vec![v0, v3]);
    }

    #[test]
    fn test_topological_order() {
        let graph = make_test_graph();
        let forward: Vec<StmtId> = graph
            .get_topological_order(false)
            .iter()
            .map(|s| s.stmt_id)
            .collect();
        assert_eq!(forward, vec![0, 1, 2]);

        let reverse: Vec<StmtId> = graph
            .get_topological_order(true)
            .iter()
            .map(|s| s.stmt_id)
            .collect();
        assert_eq!(reverse, vec![2, 1, 0]);
    }

    #[test]
    fn test_export_import_round_trip() {
        let graph = make_test_graph();
        let exported = graph.export_stmts();
        let imported = IRGraph::import_stmts(&exported).unwrap();

        assert_eq!(imported.len(), graph.len());
        for (orig, restored) in graph.stmts.iter().zip(imported.stmts.iter()) {
            assert_eq!(orig.stmt_id, restored.stmt_id);
            assert_eq!(orig.ir, restored.ir);
            assert_eq!(orig.arguments, restored.arguments);
        }
    }

    #[test]
    fn test_clone() {
        let graph = make_test_graph();
        let cloned = graph.clone();
        assert_eq!(cloned.len(), graph.len());
        assert_eq!(cloned.in_d, graph.in_d);
        assert_eq!(cloned.out_d, graph.out_d);
    }
}
