use serde::{Deserialize, Serialize};

use crate::ir_defs::IR;
use crate::types::StmtId;

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
#[derive(Debug, Clone)]
pub struct IRStatement {
    pub stmt_id: StmtId,
    pub ir: IR,
    pub arguments: Vec<StmtId>,
    pub dbg: Option<DebugInfo>,
}

impl IRStatement {
    pub fn new(
        stmt_id: StmtId,
        ir: IR,
        arguments: Vec<StmtId>,
        dbg: Option<DebugInfo>,
    ) -> Self {
        Self {
            stmt_id,
            ir,
            arguments,
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
            ir,
            arguments,
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
    /// in_d[i] = number of arguments stmt i depends on
    pub in_d: Vec<u32>,
    /// out_d[i] = number of stmts that reference stmt i
    pub out_d: Vec<u32>,
    /// in_links[i] = arguments of stmt i (same as stmts[i].arguments)
    pub in_links: Vec<Vec<StmtId>>,
    /// out_links[i] = list of stmt IDs that reference stmt i
    pub out_links: Vec<Vec<StmtId>>,
}

impl IRGraph {
    /// Create a new IRGraph from a list of statements.
    /// Recomputes all adjacency information.
    pub fn new(stmts: Vec<IRStatement>) -> Self {
        let mut graph = IRGraph {
            stmts: Vec::new(),
            in_d: Vec::new(),
            out_d: Vec::new(),
            in_links: Vec::new(),
            out_links: Vec::new(),
        };
        graph.update_graph(stmts);
        graph
    }

    /// Replace the statement list and recompute all adjacency information.
    pub fn update_graph(&mut self, stmts: Vec<IRStatement>) {
        let n = stmts.len();

        // Validate that stmt_id == index
        for (i, stmt) in stmts.iter().enumerate() {
            debug_assert_eq!(
                i as StmtId, stmt.stmt_id,
                "Statement at index {} has stmt_id {}",
                i, stmt.stmt_id
            );
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

        let in_links: Vec<Vec<StmtId>> = stmts.iter().map(|s| s.arguments.clone()).collect();

        self.stmts = stmts;
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

        // Remap arguments
        for stmt in &mut new_stmts {
            for arg in &mut stmt.arguments {
                *arg = id_mapping[arg];
            }
        }

        self.update_graph(new_stmts);
    }

    /// Export the statements list (for Python interop).
    pub fn export_stmts(&self) -> Vec<serde_json::Value> {
        self.stmts.iter().map(|s| s.export()).collect()
    }

    /// Import from a list of exported statement dicts.
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
        IRGraph::new(self.stmts.clone())
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
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 10 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 20 }, vec![], None),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
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
        // stmt0 = const(10), stmt1 = const(20) [unused], stmt2 = const(30), stmt3 = add(stmt0, stmt2)
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 10 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 20 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 30 }, vec![], None),
            IRStatement::new(3, IR::AddI, vec![0, 2], None),
        ];
        let mut graph = IRGraph::new(stmts);

        // Remove unused stmt1
        graph.remove_stmt(1);

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.stmts[0].stmt_id, 0);
        assert_eq!(graph.stmts[0].ir, IR::ConstantInt { value: 10 });
        assert_eq!(graph.stmts[1].stmt_id, 1);
        assert_eq!(graph.stmts[1].ir, IR::ConstantInt { value: 30 });
        assert_eq!(graph.stmts[2].stmt_id, 2);
        assert_eq!(graph.stmts[2].ir, IR::AddI);
        assert_eq!(graph.stmts[2].arguments, vec![0, 1]);
    }

    #[test]
    fn test_remove_unreferenced_stmt() {
        // stmt0 = const(10), stmt1 = const(20), stmt2 = const(30), stmt3 = add(stmt0, stmt2)
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 10 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 20 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 30 }, vec![], None),
            IRStatement::new(3, IR::AddI, vec![0, 2], None),
        ];
        let mut graph = IRGraph::new(stmts);

        // Remove stmt1 (unused constant 20)
        graph.remove_stmt(1);

        assert_eq!(graph.len(), 3);
        // Old stmt0 -> new id 0
        assert_eq!(graph.stmts[0].ir, IR::ConstantInt { value: 10 });
        // Old stmt2 -> new id 1
        assert_eq!(graph.stmts[1].ir, IR::ConstantInt { value: 30 });
        // Old stmt3 -> new id 2, args remapped: [0, 2] -> [0, 1]
        assert_eq!(graph.stmts[2].ir, IR::AddI);
        assert_eq!(graph.stmts[2].arguments, vec![0, 1]);
    }

    #[test]
    fn test_remove_stmt_bunch() {
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 1 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 2 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 3 }, vec![], None),
            IRStatement::new(3, IR::ConstantInt { value: 4 }, vec![], None),
            IRStatement::new(4, IR::AddI, vec![0, 3], None),
        ];
        let mut graph = IRGraph::new(stmts);

        // Remove stmts 1 and 2
        graph.remove_stmt_bunch(&[1, 2]);

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.stmts[0].ir, IR::ConstantInt { value: 1 });
        assert_eq!(graph.stmts[1].ir, IR::ConstantInt { value: 4 });
        assert_eq!(graph.stmts[2].ir, IR::AddI);
        assert_eq!(graph.stmts[2].arguments, vec![0, 1]);
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
