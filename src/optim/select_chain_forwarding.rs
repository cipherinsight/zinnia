//! Read-after-write forwarding through select chains over a bounded index.
//!
//! A runtime-index write `a[i] = w` over known cells becomes, per cell `k`,
//! `a'[k] = select([i == k], w, a[k])`, and a runtime-index read `a'[i]`
//! becomes the chain `select([i == n-1], a'[n-1], ... select([i == 1], a'[1],
//! a'[0]))`. When the circuit asserts `i < n` (`AssertU128Lt(i, n)`, emitted
//! with the index's bounds check), the read is `w`: the chain is replaced by
//! `w` and dead-code elimination removes it (and the written cells nothing
//! else reads).
//!
//! The rule, for a select `S` whose condition is `[x == k]` for a constant
//! `k`: walk the chain of selects through their false branches while each
//! link's condition is `[x == k_j]` (the same `x`) and each link's true
//! branch is `w` under that condition (`w` itself, or a select on the same
//! condition statement whose true branch is `w`). If the links' constants
//! cover `0..n` and the graph asserts `x < n`, then `S = w`.
//!
//! Soundness: every accepted witness has `0 <= x < n`, so some link's
//! constant equals `x`. Walking down from `S`, the first link whose
//! condition holds is such a link (the conditions above it are false), and
//! it yields its true branch, which is `w` under that condition. The
//! assertion is a constraint of the circuit wherever it appears (a circuit
//! has no control flow), and the rewrite keeps it. Equal conditions must be
//! the same statement, so the pass runs after duplicate-code elimination.

use std::collections::{HashMap, HashSet};

use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::types::StmtId;

use super::IRPass;

pub struct SelectChainForwarding;

fn is_select(ir: &IR) -> bool {
    matches!(ir, IR::SelectI | IR::SelectF | IR::SelectB)
}

fn const_int(graph: &IRGraph, id: StmtId) -> Option<i64> {
    match graph.stmts.get(id as usize)?.ir {
        IR::ConstantInt { value } => Some(value),
        _ => None,
    }
}

/// `(x, k)` for a condition statement `[x == k]` with a constant `k`.
fn equality_with_constant(graph: &IRGraph, cond: StmtId) -> Option<(StmtId, i64)> {
    let stmt = graph.stmts.get(cond as usize)?;
    if !matches!(stmt.ir, IR::EqI) {
        return None;
    }
    match stmt.arguments.as_slice() {
        [a, b] => match (const_int(graph, *a), const_int(graph, *b)) {
            (None, Some(k)) => Some((*a, k)),
            (Some(k), None) => Some((*b, k)),
            _ => None,
        },
        _ => None,
    }
}

/// The value statement `t` takes when `cond` holds: `t` itself, or the true
/// branch of a select of `t` on the same condition statement.
fn value_under(graph: &IRGraph, t: StmtId, cond: StmtId) -> StmtId {
    match graph.stmts.get(t as usize) {
        Some(stmt) if is_select(&stmt.ir) && stmt.arguments.first() == Some(&cond) => stmt.arguments[1],
        _ => t,
    }
}

/// `x -> n` for every `AssertU128Lt(x, n)` with a constant `n` (the smallest).
fn asserted_bounds(graph: &IRGraph) -> HashMap<StmtId, i64> {
    let mut bounds: HashMap<StmtId, i64> = HashMap::new();
    for stmt in &graph.stmts {
        if let (IR::AssertU128Lt, [x, hi]) = (&stmt.ir, stmt.arguments.as_slice()) {
            if let Some(n) = const_int(graph, *hi) {
                let entry = bounds.entry(*x).or_insert(n);
                *entry = (*entry).min(n);
            }
        }
    }
    bounds
}

/// The value the select chain at `top` always takes, if the rule applies.
fn forwarded_value(graph: &IRGraph, top: StmtId, bounds: &HashMap<StmtId, i64>) -> Option<StmtId> {
    let top_stmt = graph.stmts.get(top as usize)?;
    let (x, _) = equality_with_constant(graph, *top_stmt.arguments.first()?)?;
    let n = *bounds.get(&x)?;
    if !(1..=4096).contains(&n) {
        return None;
    }
    let w = value_under(graph, top_stmt.arguments[1], top_stmt.arguments[0]);
    let mut covered: HashSet<i64> = HashSet::new();
    let mut link = top;
    loop {
        let stmt = &graph.stmts[link as usize];
        if !is_select(&stmt.ir) || stmt.arguments.len() != 3 {
            break;
        }
        let (cond, t, f) = (stmt.arguments[0], stmt.arguments[1], stmt.arguments[2]);
        match equality_with_constant(graph, cond) {
            Some((x2, k)) if x2 == x && value_under(graph, t, cond) == w => {
                covered.insert(k);
                link = f;
            }
            _ => break,
        }
    }
    (0..n).all(|k| covered.contains(&k)).then_some(w)
}

impl IRPass for SelectChainForwarding {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let bounds = asserted_bounds(&graph);
        if bounds.is_empty() {
            return graph;
        }
        let mut replaced: HashMap<StmtId, StmtId> = HashMap::new();
        for stmt in &graph.stmts {
            if is_select(&stmt.ir) {
                if let Some(w) = forwarded_value(&graph, stmt.stmt_id, &bounds) {
                    // `w` precedes the chain; follow it if it was replaced too.
                    let w = replaced.get(&w).copied().unwrap_or(w);
                    replaced.insert(stmt.stmt_id, w);
                }
            }
        }
        if replaced.is_empty() {
            return graph;
        }
        // Users of a forwarded select read `w` instead; the selects stay
        // (dead) for dead-code elimination. Statement and value ids are
        // kept, so the provenance table stays valid.
        let stmts: Vec<IRStatement> = graph
            .stmts
            .iter()
            .map(|stmt| {
                let arguments = stmt.arguments.iter().map(|a| replaced.get(a).copied().unwrap_or(*a)).collect();
                IRStatement::new(stmt.stmt_id, stmt.value_id, stmt.ir.clone(), arguments, Vec::new(), stmt.dbg.clone())
            })
            .collect();
        let mut out = IRGraph::new(stmts);
        out.resolver_mut().on_ir_mutated(&[]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ValueId;

    fn graph(stmts: Vec<(IR, Vec<StmtId>)>) -> IRGraph {
        IRGraph::new(
            stmts
                .into_iter()
                .enumerate()
                .map(|(i, (ir, args))| IRStatement::new(i as StmtId, ValueId::next(), ir, args, Vec::new(), None))
                .collect(),
        )
    }

    /// `a = [c0, c1, c2]`, `a[i] = w` (selects), then `a[i]` read back:
    /// forwarded to `w` exactly when `i < 3` is asserted and the chain covers
    /// every row.
    fn write_then_read(bound: Option<i64>, rows: i64) -> IRGraph {
        let mut s: Vec<(IR, Vec<StmtId>)> = vec![
            (IR::ReadInteger { path: crate::circuit_input::InputPath::new("i", vec![]), is_public: false }, vec![]), // 0: i
            (IR::ReadInteger { path: crate::circuit_input::InputPath::new("w", vec![]), is_public: false }, vec![]), // 1: w
        ];
        if let Some(n) = bound {
            s.push((IR::ConstantInt { value: n }, vec![])); // 2
            s.push((IR::AssertU128Lt, vec![0, 2]));
        }
        let mut written = Vec::new();
        let mut conds = Vec::new();
        for k in 0..rows {
            let c = s.len() as StmtId;
            s.push((IR::ConstantInt { value: 10 + k }, vec![])); // old cell
            s.push((IR::ConstantInt { value: k }, vec![]));
            s.push((IR::EqI, vec![0, c + 1]));
            s.push((IR::SelectI, vec![c + 2, 1, c]));
            conds.push(c + 2);
            written.push(c + 3);
        }
        let mut acc = written[0];
        for k in 1..rows as usize {
            let id = s.len() as StmtId;
            s.push((IR::SelectI, vec![conds[k], written[k], acc]));
            acc = id;
        }
        s.push((IR::ExposePublicI, vec![acc]));
        graph(s)
    }

    fn exposed(g: &IRGraph) -> StmtId {
        g.stmts.iter().find(|s| matches!(s.ir, IR::ExposePublicI)).unwrap().arguments[0]
    }

    #[test]
    fn read_after_write_at_a_bounded_index_is_the_written_value() {
        let out = SelectChainForwarding.exec(write_then_read(Some(3), 3));
        assert_eq!(exposed(&out), 1, "the read should be w");
    }

    #[test]
    fn no_forwarding_without_the_bound_or_full_coverage() {
        for (bound, rows) in [(None, 3), (Some(4), 3), (Some(3), 2)] {
            let g = write_then_read(bound, rows);
            let before = exposed(&g);
            assert_eq!(exposed(&SelectChainForwarding.exec(g)), before, "bound {bound:?}, rows {rows}");
        }
    }
}
