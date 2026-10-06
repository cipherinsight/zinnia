//! Selection: a minimum-cost set of implementation nodes that justifies
//! every root requirement, without circular justification.
//!
//! A requirement is *justified* by a selected node whose postconditions
//! include it (for facts: a fact that implies it). A selected node needs
//! each of its preconditions justified by another selected node of lower
//! rank, so justification is acyclic. Every selected node is paid once,
//! however many nodes it justifies (sharing).
//!
//! [`select_exact`] solves this with Z3 Optimize: a 0/1 variable per node,
//! an integer rank per node, the justification constraints, and the
//! objective `sum cost * x` (then, among equal costs, the earliest nodes in
//! the order they were offered, so the choice is deterministic).
//! [`select_greedy`] justifies each requirement by the producer of smallest
//! estimated cost, depth first, and is used above [`EXACT_LIMIT`] nodes or
//! when the solver gives up.

use z3::ast::{Bool, Int};
use z3::{Optimize, Params, SatResult};

use super::graph::Requirement;

/// Problems with more nodes than this are solved greedily.
pub const EXACT_LIMIT: usize = 400;

/// Solver time budget per problem.
const TIMEOUT_MS: u32 = 10_000;

/// An implementation node as the selection sees it.
#[derive(Clone, Debug)]
pub struct Node {
    pub pre: Vec<Requirement>,
    pub post: Vec<Requirement>,
    pub cost: u64,
}

/// A selection problem. Requirements already justified for free (free
/// facts) must be left out of `roots` and of every node's `pre`.
#[derive(Clone, Debug, Default)]
pub struct Problem {
    pub nodes: Vec<Node>,
    pub roots: Vec<Requirement>,
}

/// The selected nodes, every producer before the nodes it justifies, and
/// their total cost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub nodes: Vec<usize>,
    pub cost: u64,
}

/// `true` when establishing `produced` justifies `required`.
pub fn justifies(produced: &Requirement, required: &Requirement) -> bool {
    match (produced, required) {
        (Requirement::Fact(p), Requirement::Fact(r)) => p.implies(r),
        (p, r) => p == r,
    }
}

impl Problem {
    /// The nodes whose postconditions justify `required`, in order.
    pub fn producers(&self, required: &Requirement) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&n| self.nodes[n].post.iter().any(|p| justifies(p, required)))
            .collect()
    }
}

/// The cheapest selection: exact up to [`EXACT_LIMIT`] nodes, greedy above
/// (or when the solver gives up). `None` when some root cannot be justified.
pub fn select(problem: &Problem) -> Option<Selection> {
    if problem.nodes.len() <= EXACT_LIMIT {
        if let Some(selection) = select_exact(problem) {
            return Some(selection);
        }
    }
    select_greedy(problem)
}

/// The minimum-cost selection by Z3 Optimize; `None` when no selection
/// exists or the solver gives up.
pub fn select_exact(problem: &Problem) -> Option<Selection> {
    let n = problem.nodes.len();
    if n == 0 {
        return problem.roots.is_empty().then(|| Selection { nodes: Vec::new(), cost: 0 });
    }
    let chosen: Vec<Bool> = (0..n).map(|i| Bool::new_const(format!("x{i}"))).collect();
    let rank: Vec<Int> = (0..n).map(|i| Int::new_const(format!("r{i}"))).collect();
    let opt = Optimize::new();
    let mut params = Params::new();
    params.set_u32("timeout", TIMEOUT_MS);
    opt.set_params(&params);

    for root in &problem.roots {
        let producers: Vec<Bool> = problem.producers(root).into_iter().map(|m| chosen[m].clone()).collect();
        opt.assert(&Bool::or(&producers));
    }
    for (i, node) in problem.nodes.iter().enumerate() {
        for required in &node.pre {
            let justified: Vec<Bool> = problem
                .producers(required)
                .into_iter()
                .filter(|&m| m != i)
                .map(|m| Bool::and(&[chosen[m].clone(), rank[m].lt(&rank[i])]))
                .collect();
            opt.assert(&chosen[i].implies(&Bool::or(&justified)));
        }
    }
    let weighted = |weight: &dyn Fn(usize) -> u64| -> Int {
        let terms: Vec<Int> = (0..n)
            .map(|i| chosen[i].ite(&Int::from_u64(weight(i)), &Int::from_u64(0)))
            .collect();
        Int::add(&terms)
    };
    opt.minimize(&weighted(&|i| problem.nodes[i].cost));
    opt.minimize(&weighted(&|i| i as u64 + 1));

    if opt.check(&[]) != SatResult::Sat {
        return None;
    }
    let model = opt.get_model()?;
    let mut selected: Vec<(i64, usize)> = (0..n)
        .filter(|&i| model.eval(&chosen[i], true).and_then(|b| b.as_bool()).unwrap_or(false))
        .map(|i| (model.eval(&rank[i], true).and_then(|r| r.as_i64()).unwrap_or(0), i))
        .collect();
    selected.sort_unstable();
    let nodes: Vec<usize> = selected.into_iter().map(|(_, i)| i).collect();
    let cost = nodes.iter().map(|&i| problem.nodes[i].cost).sum();
    Some(Selection { nodes, cost })
}

/// A selection built depth first: each unjustified requirement takes the
/// producer with the smallest cost plus the cheapest single producer of
/// each of its unjustified preconditions (selected nodes cost nothing),
/// backtracking when a producer's preconditions cannot be justified
/// without a cycle. `None` when some root cannot be justified.
pub fn select_greedy(problem: &Problem) -> Option<Selection> {
    let mut state = Greedy { problem, selected: vec![false; problem.nodes.len()], order: Vec::new(), active: Vec::new() };
    for root in &problem.roots {
        if !state.justify(root) {
            return None;
        }
    }
    let cost = state.order.iter().map(|&i| problem.nodes[i].cost).sum();
    Some(Selection { nodes: state.order, cost })
}

struct Greedy<'a> {
    problem: &'a Problem,
    selected: Vec<bool>,
    /// Selected nodes, producers first.
    order: Vec<usize>,
    /// Nodes whose preconditions are being justified (a cycle guard).
    active: Vec<usize>,
}

impl Greedy<'_> {
    fn justify(&mut self, required: &Requirement) -> bool {
        let producers = self.problem.producers(required);
        if producers.iter().any(|&m| self.selected[m]) {
            return true;
        }
        let mut options: Vec<(u64, usize)> = producers
            .into_iter()
            .filter(|m| !self.active.contains(m))
            .map(|m| (self.estimate(m), m))
            .collect();
        options.sort_unstable();
        for (_, m) in options {
            let (selected, order) = (self.selected.clone(), self.order.len());
            self.active.push(m);
            let ok = self.problem.nodes[m].pre.iter().all(|r| self.justify(r));
            self.active.pop();
            if ok {
                self.selected[m] = true;
                self.order.push(m);
                return true;
            }
            self.selected = selected;
            self.order.truncate(order);
        }
        false
    }

    /// `m`'s cost plus, for each precondition no selected node justifies,
    /// the cost of its cheapest producer.
    fn estimate(&self, m: usize) -> u64 {
        let node = &self.problem.nodes[m];
        let missing: u64 = node
            .pre
            .iter()
            .map(|r| {
                let producers = self.problem.producers(r);
                if producers.iter().any(|&p| self.selected[p]) {
                    0
                } else {
                    producers.iter().map(|&p| self.problem.nodes[p].cost).min().unwrap_or(u64::MAX / 4)
                }
            })
            .fold(0u64, u64::saturating_add);
        node.cost.saturating_add(missing)
    }
}
