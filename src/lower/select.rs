//! Selection: a minimum-cost set of implementation nodes that justifies
//! every root requirement, without circular justification.
//!
//! A requirement is *justified* by a selected node whose postconditions
//! include it (for facts: a fact that implies it). A selected node needs
//! each of its preconditions justified by another selected node of lower
//! rank, so justification is acyclic. Every selected node is paid once,
//! however many nodes it justifies (sharing).
//!
//! [`select_exact`] solves this with Z3: a 0/1 variable per node, an integer
//! rank per node, the justification constraints, and three objectives, each
//! minimized among the optima of the ones before it:
//!
//! 1. the cost, `sum cost_i * x_i`;
//! 2. the sum of positions, `sum (i + 1) * x_i` (the earliest nodes in the
//!    order they were offered);
//! 3. `sum 2^i * x_i`: the set that leaves out the highest-numbered node, then
//!    the next one, and so on.
//!
//! The third objective makes the optimum unique. Without it, equal sums of
//! positions still left ties (when several lanes each offer the same extra
//! postcondition, any one of them can provide it), and Z3 chose among the
//! tied optima by its internal state, which differed between processes: the
//! same program compiled to circuits whose statements were ordered
//! differently (DS-1000 case 0345 on the Noir target).
//! [`select_greedy`] justifies each requirement by the producer of smallest
//! estimated cost, depth first, and is used above [`EXACT_LIMIT`] nodes or
//! when the solver gives up.

use z3::ast::{Bool, Int};
use z3::{Model, SatResult};

use super::graph::Requirement;
use crate::optim::z3_budget;

/// Problems with more nodes than this are solved greedily.
pub const EXACT_LIMIT: usize = 400;

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

/// The minimum-cost selection by Z3 (ties broken as the module docs say);
/// `None` when no selection exists or the solver gives up.
///
/// The problem is solved in a Z3 context of its own, so the steps it takes
/// (and with them whether it fits `z3_budget::SELECT_RLIMIT`) depend on the
/// problem alone, not on the queries the compilation made before.
pub fn select_exact(problem: &Problem) -> Option<Selection> {
    z3::with_z3_config(&z3::Config::new(), || select_exact_in_context(problem))
}

fn select_exact_in_context(problem: &Problem) -> Option<Selection> {
    let n = problem.nodes.len();
    let chosen: Vec<Bool> = (0..n).map(|i| Bool::new_const(format!("x{i}"))).collect();
    let rank: Vec<Int> = (0..n).map(|i| Int::new_const(format!("r{i}"))).collect();
    let constraints = justification(problem, &chosen, &rank);
    let weighted = |weight: &dyn Fn(usize) -> Int| -> Int {
        let terms: Vec<Int> = (0..n).map(|i| chosen[i].ite(&weight(i), &Int::from_u64(0))).collect();
        Int::add(&terms)
    };
    let cost = weighted(&|i| Int::from_u64(problem.nodes[i].cost));
    let positions = weighted(&|i| Int::from_u64(i as u64 + 1));
    let highest_first = weighted(&power_of_two);

    let opt = z3_budget::optimize(z3_budget::SELECT_RLIMIT);
    for c in &constraints {
        opt.assert(c);
    }
    opt.minimize(&cost);
    opt.minimize(&positions);
    opt.minimize(&highest_first);
    if z3_budget::check_optimize("lower.select", &opt) != SatResult::Sat {
        return None;
    }
    let model = opt.get_model()?;

    let mut selected: Vec<(i64, usize)> = (0..n)
        .filter(|&i| is_chosen(&model, &chosen[i]))
        .map(|i| (model.eval(&rank[i], true).and_then(|r| r.as_i64()).unwrap_or(0), i))
        .collect();
    selected.sort_unstable();
    let nodes: Vec<usize> = selected.into_iter().map(|(_, i)| i).collect();
    let cost = nodes.iter().map(|&i| problem.nodes[i].cost).sum();
    Some(Selection { nodes, cost })
}

/// Every root justified, and every chosen node's preconditions justified by
/// chosen nodes of lower rank.
fn justification(problem: &Problem, chosen: &[Bool], rank: &[Int]) -> Vec<Bool> {
    let mut constraints = Vec::new();
    for root in &problem.roots {
        let producers: Vec<Bool> = problem.producers(root).into_iter().map(|m| chosen[m].clone()).collect();
        constraints.push(Bool::or(&producers));
    }
    for (i, node) in problem.nodes.iter().enumerate() {
        for required in &node.pre {
            let justified: Vec<Bool> = problem
                .producers(required)
                .into_iter()
                .filter(|&m| m != i)
                .map(|m| Bool::and(&[chosen[m].clone(), rank[m].lt(&rank[i])]))
                .collect();
            constraints.push(chosen[i].implies(&Bool::or(&justified)));
        }
    }
    constraints
}

/// `2^i` as a Z3 integer (beyond 64 bits when `i` is).
fn power_of_two(i: usize) -> Int {
    if i < 64 {
        Int::from_u64(1u64 << i)
    } else {
        let digits = (num_bigint::BigUint::from(1u8) << i).to_str_radix(10);
        digits.parse().expect("a decimal numeral")
    }
}

fn is_chosen(model: &Model, x: &Bool) -> bool {
    model.eval(x, true).and_then(|b| b.as_bool()).unwrap_or(false)
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
