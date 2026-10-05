//! Φ_e, the dependency cone, and the interval layer.
//!
//! Three things the oracle needs before it may call a solver.
//!
//! *Φ_e* (`revision/phase1/README.md` §4, "Checks and epochs") is the slice
//! of Φ a node at epoch `e` asks its questions over: the `var_range`,
//! `precondition` and `declaration` constraints together with the product
//! facts, plus the checks whose position is at most `e`. Checks are numbered
//! in the order phase 0 recorded them.
//!
//! *The cone* (the brief §3.2.2) is what keeps a query small: from the
//! variables of the question and of `g`, take every constraint of Φ_e that
//! mentions one of them, add that constraint's variables, and repeat to a
//! fixpoint. Nothing outside the cone can change the answer, because the
//! formula splits into two variable-disjoint halves.
//!
//! *The interval layer* (`rules/README.md` §6.1, `../../README.md` §5.6) is
//! the cheap arithmetic over the `var_range` constraints alone, extended
//! here to a three-valued evaluation of a Boolean term.

use std::collections::{BTreeSet, HashMap};

use crate::phase0::interval::{interval_of, Interval};
use crate::phase0::phi::{Constraint, ConstraintKind, Phi};
use crate::phase0::term::{BoolId, NodeView, TermArena, TermId, VarId};

/// The constraints of Φ, indexed the way the oracle needs them.
#[derive(Debug)]
pub struct Layers {
    /// Per constraint: its variables, and its position in the check order
    /// when it is a check (1-based, `None` otherwise).
    constraints: Vec<ConstraintInfo>,
    /// Per product fact: the variables it mentions.
    products: Vec<BTreeSet<VarId>>,
    /// Variable to the constraints that mention it.
    by_variable: HashMap<VarId, Vec<usize>>,
    /// Variable to the product facts that mention it.
    products_by_variable: HashMap<VarId, Vec<usize>>,
    /// The variables of each arena node, memoised over the DAG.
    variables: Vec<Option<BTreeSet<VarId>>>,
    /// How many checks Φ holds.
    checks: u32,
}

#[derive(Debug)]
struct ConstraintInfo {
    variables: BTreeSet<VarId>,
    /// 1-based position in the check order, for a check.
    position: Option<u32>,
    is_range: bool,
}

/// One sliced formula: the constraints and product facts a query asserts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Cone {
    pub constraints: Vec<usize>,
    pub products: Vec<usize>,
    pub variables: usize,
}

impl Layers {
    /// Index Φ once per compilation.
    pub fn new(terms: &TermArena, phi: &Phi) -> Layers {
        let mut layers = Layers {
            constraints: Vec::with_capacity(phi.constraints().len()),
            products: Vec::with_capacity(phi.products().len()),
            by_variable: HashMap::new(),
            products_by_variable: HashMap::new(),
            variables: vec![None; terms.node_count()],
            checks: phi.check_count(),
        };
        let mut position = 0;
        for constraint in phi.constraints() {
            let variables = layers.variables_of(terms, constraint.formula.index() as u32).clone();
            let is_check = matches!(constraint.kind, ConstraintKind::Check { .. });
            if is_check {
                position += 1;
            }
            layers.constraints.push(ConstraintInfo {
                variables,
                position: is_check.then_some(position),
                is_range: matches!(constraint.kind, ConstraintKind::VarRange(_)),
            });
        }
        for (at, fact) in phi.products().iter().enumerate() {
            let mut variables = BTreeSet::new();
            variables.insert(fact.var);
            for &factor in &fact.factors {
                variables.extend(layers.variables_of(terms, factor.index() as u32).iter().copied());
            }
            for &variable in &variables {
                layers.products_by_variable.entry(variable).or_default().push(at);
            }
            layers.products.push(variables);
        }
        for (at, info) in layers.constraints.iter().enumerate() {
            for &variable in &info.variables {
                layers.by_variable.entry(variable).or_default().push(at);
            }
        }
        layers
    }

    /// How many checks Φ holds; the epoch of a node that assumed them all.
    pub fn check_count(&self) -> u32 {
        self.checks
    }

    /// The 1-based position of the check at constraint index `at`.
    pub fn position(&self, at: usize) -> Option<u32> {
        self.constraints[at].position
    }

    /// The constraint index of the check at 1-based `position`.
    pub fn check_at(&self, position: u32) -> Option<usize> {
        self.constraints
            .iter()
            .position(|info| info.position == Some(position))
    }

    /// Whether constraint `at` belongs to Φ_e.
    pub fn in_epoch(&self, at: usize, epoch: u32) -> bool {
        match self.constraints[at].position {
            Some(position) => position <= epoch,
            None => true,
        }
    }

    /// The variables of the arena node at `index`, memoised over the DAG.
    ///
    /// The walk is iterative: the DAG below an unrolled loop is as deep as
    /// the loop is long, and phase 1 must not overflow the stack on a
    /// program phase 0 accepted.
    pub fn variables_of(&mut self, terms: &TermArena, index: u32) -> &BTreeSet<VarId> {
        if self.variables.len() < terms.node_count() {
            self.variables.resize(terms.node_count(), None);
        }
        let mut stack = vec![index];
        while let Some(&at) = stack.last() {
            if self.variables[at as usize].is_some() {
                stack.pop();
                continue;
            }
            let node = terms.node_at(at as usize);
            let children = children_of(node);
            let pending: Vec<u32> = children
                .iter()
                .copied()
                .filter(|&child| self.variables[child as usize].is_none())
                .collect();
            if !pending.is_empty() {
                stack.extend(pending);
                continue;
            }
            let mut variables = BTreeSet::new();
            match node {
                NodeView::Var(v) | NodeView::BVar(v) => {
                    variables.insert(v);
                }
                _ => {
                    for child in children {
                        // Filled above.
                        if let Some(set) = &self.variables[child as usize] {
                            variables.extend(set.iter().copied());
                        }
                    }
                }
            }
            self.variables[at as usize] = Some(variables);
            stack.pop();
        }
        // Filled by the loop.
        self.variables[index as usize].as_ref().expect("the node was just filled")
    }

    /// The dependency cone of a question over Φ_e (the brief §3.2.2).
    ///
    /// `seeds` are the arena nodes of the question's terms and of `g`.
    pub fn cone(&mut self, terms: &TermArena, seeds: &[u32], epoch: u32) -> Cone {
        let mut variables: BTreeSet<VarId> = BTreeSet::new();
        for &seed in seeds {
            let of_seed = self.variables_of(terms, seed).clone();
            variables.extend(of_seed);
        }
        let mut constraints: BTreeSet<usize> = BTreeSet::new();
        let mut products: BTreeSet<usize> = BTreeSet::new();
        let mut frontier: Vec<VarId> = variables.iter().copied().collect();
        while let Some(variable) = frontier.pop() {
            let mut fresh: BTreeSet<VarId> = BTreeSet::new();
            if let Some(indices) = self.by_variable.get(&variable) {
                for &at in indices {
                    if !self.in_epoch(at, epoch) || !constraints.insert(at) {
                        continue;
                    }
                    for &next in &self.constraints[at].variables {
                        if variables.insert(next) {
                            fresh.insert(next);
                        }
                    }
                }
            }
            if let Some(indices) = self.products_by_variable.get(&variable) {
                for &at in indices {
                    if !products.insert(at) {
                        continue;
                    }
                    for &next in &self.products[at] {
                        if variables.insert(next) {
                            fresh.insert(next);
                        }
                    }
                }
            }
            frontier.extend(fresh);
        }
        Cone {
            constraints: constraints.into_iter().collect(),
            products: products.into_iter().collect(),
            variables: variables.len(),
        }
    }

    /// Whether the cone holds a constraint that is not a `var_range`, which
    /// is what decides whether the interval layer is exact: intervals read
    /// the ranges only (`rules/README.md` §6.1).
    pub fn cone_is_ranges_only(&self, cone: &Cone) -> bool {
        cone.products.is_empty()
            && cone
                .constraints
                .iter()
                .all(|&at| self.constraints[at].is_range)
    }

    /// Whether every variable of the node at `index` occurs once in it.
    ///
    /// `../../README.md` §5.6: when it does, the interval result is the
    /// exact answer of a MAX. Two children that share a variable make an
    /// occurrence count of at least two, and so does a node reached twice.
    pub fn occurs_once(&mut self, terms: &TermArena, index: u32) -> bool {
        // The nodes in post-order: a node is decided after every child of
        // it, which a reversed pre-order does not give on a DAG — a node
        // reached twice sits at its first position, which may precede a
        // parent found later.
        let mut single: HashMap<u32, bool> = HashMap::new();
        let mut stack: Vec<(u32, bool)> = vec![(index, false)];
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        while let Some((at, expanded)) = stack.pop() {
            if expanded {
                let children = children_of(terms.node_at(at as usize));
                let mut ok = children.iter().all(|child| single[child]);
                for (left, &a) in children.iter().enumerate() {
                    for &b in children.iter().skip(left + 1) {
                        let shared = {
                            let first = self.variables_of(terms, a).clone();
                            let second = self.variables_of(terms, b);
                            first.iter().any(|variable| second.contains(variable))
                        };
                        ok &= !shared;
                    }
                }
                single.insert(at, ok);
                continue;
            }
            if !seen.insert(at) {
                continue;
            }
            stack.push((at, true));
            for child in children_of(terms.node_at(at as usize)) {
                stack.push((child, false));
            }
        }
        single[&index]
    }
}

/// The children of a node, as raw arena indices.
pub fn children_of(node: NodeView) -> Vec<u32> {
    match node {
        NodeView::Const(_) | NodeView::Var(_) | NodeView::BVar(_) => Vec::new(),
        NodeView::True | NodeView::False => Vec::new(),
        NodeView::Add(a, b)
        | NodeView::Sub(a, b)
        | NodeView::Min(a, b)
        | NodeView::Max(a, b)
        | NodeView::Eq(a, b)
        | NodeView::Lt(a, b)
        | NodeView::Le(a, b)
        | NodeView::And(a, b)
        | NodeView::Or(a, b) => vec![a, b],
        NodeView::Mul(_, t) | NodeView::Div(t, _) | NodeView::Mod(t, _) | NodeView::Not(t) => {
            vec![t]
        }
        NodeView::Ite(c, a, b) => vec![c, a, b],
    }
}

/// The three-valued value of a Boolean term over the ranges of Φ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trivalent {
    True,
    False,
    Unknown,
}

impl Trivalent {
    pub fn of_bool(value: bool) -> Trivalent {
        if value {
            Trivalent::True
        } else {
            Trivalent::False
        }
    }

    pub fn not(self) -> Trivalent {
        match self {
            Trivalent::True => Trivalent::False,
            Trivalent::False => Trivalent::True,
            Trivalent::Unknown => Trivalent::Unknown,
        }
    }

    pub fn and(self, other: Trivalent) -> Trivalent {
        match (self, other) {
            (Trivalent::False, _) | (_, Trivalent::False) => Trivalent::False,
            (Trivalent::True, Trivalent::True) => Trivalent::True,
            _ => Trivalent::Unknown,
        }
    }

    pub fn or(self, other: Trivalent) -> Trivalent {
        match (self, other) {
            (Trivalent::True, _) | (_, Trivalent::True) => Trivalent::True,
            (Trivalent::False, Trivalent::False) => Trivalent::False,
            _ => Trivalent::Unknown,
        }
    }
}

/// Evaluate a Boolean term over the intervals of Φ's ranges.
///
/// The ranges hold in every admitted run (`rules/README.md` §6.1), so a
/// `True` here is a proof and a `False` a refutation under the ranges alone.
pub fn evaluate_bool(terms: &TermArena, phi: &Phi, formula: BoolId) -> Trivalent {
    evaluate_node(terms, phi, formula.index() as u32)
}

fn evaluate_node(terms: &TermArena, phi: &Phi, index: u32) -> Trivalent {
    match terms.node_at(index as usize) {
        NodeView::True => Trivalent::True,
        NodeView::False => Trivalent::False,
        NodeView::BVar(_) => Trivalent::Unknown,
        NodeView::Not(b) => evaluate_node(terms, phi, b).not(),
        NodeView::And(a, b) => {
            evaluate_node(terms, phi, a).and(evaluate_node(terms, phi, b))
        }
        NodeView::Or(a, b) => evaluate_node(terms, phi, a).or(evaluate_node(terms, phi, b)),
        NodeView::Eq(a, b) => compare(terms, phi, a, b, Relation::Eq),
        NodeView::Lt(a, b) => compare(terms, phi, a, b, Relation::Lt),
        NodeView::Le(a, b) => compare(terms, phi, a, b, Relation::Le),
        // Not a Boolean node; a `BoolId` never names one.
        _ => Trivalent::Unknown,
    }
}

enum Relation {
    Eq,
    Lt,
    Le,
}

/// A comparison decided by the interval of the difference, which is how the
/// scalar_control group folds an interval-decided gate (`scalar_control/README.md` §2).
fn compare(terms: &TermArena, phi: &Phi, a: u32, b: u32, relation: Relation) -> Trivalent {
    // Both children of a comparison are integer-sorted.
    let (Some(left), Some(right)) = (terms.int_at(a), terms.int_at(b)) else {
        return Trivalent::Unknown;
    };
    let x = interval_of(terms, phi, left);
    let y = interval_of(terms, phi, right);
    match relation {
        Relation::Lt => order(x, y, false),
        Relation::Le => order(x, y, true),
        Relation::Eq => match (order(x, y, false), order(y, x, false)) {
            (Trivalent::True, _) | (_, Trivalent::True) => Trivalent::False,
            (Trivalent::False, Trivalent::False) => match (x.lo, x.hi, y.lo, y.hi) {
                (Some(a), Some(b), Some(c), Some(d)) if a == b && c == d && a == c => {
                    Trivalent::True
                }
                _ => Trivalent::Unknown,
            },
            _ => Trivalent::Unknown,
        },
    }
}

/// `x < y` (or `x ≤ y`) decided by the two intervals.
fn order(x: Interval, y: Interval, inclusive: bool) -> Trivalent {
    let strict = |a: i128, b: i128| if inclusive { a <= b } else { a < b };
    if let (Some(hi), Some(lo)) = (x.hi, y.lo) {
        if strict(hi, lo) {
            return Trivalent::True;
        }
    }
    if let (Some(lo), Some(hi)) = (x.lo, y.hi) {
        if !strict(lo, hi) {
            return Trivalent::False;
        }
    }
    Trivalent::Unknown
}

/// The conjuncts of a condition, for the syntactic layer and for the
/// weaker-condition lookup of `../README.md` §4.
pub fn conjuncts(terms: &TermArena, formula: BoolId) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let mut stack = vec![formula.index() as u32];
    while let Some(at) = stack.pop() {
        match terms.node_at(at as usize) {
            NodeView::True => {}
            NodeView::And(a, b) => stack.extend([a, b]),
            _ => {
                out.insert(at);
            }
        }
    }
    out
}

/// The interval of a term over the ranges of Φ (`rules/README.md` §6.1).
pub fn interval(terms: &TermArena, phi: &Phi, term: TermId) -> Interval {
    interval_of(terms, phi, term)
}

/// The provenance string of a constraint, for the evidence of an answer.
pub fn provenance(constraint: &Constraint) -> &str {
    &constraint.provenance
}
