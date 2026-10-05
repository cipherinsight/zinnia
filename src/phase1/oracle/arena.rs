//! The term DAG as Z3 expressions.
//!
//! One Z3 expression per node of the phase 0 arena, built on demand and
//! memoised, so a term shared by many questions is translated once. This is
//! `testing/phase0/phi_solver.py`'s `Z3Arena` in Rust, with the same floor
//! `div` / `mod` rewrites `src/phase0/term/print.rs` states: SMT-LIB's and
//! Z3's integer division is Euclidean, Python's is floor, and the two differ
//! exactly when the divisor is negative.

use std::str::FromStr;

use z3::ast::{Bool, Int};

use crate::phase0::term::{BoolId, NodeView, TermArena, TermId, VarId};

use super::layers::children_of;

/// One node's Z3 expression, at whichever sort the node has.
#[derive(Debug, Clone)]
pub enum Expr {
    Int(Int),
    Bool(Bool),
}

impl Expr {
    fn int(&self) -> &Int {
        match self {
            Expr::Int(value) => value,
            // Unreachable: the sort of a child follows from the parent's
            // constructor, and the phase 0 arena keeps the two apart.
            Expr::Bool(_) => unreachable!("an integer-sorted child is an Int"),
        }
    }

    fn boolean(&self) -> &Bool {
        match self {
            Expr::Bool(value) => value,
            Expr::Int(_) => unreachable!("a Boolean-sorted child is a Bool"),
        }
    }
}

/// The arena's nodes as Z3 expressions, filled lazily.
pub struct Z3Arena {
    nodes: Vec<Option<Expr>>,
    variables: Vec<Option<Expr>>,
}

impl Z3Arena {
    pub fn new(terms: &TermArena) -> Z3Arena {
        Z3Arena {
            nodes: vec![None; terms.node_count()],
            variables: vec![None; terms.var_count()],
        }
    }

    /// The Z3 constant of a variable of Φ, by its declared name and sort.
    pub fn variable(&mut self, terms: &TermArena, v: VarId) -> Expr {
        if self.variables.len() < terms.var_count() {
            self.variables.resize(terms.var_count(), None);
        }
        if let Some(existing) = &self.variables[v.0 as usize] {
            return existing.clone();
        }
        let declaration = terms.var_decl(v);
        let built = if declaration.sort.is_bool() {
            Expr::Bool(Bool::new_const(declaration.name.clone()))
        } else {
            Expr::Int(Int::new_const(declaration.name.clone()))
        };
        self.variables[v.0 as usize] = Some(built.clone());
        built
    }

    pub fn int(&mut self, terms: &TermArena, t: TermId) -> Int {
        match self.node(terms, t.index() as u32) {
            Expr::Int(value) => value,
            Expr::Bool(_) => unreachable!("a TermId names an integer-sorted node"),
        }
    }

    pub fn boolean(&mut self, terms: &TermArena, b: BoolId) -> Bool {
        match self.node(terms, b.index() as u32) {
            Expr::Bool(value) => value,
            Expr::Int(_) => unreachable!("a BoolId names a Boolean-sorted node"),
        }
    }

    /// The expression of the node at arena index `index`.
    ///
    /// Iterative, for the reason `src/phase0/term/print.rs` gives: the DAG
    /// below an unrolled loop is as deep as the loop is long.
    pub fn node(&mut self, terms: &TermArena, index: u32) -> Expr {
        if self.nodes.len() < terms.node_count() {
            self.nodes.resize(terms.node_count(), None);
        }
        let mut stack = vec![index];
        while let Some(&at) = stack.last() {
            if self.nodes[at as usize].is_some() {
                stack.pop();
                continue;
            }
            let node = terms.node_at(at as usize);
            let pending: Vec<u32> = children_of(node)
                .into_iter()
                .filter(|&child| self.nodes[child as usize].is_none())
                .collect();
            if !pending.is_empty() {
                stack.extend(pending);
                continue;
            }
            let built = self.build(terms, node);
            self.nodes[at as usize] = Some(built);
            stack.pop();
        }
        // Filled by the loop.
        self.nodes[index as usize]
            .clone()
            .expect("the node was just filled")
    }

    fn build(&mut self, terms: &TermArena, node: NodeView) -> Expr {
        let child = |arena: &Self, index: u32| {
            arena.nodes[index as usize]
                .clone()
                .expect("a child is built before its parent")
        };
        let pair = |arena: &Self, a: u32, b: u32| {
            (child(arena, a).int().clone(), child(arena, b).int().clone())
        };
        match node {
            NodeView::Const(n) => Expr::Int(integer(n)),
            NodeView::Var(v) | NodeView::BVar(v) => self.variable(terms, v),
            NodeView::Add(a, b) => {
                let (x, y) = pair(self, a, b);
                Expr::Int(Int::add(&[x, y]))
            }
            NodeView::Sub(a, b) => {
                let (x, y) = pair(self, a, b);
                Expr::Int(Int::sub(&[x, y]))
            }
            NodeView::Mul(k, t) => {
                let x = child(self, t).int().clone();
                Expr::Int(Int::mul(&[integer(k), x]))
            }
            NodeView::Div(t, k) => Expr::Int(floor_div(child(self, t).int(), k)),
            NodeView::Mod(t, k) => Expr::Int(floor_mod(child(self, t).int(), k)),
            NodeView::Min(a, b) => {
                let (x, y) = pair(self, a, b);
                Expr::Int(x.le(y.clone()).ite(&x, &y))
            }
            NodeView::Max(a, b) => {
                let (x, y) = pair(self, a, b);
                Expr::Int(x.le(y.clone()).ite(&y, &x))
            }
            NodeView::Ite(c, a, b) => {
                let condition = child(self, c).boolean().clone();
                let (x, y) = pair(self, a, b);
                Expr::Int(condition.ite(&x, &y))
            }
            NodeView::True => Expr::Bool(Bool::from_bool(true)),
            NodeView::False => Expr::Bool(Bool::from_bool(false)),
            NodeView::Eq(a, b) => {
                let (x, y) = pair(self, a, b);
                Expr::Bool(x.eq(y))
            }
            NodeView::Lt(a, b) => {
                let (x, y) = pair(self, a, b);
                Expr::Bool(x.lt(y))
            }
            NodeView::Le(a, b) => {
                let (x, y) = pair(self, a, b);
                Expr::Bool(x.le(y))
            }
            NodeView::Not(b) => Expr::Bool(child(self, b).boolean().not()),
            NodeView::And(a, b) => Expr::Bool(Bool::and(&[
                child(self, a).boolean().clone(),
                child(self, b).boolean().clone(),
            ])),
            NodeView::Or(a, b) => Expr::Bool(Bool::or(&[
                child(self, a).boolean().clone(),
                child(self, b).boolean().clone(),
            ])),
        }
    }
}

/// An `i128` as a Z3 integer literal. The arena holds `2^64` in a `wrap64`,
/// which does not fit an `i64`, so the decimal text is what crosses.
pub fn integer(n: i128) -> Int {
    Int::from_str(&n.to_string()).expect("a decimal integer is a Z3 numeral")
}

/// Python's floor division in Z3, whose integer `/` is Euclidean
/// (`print.rs::smt_floor_div`): the two agree for `k > 0`; for `k < 0`,
/// `floor(a/k) = floor(-a/-k)`, whose divisor is positive.
fn floor_div(child: &Int, k: i128) -> Int {
    if k > 0 {
        child.div(integer(k))
    } else {
        child.unary_minus().div(integer(-k))
    }
}

/// Python's `%` in Z3 (`print.rs::smt_floor_mod`).
fn floor_mod(child: &Int, k: i128) -> Int {
    if k > 0 {
        child.modulo(integer(k))
    } else {
        let quotient = child.unary_minus().div(integer(-k));
        Int::add(&[child.clone(), Int::mul(&[integer(-k), quotient])])
    }
}
