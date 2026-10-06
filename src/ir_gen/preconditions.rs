//! What a circuit may assume from its scalar preconditions (`@requires`).
//!
//! A precondition is asserted in the circuit (so a prover cannot violate
//! it) and also recorded as facts for the optimizer and the bound
//! inference. The facts are stated in the shapes their consumers read:
//! `optim::resolver::derive_bounds_from_facts` looks for `value <op>
//! literal` comparisons, so a conjunction is recorded as its conjuncts and
//! literal arithmetic is evaluated (a precondition spells `-3` as `0 - 3`).

use crate::optim::predicates::formula::{ArithOp, BoolOp, ContractTerm};

/// The value of an integer literal term, including literal arithmetic such
/// as `0 - 3`.
pub(super) fn literal_int(term: &ContractTerm) -> Option<i64> {
    match term {
        ContractTerm::LitInt(n) => Some(*n),
        ContractTerm::Arith { op, lhs, rhs } => {
            let (a, b) = (literal_int(lhs)?, literal_int(rhs)?);
            match op {
                ArithOp::Add => a.checked_add(b),
                ArithOp::Sub => a.checked_sub(b),
                ArithOp::Mul => a.checked_mul(b),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The facts a precondition states: its conjuncts (each holds whenever the
/// conjunction does), with every integer-literal subterm evaluated. For
/// `-3 <= n <= 5` these are `-3 <= n` and `n <= 5`, which give `n` a
/// two-sided bound.
pub(super) fn precondition_facts(term: &ContractTerm) -> Vec<ContractTerm> {
    match term {
        ContractTerm::BoolComb { op: BoolOp::And, operands } => {
            operands.iter().flat_map(precondition_facts).collect()
        }
        other => vec![evaluate_literals(other)],
    }
}

/// `term` with every integer-literal arithmetic subterm replaced by its value.
fn evaluate_literals(term: &ContractTerm) -> ContractTerm {
    if let Some(n) = literal_int(term) {
        return ContractTerm::LitInt(n);
    }
    let boxed = |t: &ContractTerm| Box::new(evaluate_literals(t));
    match term {
        ContractTerm::Arith { op, lhs, rhs } => ContractTerm::Arith { op: *op, lhs: boxed(lhs), rhs: boxed(rhs) },
        ContractTerm::Cmp { op, lhs, rhs } => ContractTerm::Cmp { op: *op, lhs: boxed(lhs), rhs: boxed(rhs) },
        ContractTerm::BoolComb { op, operands } => ContractTerm::BoolComb {
            op: *op,
            operands: operands.iter().map(evaluate_literals).collect(),
        },
        ContractTerm::Not(inner) => ContractTerm::Not(boxed(inner)),
        ContractTerm::PredicateApp { kind, args } => ContractTerm::PredicateApp {
            kind: kind.clone(),
            args: args.iter().map(evaluate_literals).collect(),
        },
        leaf => leaf.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::predicates::formula::CmpOp;

    fn negative_three() -> ContractTerm {
        ContractTerm::Arith {
            op: ArithOp::Sub,
            lhs: Box::new(ContractTerm::LitInt(0)),
            rhs: Box::new(ContractTerm::LitInt(3)),
        }
    }

    #[test]
    fn a_two_sided_bound_becomes_two_literal_comparisons() {
        let n = || Box::new(ContractTerm::var_in("n"));
        let term = ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp { op: CmpOp::Le, lhs: Box::new(negative_three()), rhs: n() },
                ContractTerm::Cmp { op: CmpOp::Le, lhs: n(), rhs: Box::new(ContractTerm::LitInt(5)) },
            ],
        };
        assert_eq!(
            precondition_facts(&term),
            vec![
                ContractTerm::Cmp { op: CmpOp::Le, lhs: Box::new(ContractTerm::LitInt(-3)), rhs: n() },
                ContractTerm::Cmp { op: CmpOp::Le, lhs: n(), rhs: Box::new(ContractTerm::LitInt(5)) },
            ]
        );
    }

    #[test]
    fn seeded_facts_give_a_two_sided_bound() {
        use crate::optim::predicates::formula::ContractVar;
        let vid = crate::types::ValueId::next();
        let n = || Box::new(ContractTerm::Var(ContractVar::Value(vid)));
        let precondition = ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp { op: CmpOp::Le, lhs: Box::new(negative_three()), rhs: n() },
                ContractTerm::Cmp { op: CmpOp::Le, lhs: n(), rhs: Box::new(ContractTerm::LitInt(5)) },
            ],
        };
        let mut facts = crate::optim::predicates::FactStack::new();
        facts.insert_for(vid, precondition.clone());
        // The conjunction as one fact, with `0 - 3`, bounds nothing.
        assert_eq!(crate::optim::resolver::derive_bounds_from_facts(&facts, vid), None);
        let mut facts = crate::optim::predicates::FactStack::new();
        for fact in precondition_facts(&precondition) {
            facts.insert_for(vid, fact);
        }
        assert_eq!(crate::optim::resolver::derive_bounds_from_facts(&facts, vid), Some((-3, 5)));
    }

    #[test]
    fn non_literal_arithmetic_is_kept() {
        let sum = ContractTerm::Arith {
            op: ArithOp::Add,
            lhs: Box::new(ContractTerm::var_in("n")),
            rhs: Box::new(negative_three()),
        };
        assert_eq!(
            precondition_facts(&sum),
            vec![ContractTerm::Arith {
                op: ArithOp::Add,
                lhs: Box::new(ContractTerm::var_in("n")),
                rhs: Box::new(ContractTerm::LitInt(-3)),
            }]
        );
    }
}
