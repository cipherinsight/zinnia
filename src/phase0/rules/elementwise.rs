//! Elementwise operations and broadcasting: `definition.md` §5.3.

use super::helpers::broadcast_dims;
use super::Rule;
use super::{require_array, RuleCtx};
use crate::phase0::desc::{Desc, Dtype};
use crate::phase0::term::BoolId;
use crate::phase0::Reject;

/// `E-Map` (§5.3): an elementwise unary `f(a)`.
///
/// `domain` is the domain condition of `f`, which the caller supplies from
/// the operator table: true for wrapping integer arithmetic, `a >= 0` for
/// `sqrt` on a real domain, and so on. The shape is unchanged.
pub fn e_map(ctx: &mut RuleCtx, a: &Desc, out: Dtype, domain: BoolId) -> Result<Desc, Reject> {
    let arr = require_array(Rule::EMap, a)?;
    let dims = arr.dims.clone();
    Ok(ctx.array(Rule::EMap, out, dims, domain))
}

/// `E-Zip` (§5.3): an elementwise binary `a op b` with broadcasting.
///
/// There is no premise: ranks are equalised by prepending 1s. A scalar
/// operand is rank 0, so scalar broadcasting is the same rule (§8.1's
/// `s - p`). `out` is the operator's result dtype — the promotion of the
/// operand dtypes for arithmetic, `Bool` for a comparison.
///
/// Both operands scalar is `E-Arith` or `E-Cmp`, not this rule; the walker
/// dispatches on the descriptor kind.
pub fn e_zip(ctx: &mut RuleCtx, a: &Desc, b: &Desc, out: Dtype) -> Result<Desc, Reject> {
    let (da, db) = match (a.dims(), b.dims()) {
        (Some(x), Some(y)) => (x.to_vec(), y.to_vec()),
        _ => {
            return Err(Reject::Premise {
                rule: Rule::EZip,
                detail: format!("on a {} and a {}", a.kind_str(), b.kind_str()),
            })
        }
    };
    if a.as_array().is_none() && b.as_array().is_none() {
        return Err(Reject::Premise {
            rule: Rule::EZip,
            detail: "on two scalars (the walker dispatches those to E-Arith / E-Cmp)".to_string(),
        });
    }
    let (dims, defcond) = broadcast_dims(ctx.terms, &[&da, &db]);
    Ok(ctx.array(Rule::EZip, out, dims, defcond))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase0::tests::Fixture;

    /// §5.3: `f(a)` keeps the dims and records `defcond_f`.
    #[test]
    fn e_map_keeps_the_dims_and_records_the_domain_condition() {
        let mut f = Fixture::new();
        let a = Desc::Array(f.input(Dtype::Float, &[2, 3]));
        let tt = f.analysis.terms.tt();
        let total = e_map(&mut f.ctx(), &a, Dtype::Float, tt).unwrap();
        assert_eq!(f.dims_of(&total), vec![Some(2), Some(3)]);
        assert!(f.checks().is_empty(), "a total operator records nothing");
        let key = f.key();
        let domain = f.ctx().definedness("in_domain", &key);
        e_map(&mut f.ctx(), &a, Dtype::Float, domain).unwrap();
        assert_eq!(f.check_rules(), vec!["E-Map".to_string()]);
    }

    /// §5.3: `bjoin` equalises ranks by prepending 1s and joins each axis.
    #[test]
    fn e_zip_broadcasts_and_records_the_compatibility_condition() {
        let mut f = Fixture::new();
        let a = Desc::Array(f.input(Dtype::Float, &[3]));
        let b = Desc::Array(f.input(Dtype::Float, &[2, 3]));
        let built = e_zip(&mut f.ctx(), &a, &b, Dtype::Float).unwrap();
        assert_eq!(f.dims_of(&built), vec![Some(2), Some(3)]);
        assert!(f.checks().is_empty(), "equal extents fold the condition");
    }

    /// §5.3's premise: at least one operand is an array; two scalars are
    /// `E-Arith` / `E-Cmp`.
    #[test]
    fn e_zip_on_two_scalars_is_a_premise_violation() {
        let mut f = Fixture::new();
        let a = Desc::Scalar(f.literal(1));
        let b = Desc::Scalar(f.literal(2));
        let outcome = e_zip(&mut f.ctx(), &a, &b, Dtype::Idx);
        assert!(
            matches!(
                outcome,
                Err(Reject::Premise {
                    rule: Rule::EZip,
                    ..
                })
            ),
            "got {:?}",
            outcome.err()
        );
    }
}
