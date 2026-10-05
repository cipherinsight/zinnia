//! `range`: the iterable of `S-For-Unrolled` and `S-For-Range-Bounded`
//! (`definition.md` §6.2, `implementation.md` §8).
//!
//! A `range` call is read once, here, for both of its consumers: the builtin
//! ([`super::values`]), which materialises a list of constants the way the
//! baseline's `builtin_range` does, and the `for` statement
//! ([`super::loops`]), which also accepts the bounded form.
//!
//! The two forms differ in what the bounds are:
//!
//! - every bound a constant: [`RangePlan::Constant`], the trip count is a
//!   number and the elements are numbers — `S-For-Unrolled`, and the only
//!   form that is a value in its own right;
//! - a bound that is a term but not a constant: [`RangePlan::Bounded`], the
//!   trip count `T = ceil((t_hi - t_lo) / step)` is a term whose upper bound
//!   `C` comes from the ranges of $\Phi$
//!   ([`crate::phase0::interval::declared_maximum`]). Unrolling it needs the
//!   guard `j < T` on every iteration, so only a `for` can consume it; as an
//!   expression it is rejected, because a list of `C` elements is not what
//!   the source built.
//!
//! A bound with no usable maximum — an element value, whose only bound is
//! the int64 dtype range — is [`Reject::NonConstantRangeBound`], as §6.2
//! says: the baseline compiles it to zero iterations (`coverage.md` §9).

use crate::ast::{ASTNamedAttribute, ASTNode};
use crate::phase0::desc::{Desc, Dtype, Origin, ScalarTerm};
use crate::phase0::interval::declared_maximum;
use crate::phase0::rules::helpers::constant_range_len;
use crate::phase0::rules::{range_len, require_int_term, require_scalar};
use crate::phase0::term::TermId;
use crate::phase0::Reject;

use super::calls::Args;
use super::Walker;
use crate::phase0::rules::Rule;

/// The largest `range` phase 0 materialises as a list of constants. A
/// constant `range` is a composite (`coverage.md` §5.4), so an enormous one
/// would exhaust memory before it could say anything useful; the baseline has
/// the same shape of limit in `loop_limit`. The bounded form is capped by the
/// work budget instead, which is the number of iterations it unrolls.
const MAX_RANGE: i128 = 1 << 22;

/// How a `range` call is unrolled. Both forms produce the value
/// `start + j * step` for `j` in `0..count` / `0..cap`; they differ in
/// whether that many iterations is the trip count or only a bound on it.
pub(super) enum RangePlan {
    /// Every bound is a constant: `count` is the trip count
    /// (`S-For-Unrolled`), and the `range` is also a value in its own right.
    Constant {
        /// `t_lo`, the first value, itself a constant term.
        start: TermId,
        /// The stride, a non-zero constant.
        step: i128,
        /// The trip count.
        count: usize,
    },
    /// The trip count is a term (`S-For-Range-Bounded`).
    Bounded {
        /// `t_lo`, the first value.
        start: TermId,
        /// The stride, a non-zero constant.
        step: i128,
        /// `T`, the trip count.
        trip: TermId,
        /// `C`, the number of iterations to unroll.
        cap: usize,
    },
}

/// The `range(...)` call a `for` statement iterates, when that is what its
/// iterable is.
///
/// §6.2 states `S-For-Unrolled` over the syntactic form `for i in range(…)`,
/// and so does `S-For-Range-Bounded`: the bounded form is a loop, not a
/// value. A chip named `range` shadows the builtin, exactly as in
/// [`super::calls::Walker::bare_call`].
pub(super) fn as_range_call<'a>(
    w: &Walker,
    iterable: &'a ASTNode,
) -> Option<&'a ASTNamedAttribute> {
    match iterable {
        ASTNode::ASTNamedAttribute(call)
            if call.target.is_none() && call.member == "range" && !w.is_chip("range") =>
        {
            Some(call)
        }
        _ => None,
    }
}

impl Walker<'_> {
    /// Read the evaluated arguments of a `range` call (§6.2).
    ///
    /// `range(stop)`, `range(start, stop)` and `range(start, stop, step)`,
    /// with the same defaults Python gives them. The step must be a constant:
    /// §3.1 has no division by a variable, so a runtime step leaves the trip
    /// count outside the term language.
    pub(super) fn range_plan(&mut self, args: &Args) -> Result<RangePlan, Reject> {
        let bounds = self.range_bounds(args)?;
        let (start, stop, step) = match bounds[..] {
            [stop] => (None, stop, 1),
            [start, stop] => (Some(start), stop, 1),
            [start, stop, step] => (Some(start), stop, self.range_step(step)?),
            _ => {
                return Err(Reject::Unsupported {
                    construct: format!("`range` with {} arguments", bounds.len()),
                })
            }
        };
        let zero = self.an.terms.constant(0);
        let start = start.unwrap_or(zero);
        match (
            self.an.terms.constant_value(start),
            self.an.terms.constant_value(stop),
        ) {
            (Some(lo), Some(hi)) => Ok(RangePlan::Constant {
                start,
                step,
                count: constant_trip_count(lo, hi, step)?,
            }),
            _ => self.bounded_plan(start, stop, step),
        }
    }

    /// `S-For-Range-Bounded`: the trip count as a term, and the cap `C` that
    /// $\Phi$'s ranges give it.
    fn bounded_plan(
        &mut self,
        start: TermId,
        stop: TermId,
        step: i128,
    ) -> Result<RangePlan, Reject> {
        let trip = range_len(&mut self.an.terms, start, stop, step as i64).ok_or_else(|| {
            Reject::Unsupported {
                construct: "`range` with step 0".to_string(),
            }
        })?;
        let budget = self.config.work_budget as i128;
        let cap = declared_maximum(&self.an.terms, &self.an.phi, trip, budget).map_err(|why| {
            Reject::NonConstantRangeBound {
                detail: format!(
                    "the trip count `{}` {}",
                    self.an.terms.pretty(trip),
                    why.describe()
                ),
            }
        })?;
        let cap = usize::try_from(cap.max(0)).map_err(|_| Reject::NonConstantRangeBound {
            detail: format!(
                "a trip count of {} is not a usable number of iterations",
                cap
            ),
        })?;
        Ok(RangePlan::Bounded {
            start,
            step,
            trip,
            cap,
        })
    }

    /// The bound terms of a `range` call, left to right.
    ///
    /// A bound has to carry an integer term at all: a `Float` scalar carries
    /// none (§3.2), so `range(TSTEPS)` with a `Float` parameter is where the
    /// rejection message names the annotation to fix.
    fn range_bounds(&mut self, args: &Args) -> Result<Vec<TermId>, Reject> {
        let mut bounds = Vec::with_capacity(args.pos.len());
        for (index, argument) in args.pos.iter().enumerate() {
            let scalar = require_scalar(Rule::SForRangeBounded, argument).map_err(|_| {
                Reject::NonConstantRangeBound {
                    detail: format!(
                        "argument {} of `range` is a {}, not a scalar",
                        index,
                        argument.kind_str()
                    ),
                }
            })?;
            bounds.push(
                require_int_term(Rule::SForRangeBounded, scalar).map_err(|_| {
                    Reject::NonConstantRangeBound {
                        detail: format!(
                            "argument {} of `range` is a {} scalar, which carries no term",
                            index,
                            scalar.dtype.as_str()
                        ),
                    }
                })?,
            );
        }
        Ok(bounds)
    }

    /// The stride of a `range`, which §3.1 requires to be a constant.
    fn range_step(&self, step: TermId) -> Result<i128, Reject> {
        self.an
            .terms
            .constant_value(step)
            .ok_or_else(|| Reject::NonConstantRangeBound {
                detail: "the step of a `range` is not a constant".to_string(),
            })
    }

    /// The `j`-th value of a `range`: the shape scalar `t_lo + j * step`
    /// (§6.2, `S-For-Unrolled`'s `i`, and `S-For-Range-Bounded`'s).
    ///
    /// It is built inside iteration `j`, so its $\psi$ is that iteration's
    /// path condition — the bounded form's guard `j < T` included (§2).
    ///
    /// The rule is `E-Const` and not the loop's: the value is the constant
    /// `t_lo + j*step`, and the loop rule is the statement's. The element is
    /// an intermediate of the `range` call (§3.3), so it is a value of
    /// $\Delta$ with a row phase 1 can answer rather than a binding with an
    /// empty operand list.
    pub(super) fn range_element(&mut self, start: TermId, step: i128, j: usize) -> Desc {
        let offset = self.an.terms.constant(step * j as i128);
        let value = self.an.terms.add(start, offset);
        let tt = self.ctx().tt();
        self.ctx().scalar(
            Rule::EConst,
            Dtype::Idx,
            ScalarTerm::Int(value),
            Origin::Shape,
            tt,
        )
    }
}

/// The number of values `range(lo, hi, step)` has when every bound is a
/// constant.
fn constant_trip_count(lo: i128, hi: i128, step: i128) -> Result<usize, Reject> {
    let count = constant_range_len(lo, hi, step).ok_or_else(|| Reject::Unsupported {
        construct: "`range` with step 0".to_string(),
    })?;
    if count > MAX_RANGE {
        // A limit on how much phase 0 will materialise, like the work
        // budget, and not a construct §5 has no rule for.
        return Err(Reject::BudgetExceeded {
            units: count.max(0) as u64,
            budget: MAX_RANGE as u64,
        });
    }
    Ok(count as usize)
}
