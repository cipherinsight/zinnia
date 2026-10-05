//! `SHAPE_CHECK` (`revision/phase1/rules/scalar_control/shape_check.md`).
//!
//! Gives a wire for a product of run-time extents, the one shape quantity the
//! term language cannot write: when two or more axes of a value have
//! non-constant extents, phase 0 introduces σ with the product fact
//! `σ = ∏_k dims[k]`, and `E-Reshape` with a non-constant `−1` divisor
//! records `size(a) = ℓ·p` through such a fact. The circuit multiplies the
//! run-time factors exactly, with no range decomposition: P1 keeps the
//! product below the field.

use crate::phase0::desc::Dtype;
use crate::phase0::term::VarId;

use crate::phase1::build::{width_of, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ShapeCheck;

impl Rule for ShapeCheck {
    fn id(&self) -> RuleId {
        RuleId::ShapeCheck
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        // source_pattern: a product fact `σ = ∏_j f_j` of Φ. The row is σ's
        // own term requirement, or the row of the scalar that carries it —
        // `a.size` of a value with two or more run-time axes, whose row and
        // its term's are one (`rules/README.md` §3.2).
        let term = match &request.target {
            Target::Term(term) => Some(*term),
            Target::Value(key) => builder.scalar(key).and_then(|scalar| scalar.term.as_int()),
            Target::BoolTerm(_) => None,
        };
        let Some(term) = term else {
            return Ok(Vec::new());
        };
        let Some(variable) = builder.terms.var_of(term) else {
            return Ok(Vec::new());
        };
        let Some(factors) = self.factors(builder, variable) else {
            return Ok(Vec::new());
        };
        let runtime: Vec<crate::phase0::term::TermId> = factors
            .iter()
            .copied()
            .filter(|&factor| builder.terms.constant_value(factor).is_none())
            .collect();
        // P2 (structural): two or more non-constant factors; with one the
        // product is `k·f`, a term, and `TERM_TO_WIRES`'s.
        if runtime.len() < 2 {
            return Ok(Vec::new());
        }
        // P1 (width): the widths sum below `Ω.max_width`, so the product
        // cannot wrap in the field.
        let width: u32 = runtime
            .iter()
            .map(|&factor| width_of(builder.interval(factor)))
            .sum();
        if !builder.omega.admits(width) {
            return Ok(Vec::new());
        }

        let mut candidate = Candidate::new(RuleId::ShapeCheck, None, request.epoch);
        if let Target::Value(key) = &request.target {
            candidate.matched.push(key.clone());
        }
        for factor in &runtime {
            let row =
                builder.child(request, Target::Term(*factor), Family::Wire, request.condition)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        // emitter: `z_{j+1} = mul(z_j, f_{j+1})`, m − 1 products.
        candidate = candidate
            .emit(Primitive::Mul, runtime.len() as i128 - 1)
            .output(
                request.target.clone(),
                Interface::Wire(Dtype::Idx),
                request.condition,
            )
            .parameter("w", width as i128);
        Ok(vec![candidate])
    }
}

impl ShapeCheck {
    /// The factors of the product fact a variable is the subject of.
    fn factors(
        &self,
        builder: &Builder<'_>,
        variable: VarId,
    ) -> Option<Vec<crate::phase0::term::TermId>> {
        builder
            .result
            .phi
            .products()
            .iter()
            .find(|fact| fact.var == variable)
            .map(|fact| fact.factors.clone())
    }
}
