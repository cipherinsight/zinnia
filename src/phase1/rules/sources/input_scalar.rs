//! `INPUT_SCALAR` (`revision/phase1/rules/sources/input_scalar.md`).
//!
//! Binds a scalar parameter to one wire and enforces its admitted domain.
//! Phase 0 makes a scalar input a data scalar whose variable carries the
//! `var_range` [a, b]: the dtype range tightened by the `@requires` facts
//! that bound the parameter alone. The rule produces `Wire[τ]` over the cell
//! the front end binds, with [a, b] enforced. It has no port, and for a
//! scalar `Represent(x, Wire[τ])` and `Fetch(x, [()], g)` are one
//! requirement (§3.2), so this is the only row of x.

use crate::phase0::desc::ScalarTerm;
use crate::phase0::source::Producer;

use crate::phase1::build::{bits_for, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct InputScalar;

impl Rule for InputScalar {
    fn id(&self) -> RuleId {
        RuleId::InputScalar
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        // source_pattern: a parameter whose annotation is a scalar type,
        // with the producer `Input` and a scalar descriptor.
        let Target::Value(key) = &request.target else {
            return Ok(Vec::new());
        };
        let Some(entry) = builder.entry(key) else {
            return Ok(Vec::new());
        };
        if !matches!(entry.producer, Producer::Input) {
            return Ok(Vec::new());
        }
        let Some(scalar) = entry.desc.as_scalar() else {
            return Ok(Vec::new());
        };
        let (dtype, term) = (scalar.dtype, scalar.term);
        if !super::has_encoding(dtype) {
            return Ok(Vec::new());
        }
        let Some(parameter) = builder.input_node(key) else {
            return Ok(Vec::new());
        };
        let private = super::is_private(&parameter.annotation);
        let hashed = super::declared(&parameter.annotation)
            .map(|(_, hashed)| hashed)
            .unwrap_or(false);

        // A Float parameter is one wire in the backend's encoding: it
        // carries no term, has no interval and takes no range check, and the
        // front end binds it like any other cell (§2.3). The hash cell is
        // the only gate it can carry.
        if super::is_float(dtype) {
            let mut candidate =
                Candidate::new(RuleId::InputScalar, Some("wire"), request.epoch).output(
                    request.target.clone(),
                    Interface::Wire(dtype),
                    request.condition,
                );
            candidate.matched.push(key.clone());
            if hashed {
                candidate = candidate.emit(Primitive::Hash { cells: 1 }, 1);
            }
            return Ok(vec![candidate]);
        }

        // resource_relations: [a, b] from S, through the `var_range` of the
        // parameter's variable; w = ⌈log2(b − a + 1)⌉ by the width policy.
        let (lo, hi) = match term {
            ScalarTerm::Bool(_) => (Some(0), Some(1)),
            ScalarTerm::Int(t) => match builder.terms.var_of(t) {
                Some(variable) => builder.var_range(variable),
                // A scalar input whose term is not a bare variable is not
                // this rule's: phase 0 made it a shape scalar.
                None => return Ok(Vec::new()),
            },
            ScalarTerm::Opaque => return Ok(Vec::new()),
        };
        // premises: for an integer τ the interval is finite. Without it the
        // width is undefined and the node is not admitted (§6.1).
        let (Some(lo), Some(hi)) = (lo, hi) else {
            return Ok(Vec::new());
        };
        let width = builder.omega.round(bits_for(hi - lo + 1));
        if !builder.omega.admits(width) {
            return Ok(Vec::new());
        }

        let epoch = request.epoch;
        let constant = lo == hi;
        let mut candidate = Candidate::new(
            RuleId::InputScalar,
            Some(if constant { "const" } else { "wire" }),
            epoch,
        )
        .output(
            request.target.clone(),
            Interface::Wire(dtype),
            request.condition,
        )
        .parameter("a", lo)
        .parameter("b", hi)
        .parameter("w", width as i128);
        candidate.matched.push(key.clone());

        // emitter: (S1) the range checks, (S2) the pinned constant, (S3) the
        // hash cell.
        if !constant && (private || builder.omega.range_check_public) {
            if matches!(term, ScalarTerm::Bool(_)) {
                candidate = candidate.emit(Primitive::Bool, 1);
            } else {
                candidate = candidate.emit(Primitive::Range { width }, 1);
                // The second check is exact only when the interval is not a
                // full power of two.
                if hi - lo + 1 != 1i128 << width {
                    candidate = candidate.emit(Primitive::Range { width }, 1);
                }
            }
        }
        if hashed {
            candidate = candidate.emit(Primitive::Hash { cells: 1 }, 1);
        }
        Ok(vec![candidate])
    }
}
