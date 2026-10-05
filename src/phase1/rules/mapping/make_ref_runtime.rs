//! `MAKE_REF_RUNTIME` (`revision/phase1/rules/mapping/make_ref_runtime.md`).
//!
//! The same views as `MAKE_REF` when the map is not constant: a slice start
//! that is a run-time term (R1), a negative index on an axis of run-time
//! extent (R2), or a broadcast whose "extent is 1" test is run-time (R5).
//! R5's parameter is the Boolean `¬(dims_u[a] = 1)`, which phase 1 builds
//! over `dims_u[a]` and `TERM_TO_WIRES` compiles.
//! Each parameter is a wire of π, answered by the node that computes it —
//! `TERM_TO_WIRES` for a slice start, `INDEX_NORMALIZE` for a normalised
//! index — and paid for there. It instantiates for a (v, owner) exactly when
//! a P1 question of `MAKE_REF` failed, whose answers it reads out of A.
//!
//! R4, a view stride that is not constant, is a parameter wire of its own:
//! `f.reshape(2, -1)` over a `Packed` of run-time length gives the key form
//! `π_0 × i_0 + i_1` with `π_0 = dims_w[1]`, and `APPLY_INDEX_MAP` charges
//! the product per entry. R3, an *owner* stride that is not constant, is not
//! offered: `MuForm::Coordinate` carries its strides as integers, so a
//! run-time one would be an interface change rather than a rule, and no
//! owner of the corpus has one (`mapping/README.md` §2.2).

use crate::phase1::build::Builder;
use crate::phase1::graph::Target;
use crate::phase1::interface::Family;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

use super::{MapParam, Runtime, Start};

pub(crate) struct MakeRefRuntime;

impl Rule for MakeRefRuntime {
    fn id(&self) -> RuleId {
        RuleId::MakeRefRuntime
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Ref]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(matched) = super::matches(builder, request) else {
            return Ok(Vec::new());
        };
        let mut probe = Candidate::new(RuleId::MakeRefRuntime, None, request.epoch);
        let under = builder.under(request.condition);
        let Some(parameters) =
            super::parameters(builder, &mut probe, &matched, request.epoch, under)
        else {
            return Ok(Vec::new());
        };
        // premises: at least one of R1 to R5 applies. R1, R2 and R5 are
        // visible here, as a parameter that is not a literal; R3 and R4 are
        // strides of the map, which `make_ref::build_refs` decides per owner
        // bundle and which drop this node where `MAKE_REF` built one.

        // One port per parameter wire, in π order; the map fragment of an
        // axis whose parameter is a wire is that wire.
        let mut starts: Vec<MapParam> = Vec::with_capacity(parameters.len());
        let mut terms: Vec<crate::phase0::term::TermId> = Vec::new();
        for start in &parameters {
            match start {
                Start::Literal(value) => starts.push(MapParam::Literal(*value)),
                Start::Runtime(runtime) => {
                    let Some(term) = self.port(builder, &mut probe, request, runtime)? else {
                        return Ok(Vec::new());
                    };
                    starts.push(MapParam::Wire(terms.len()));
                    terms.push(term);
                }
            }
        }

        super::make_ref::build_refs(
            builder,
            request,
            &matched,
            &mut probe,
            &starts,
            &terms,
            RuleId::MakeRefRuntime,
            /* flatten = */ false,
        )
    }
}

impl MakeRefRuntime {
    /// The port one run-time parameter is answered by, with the term its
    /// wire holds.
    fn port(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        request: &RowRequest,
        runtime: &Runtime,
    ) -> Result<Option<crate::phase0::term::TermId>, Reject> {
        let row = match runtime {
            // R1: `Represent(L, Wire[Idx])`, answered by `TERM_TO_WIRES`.
            Runtime::Start(term) => {
                let row =
                    builder.child(request, Target::Term(*term), Family::Wire, request.condition)?;
                match row.entries.first() {
                    Some(first) => {
                        candidate.inputs.push(first.requirement);
                        candidate.epoch = candidate.epoch.max(first.epoch);
                        return Ok(Some(*term));
                    }
                    None => return Ok(None),
                }
            }
            // R2: `Represent(⟨i, d⟩, IndexNorm)`, answered by
            // `INDEX_NORMALIZE`, whose `j` is the normalised position.
            Runtime::Index { index, extent } => builder.index_norm_port(
                request,
                index.clone(),
                *extent,
                request.condition,
            )?,
            // R5: the broadcast test `¬(dims_u[a] = 1)`, built over the
            // extent and answered by `TERM_TO_WIRES`.
            Runtime::Broadcast { extent } => {
                let one = builder.literal(1);
                let is_one = builder.terms.eq(*extent, one);
                let test = builder.terms.not(is_one);
                let row = builder.child(
                    request,
                    Target::BoolTerm(test),
                    Family::Wire,
                    request.condition,
                )?;
                match row.entries.first() {
                    Some(first) => {
                        candidate.inputs.push(first.requirement);
                        candidate.epoch = candidate.epoch.max(first.epoch);
                        return Ok(Some(*extent));
                    }
                    None => return Ok(None),
                }
            }
        };
        let term = match runtime {
            Runtime::Index { extent, .. } => *extent,
            _ => return Ok(None),
        };
        let Some(first) = row.entries.first() else {
            return Ok(None);
        };
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
        Ok(Some(term))
    }
}
