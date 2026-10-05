//! `ELEMENTWISE_BATCH`
//! (`revision/phase1/rules/elementwise/elementwise_batch.md`).
//!
//! Computes an elementwise result at the entries of a batch without building
//! a representation of it: it fetches every array operand at the coordinate
//! broadcasting assigns to each entry and applies the primitive of
//! `map_aligned.md`'s table to the fetched results. It is the route of a
//! consumer that reads v rather than represents it — the fold over
//! `(s − p)·(s − p)` that materialises neither d nor e — and, through
//! `BUILD_DENSE`, the route to a `Store` of v when no operand bundles are
//! aligned. A chain of elementwise operations fuses into one per-entry
//! computation, because each operand `Fetch` may itself be answered here.

use crate::phase0::term::TermId;

use crate::phase1::build::Builder;
use crate::phase1::graph::Target;
use crate::phase1::interface::{Batch, Coordinate, Family, Interface, RangeAxis};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::{Condition, Question};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct ElementwiseBatch;

impl Rule for ElementwiseBatch {
    fn id(&self) -> RuleId {
        RuleId::ElementwiseBatch
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Values, Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        // The `scalar` variant: a rank-0 result is a scalar, and its
        // one-entry `Values` is the `Wire` of `rules/README.md` §3.2.
        if request.family == Family::Wire {
            return self.scalar(builder, request);
        }
        let (Some(key), Some(batch)) = (request.target.as_value(), request.batch.clone()) else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        let Some(mut matched) = super::matches(builder, &key) else {
            return Ok(Vec::new());
        };
        let mut candidate = Candidate::new(RuleId::ElementwiseBatch, None, request.epoch);
        let under = builder.under(request.condition);
        super::resolve_exponent(builder, &mut candidate, &mut matched, request.epoch, under);
        let Some(gadget) = super::gadget(builder, &matched) else {
            builder.missing.push(format!(
                "ELEMENTWISE_BATCH has no row for the primitive `{}` on {}",
                matched.primitive,
                matched.dtype.as_str()
            ));
            return Ok(Vec::new());
        };
        candidate.matched = std::iter::once(key.clone())
            .chain(matched.arrays.iter().cloned())
            .collect();

        for operand in &matched.arrays {
            let Some(projected) = self.project(
                builder,
                &mut candidate,
                &batch,
                &matched.dims,
                operand,
                request.epoch,
                under,
            ) else {
                return Ok(Vec::new());
            };
            let row = builder.fetch_port(&mut candidate, request, operand, &projected)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        for operand in &matched.scalars {
            let psi = builder.psi(operand);
            let condition =
                builder.child_under_g(&mut candidate, request, psi, &operand.encode())?;
            let row =
                builder.child(request, Target::Value(operand.clone()), Family::Wire, condition)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }

        // emitter: one gadget per entry, active or not (§5).
        let queries = batch.len();
        for (primitive, times) in gadget {
            candidate = candidate.emit(primitive, times * queries);
        }
        let Some(interface) = super::super::reading::values(builder, &key, &batch) else {
            return Ok(Vec::new());
        };
        candidate = candidate
            .answers_fetch(
                request.target.clone(),
                batch.clone(),
                interface,
                request.condition,
            )
            .parameter("Q", queries);
        Ok(vec![candidate])
    }
}

impl ElementwiseBatch {
    /// The `scalar` variant (`elementwise_batch.md`): an elementwise
    /// operation whose result has rank 0.
    ///
    /// Every operand is then a scalar operand, read through
    /// `Represent(p_j, Wire[τ_j])`, the batch is the one entry `[()]`, and
    /// the `Values` of that batch is the result's own wire
    /// (`rules/README.md` §3.2). One gadget of the table, not Q of them.
    fn scalar(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(key) = request.target.as_value() else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        let Some(mut matched) = super::matches_scalar(builder, &key) else {
            return Ok(Vec::new());
        };
        let mut candidate =
            Candidate::new(RuleId::ElementwiseBatch, Some("scalar"), request.epoch);
        let under = builder.under(request.condition);
        super::resolve_exponent(builder, &mut candidate, &mut matched, request.epoch, under);
        let Some(gadget) = super::gadget(builder, &matched) else {
            builder.missing.push(format!(
                "ELEMENTWISE_BATCH has no row for the primitive `{}` on {}",
                matched.primitive,
                matched.dtype.as_str()
            ));
            return Ok(Vec::new());
        };
        candidate.matched = std::iter::once(key.clone())
            .chain(matched.scalars.iter().cloned())
            .collect();
        for operand in &matched.scalars {
            let psi = builder.psi(operand);
            let condition =
                builder.child_under_g(&mut candidate, request, psi, &operand.encode())?;
            let row =
                builder.child(request, Target::Value(operand.clone()), Family::Wire, condition)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        for (primitive, times) in gadget {
            candidate = candidate.emit(primitive, times);
        }
        candidate = candidate
            .output(
                request.target.clone(),
                Interface::Wire(matched.dtype),
                request.condition,
            )
            .parameter("Q", 1);
        Ok(vec![candidate])
    }

    /// `π_j(batch)`, the batch one array operand is fetched at
    /// (`elementwise_batch.md`, premises).
    ///
    /// Per output axis k the operand axis is `k′ = k − (r_v − r_j)`: case 1
    /// keeps the template when PROVE says the extents agree, case 2 reads
    /// position 0 when CONST says the operand's extent is 1. Case 3, an
    /// undecided broadcast, is not offered here: its coordinate
    /// `ite(dims_j[k′] = 1, 0, t + s·j)` is a term (`RangeAxis.broadcast`),
    /// but its liveness is not one thermometer per axis, which is what a
    /// range batch's flags are (`rules/README.md` §3.3), so the operand is
    /// read through `BROADCAST_MAP` instead.
    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        builder: &mut Builder<'_>,
        candidate: &mut Candidate,
        batch: &Batch,
        dims: &[TermId],
        operand: &crate::phase0::desc::ValueKey,
        epoch: u32,
        under: Condition,
    ) -> Option<Batch> {
        let operand_dims = builder.array(operand).map(|array| array.dims.clone())?;
        let offset = dims.len().checked_sub(operand_dims.len())?;
        match batch {
            Batch::Scalar => Some(Batch::Scalar),
            Batch::Range { axes, live } => {
                let mut projected = Vec::with_capacity(operand_dims.len());
                for (axis, template) in axes.iter().enumerate() {
                    if axis < offset {
                        continue;
                    }
                    let operand_axis = axis - offset;
                    // Case 1: the extents agree, so the template is the same.
                    if builder.prove_eq(
                        candidate,
                        operand_dims[operand_axis],
                        dims[axis],
                        epoch,
                        under,
                    ) {
                        projected.push(template.clone());
                        continue;
                    }
                    // Case 2: the operand's extent is 1, so it is read at 0.
                    match builder.ask_for(
                        candidate,
                        Question::Const(operand_dims[operand_axis]),
                        epoch,
                        under,
                    ) {
                        Verdict::Constant(1) => projected.push(RangeAxis {
                            base: Coordinate::Literal(0),
                            stride: 0,
                            width: template.width,
                            broadcast: None,
                        }),
                        _ => return None,
                    }
                }
                Some(Batch::Range { axes: projected, live: *live })
            }
            // A general batch keeps its entries; only a rank-equal operand is
            // offered, because a projected entry would need a new coordinate.
            Batch::General(_) if operand_dims.len() == dims.len() => Some(batch.clone()),
            Batch::General(_) => None,
        }
    }
}

/// The `Values` interface of a batch, for a caller outside this rule.
#[allow(dead_code)]
pub(crate) fn values(interface: &Interface) -> Option<&Interface> {
    matches!(interface, Interface::Values { .. }).then_some(interface)
}
