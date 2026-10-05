//! `TERM_TO_WIRES` (`revision/phase1/rules/scalar_control/term_to_wires.md`).
//!
//! Compiles a term of Φ into one wire, with one port per variable and one
//! gadget per term node. It answers the term requirements the library raises
//! for a term that is the term of no value: a check's φ_k and defcond_k, a
//! dimension term, a slice endpoint, a merge condition that is a shape term.
//! A sub-term that is the term of a value of Δ is that value's row
//! (`rules/README.md` §3.2), which the provider table resolves before this
//! rule is reached; two occurrences of one term id are one wire, because the
//! row is memoised on the term id.
//!
//! A variable is resolved to the value that introduced it, and to the
//! element inside that value when the value is a composite (§2). Step 1 has
//! a provider for a data scalar, a data Boolean and an input extent; a count
//! `c`, an `ℓ`, a `c′` or a `σ` is reported as having no rule yet, which is
//! what the missing `COUNT`, `LEN_SHAPE` and `SHAPE_CHECK` providers of
//! later steps will fill in.

use crate::phase0::desc::Dtype;
use crate::phase0::phi::VarOrigin;
use crate::phase0::term::NodeView;

use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::graph::{ReqId, Target};
use crate::phase1::interface::{Family, Interface};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct TermToWires;

/// The predicate one domain of `../../phase0/ops.rs` is, in the primitives of
/// `rules/README.md` §7: a float comparison on a Float operand and a `cmp_w`
/// on an integer one. `Domain::All` needs no wire — phase 0 records `true`
/// there and mints no variable.
fn domain_gadget(
    builder: &Builder<'_>,
    domain: crate::phase0::ops::Domain,
    dtype: Dtype,
) -> Option<Vec<(crate::phase1::cost::Primitive, i128)>> {
    use crate::phase0::ops::Domain;
    use crate::phase1::cost::{FloatOp, Primitive};
    if !super::super::sources::has_encoding(dtype) {
        return None;
    }
    let float = super::super::sources::is_float(dtype);
    let compare = |above: FloatOp| -> (Primitive, i128) {
        if float {
            (Primitive::Float(above), 1)
        } else {
            (Primitive::Cmp { width: builder.omega.round(builder.dtype_width(dtype)) }, 1)
        }
    };
    Some(match domain {
        // `x ≥ 0` and `x > 0`: one comparison.
        Domain::NonNegative => vec![compare(FloatOp::Gte)],
        Domain::Positive => vec![compare(FloatOp::Gt)],
        // `−1 ≤ x ≤ 1`: two comparisons and their conjunction.
        Domain::UnitInterval => {
            vec![compare(FloatOp::Gte), compare(FloatOp::Lte), (Primitive::And, 1)]
        }
        Domain::All => return None,
    })
}

impl Rule for TermToWires {
    fn id(&self) -> RuleId {
        RuleId::TermToWires
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let (index, dtype) = match &request.target {
            Target::Term(t) => (t.index() as u32, Dtype::Idx),
            Target::BoolTerm(b) => (b.index() as u32, Dtype::Bool),
            // A value's wire is the value's own rule's.
            Target::Value(_) => return Ok(Vec::new()),
        };
        let node = builder.terms.node_at(index as usize);
        let mut candidate = Candidate::new(RuleId::TermToWires, None, request.epoch);

        // P3: a sub-term of the form `−d ≤ t ∧ t < d` is
        // `INDEX_NORMALIZE`'s row, which computes the same bit with two
        // comparisons instead of three and names it together with the
        // normalised position. When that row exists it is used.
        if let Target::BoolTerm(formula) = &request.target {
            if super::index_normalize::in_bounds_pattern(builder, *formula).is_some() {
                return Ok(Vec::new());
            }
        }
        match node {
            // A constant is substituted at every use: no wire, no port.
            NodeView::Const(_) | NodeView::True | NodeView::False => {}
            NodeView::Var(variable) | NodeView::BVar(variable) => {
                let origin = builder.var_origin(variable).clone();
                match origin {
                    VarOrigin::DataScalar { .. } | VarOrigin::DataBool { .. } => {
                        // §2: the value that introduced the variable, and the
                        // element inside it when that value is a composite.
                        let Some((value, path)) = builder.variable_origin(index) else {
                            return Ok(Vec::new());
                        };
                        let Some(target) = builder.element_target(&value, &path) else {
                            return Ok(Vec::new());
                        };
                        // The element is this very term, so the value's own
                        // rule answers the row and this one offers nothing:
                        // a port back to the row being filled would be the
                        // cycle of §4.
                        if target == request.target {
                            return Ok(Vec::new());
                        }
                        let psi = builder.element_psi(&value, &path);
                        let condition = builder.child_under_g(
                            &mut candidate,
                            request,
                            psi,
                            &target.render(),
                        )?;
                        let row = builder.child(request, target, Family::Wire, condition)?;
                        let Some(first) = row.entries.first() else {
                            return Ok(Vec::new());
                        };
                        candidate.inputs.push(first.requirement);
                        candidate.epoch = candidate.epoch.max(first.epoch);
                    }
                    // β_I, the in-range condition of an index array: the
                    // `β_I` row of the rule file, one `INDEX_NORMALIZE` ok
                    // gadget per physical position of I, folded with `and`.
                    VarOrigin::Definedness { value } => {
                        if self.in_range_row(builder, request, &mut candidate, &value)? {
                            let primitives = self.in_range_gadgets(builder, &value);
                            for (primitive, times) in primitives {
                                candidate = candidate.emit(primitive, times);
                            }
                            candidate = candidate.output(
                                request.target.clone(),
                                Interface::Wire(Dtype::Bool),
                                request.condition,
                            );
                            return Ok(vec![candidate]);
                        }
                        // The in-range condition of `ravel_multi_index`
                        // over a run-time `dims`, which constrains two
                        // arrays rather than one operand's elements.
                        let ravel = self.ravel_domain_row(builder, request, &value)?;
                        if !ravel.is_empty() {
                            return Ok(ravel);
                        }
                        // The other definedness variable is the domain
                        // Boolean of a `math.*` function whose operand
                        // carries no term (`term_to_wires.md`, the `domain`
                        // row).
                        return self.domain_row(builder, request, &value);
                    }
                    // Every other variable is sized by a value, and
                    // `LEN_SHAPE`, `COUNT` or `SHAPE_CHECK` answers it. The
                    // first of those exists in step 1 and matches the term
                    // row itself, so this rule offers nothing here; the row
                    // note names the rule that would (`build.rs`,
                    // `provenance`).
                    _ => return Ok(Vec::new()),
                }
            }
            other => {
                // One port per sub-term, under the same g: §3.2 makes every
                // sub-term its own term requirement, memoised on its id.
                for child in crate::phase1::oracle::layers::children_of(other) {
                    let Some(port) = self.child_wire(builder, request, child)? else {
                        return Ok(Vec::new());
                    };
                    candidate.inputs.push(port.0);
                    candidate.epoch = candidate.epoch.max(port.1);
                }
            }
        }

        // emitter: one row per term node on the children's wires.
        let primitives = super::node_gadget(builder, index);
        if !super::widths_admitted(builder, &primitives) {
            return Ok(Vec::new());
        }
        for (primitive, times) in primitives {
            candidate = candidate.emit(primitive, times);
        }
        candidate = candidate.output(
            request.target.clone(),
            Interface::Wire(dtype),
            request.condition,
        );
        Ok(vec![candidate])
    }
}

impl TermToWires {
    /// The `ravel` domain row: the in-range condition
    /// `⋀_k (0 ≤ index_k ∧ index_k < dims_k)` of
    /// `np.ravel_multi_index(index, dims, order)` over a run-time `dims`
    /// (`term_to_wires.md`, the `domain` row; `../../phase0/definition.md`
    /// §5.4, `E-Ravel`).
    ///
    /// Phase 0 mints the condition as an unconstrained Boolean of the
    /// `E-Take` kind, because the coordinates and the extents are element
    /// values it does not model. The wire is the condition itself, read off
    /// the two arrays `SCALAR_OP`'s `/ravel` row already fetches: two
    /// comparisons per axis and a fold with `and`. Empty when the value is
    /// no such call.
    fn ravel_domain_row(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        value: &crate::phase0::desc::ValueKey,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(entry) = builder.entry(value) else {
            return Ok(Vec::new());
        };
        let crate::phase0::source::Producer::Applied { operands, .. } = &entry.producer else {
            return Ok(Vec::new());
        };
        let operands = operands.clone();
        if builder.node(value).and_then(|node| node.member()) != Some("ravel_multi_index") {
            return Ok(Vec::new());
        }
        let arrays: Vec<crate::phase0::desc::ValueKey> = operands
            .iter()
            .filter(|operand| builder.array(operand).is_some())
            .cloned()
            .collect();
        let [index, dims] = arrays.as_slice() else {
            return Ok(Vec::new());
        };
        let (index, dims) = (index.clone(), dims.clone());
        let mut candidate = Candidate::new(RuleId::TermToWires, Some("ravel"), request.epoch);
        candidate.matched = vec![value.clone(), index.clone(), dims.clone()];
        let under = builder.under(request.condition);
        let extent = builder.array(&index).map(|array| array.dims.clone()).unwrap_or_default();
        let [only] = extent.as_slice() else {
            return Ok(Vec::new());
        };
        let Verdict::Constant(axes) =
            builder.ask_for(&mut candidate, Question::Const(*only), request.epoch, under)
        else {
            return Ok(Vec::new());
        };
        if axes < 1 {
            return Ok(Vec::new());
        }
        let batch = crate::phase1::interface::Batch::grid(&[axes]);
        let mut dtype = Dtype::I64;
        for array in [&index, &dims] {
            if let Some(array) = builder.array(array) {
                dtype = Dtype::promote(dtype, array.dtype).unwrap_or(dtype);
            }
            let row = builder.fetch_port(&mut candidate, request, array, &batch)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        // Two comparisons per axis — `0 ≤ index_k` and `index_k < dims_k` —
        // and the fold of the 2r bits with `and`.
        let width = builder.omega.round(builder.dtype_width(dtype));
        Ok(vec![candidate
            .emit(Primitive::Cmp { width }, 2 * axes)
            .emit(Primitive::And, (2 * axes - 1).max(0))
            .parameter("r", axes)
            .output(
                request.target.clone(),
                Interface::Wire(Dtype::Bool),
                request.condition,
            )])
    }

    /// The `domain` row: the wire of the domain Boolean phase 0 mints for a
    /// `math.*` function whose operand carries no term of Φ — a Float
    /// scalar, or an integer scalar of data origin
    /// (`../../phase0/walk/exprs.rs`, `domain_condition`).
    ///
    /// The predicate is the domain of `../../phase0/ops.rs`, read off the
    /// AST node's member name, in the comparisons of `rules/README.md` §7.
    /// Over a scalar operand the one port is that scalar's wire. Over an
    /// operand of rank ≥ 1 the Boolean is the fold `all` of the per-element
    /// condition at the valid positions (`rules/README.md` §2): the port is
    /// then a grid batch over the operand's capacity box, the predicate runs
    /// at every slot, an inactive slot is passed by its flag, and the slots
    /// are folded with `and` — the same shape as the `β_I` row above.
    fn domain_row(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        value: &crate::phase0::desc::ValueKey,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(entry) = builder.entry(value) else {
            return Ok(Vec::new());
        };
        let crate::phase0::source::Producer::Applied { rules, operands, .. } = &entry.producer
        else {
            return Ok(Vec::new());
        };
        if !rules
            .iter()
            .all(|rule| matches!(rule, crate::phase0::rules::Rule::EMap))
        {
            return Ok(Vec::new());
        }
        let operands = operands.clone();
        let Some(member) = builder.node(value).and_then(|node| node.member()) else {
            return Ok(Vec::new());
        };
        let member = member.to_string();
        let domain = match crate::phase0::ops::numpy(&member)
            .or_else(|| crate::phase0::ops::math(&member))
        {
            Some(crate::phase0::ops::Class::Map(_, domain)) => domain,
            _ => return Ok(Vec::new()),
        };
        // The one operand whose elements the domain constrains; a rank-0
        // array is a scalar like any other (`elementwise/README.md` §2).
        let Some((operand, dtype, elements)) = operands.iter().find_map(|operand| {
            match &builder.entry(operand)?.desc {
                crate::phase0::desc::Desc::Scalar(scalar) => {
                    Some((operand.clone(), scalar.dtype, false))
                }
                crate::phase0::desc::Desc::Array(array) => {
                    Some((operand.clone(), array.dtype, array.rank() > 0))
                }
                _ => None,
            }
        }) else {
            return Ok(Vec::new());
        };
        let Some(primitives) = domain_gadget(builder, domain, dtype) else {
            return Ok(Vec::new());
        };
        let mut candidate = Candidate::new(RuleId::TermToWires, Some("domain"), request.epoch);
        candidate.matched.push(value.clone());
        let slots = if elements {
            self.domain_batch(builder, request, &mut candidate, &operand)?
        } else {
            let psi = builder.psi(&operand);
            let condition =
                builder.child_under_g(&mut candidate, request, psi, &operand.encode())?;
            let row = builder.child(
                request,
                Target::Value(operand.clone()),
                Family::Wire,
                condition,
            )?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
            Some(1)
        };
        let Some(slots) = slots else {
            return Ok(Vec::new());
        };
        for (primitive, times) in primitives {
            candidate = candidate.emit(primitive, times * slots);
        }
        if elements {
            // An inactive slot passes; the slots are folded with `and`.
            candidate = candidate
                .emit(crate::phase1::cost::Primitive::Or, slots)
                .emit(crate::phase1::cost::Primitive::And, (slots - 1).max(0));
        }
        Ok(vec![candidate.output(
            request.target.clone(),
            Interface::Wire(Dtype::Bool),
            request.condition,
        )])
    }

    /// The `domain` row's port over an array operand: the grid batch over
    /// the operand's capacity box under its own ψ, whose flags say which
    /// slots hold elements. Returns the number of slots.
    fn domain_batch(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        candidate: &mut Candidate,
        operand: &crate::phase0::desc::ValueKey,
    ) -> Result<Option<i128>, Reject> {
        let Some(dims) = builder.array(operand).map(|array| array.dims.clone()) else {
            return Ok(None);
        };
        let under = builder.under(builder.psi(operand));
        let boxes = builder.capacity_box(candidate, &dims, request.epoch, under);
        let Some(capacities) = boxes.first().cloned() else {
            return Ok(None);
        };
        let batch = crate::phase1::interface::Batch::grid(&capacities.axes);
        let row = builder.fetch_port(candidate, request, operand, &batch)?;
        let Some(first) = row.entries.first() else {
            return Ok(None);
        };
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
        Ok(Some(capacities.slots()))
    }

    /// The ports of the `β_I` row: the elements of the index array at every
    /// physical position, and the extent of the axis they address.
    ///
    /// The batch is the grid over the capacity of I under ψ_I, so the flags
    /// are the batch's own and the gadget below reads a well-formed element
    /// at every slot (`term_to_wires.md`, the `β_I` row).
    fn in_range_row(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        candidate: &mut Candidate,
        value: &crate::phase0::desc::ValueKey,
    ) -> Result<bool, Reject> {
        let Some(matched) = crate::phase1::rules::reading::gather::take_of(builder, value) else {
            return Ok(false);
        };
        let Some(index_dims) = builder.array(&matched.index).map(|array| array.dims.clone())
        else {
            // A rank-0 index is one scalar, whose in-range bit is
            // `INDEX_NORMALIZE`'s `ok` over its own wire.
            let Some(extent) = builder
                .array(&matched.source)
                .map(|array| array.dims[matched.axis])
            else {
                return Ok(false);
            };
            let row = builder.index_norm_port(
                request,
                Target::Value(matched.index.clone()),
                extent,
                request.condition,
            )?;
            let Some(first) = row.entries.first() else {
                return Ok(false);
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
            return Ok(true);
        };
        // `idx`: the grid batch over the capacities of I, whose widths are
        // the MAX of its extents under ψ_I.
        let under = builder.under(builder.psi(&matched.index));
        let boxes = builder.capacity_box(candidate, &index_dims, request.epoch, under);
        let Some(capacities) = boxes.first().cloned() else {
            return Ok(false);
        };
        let batch = crate::phase1::interface::Batch::grid(&capacities.axes);
        let row = builder.fetch_port(candidate, request, &matched.index, &batch)?;
        let Some(first) = row.entries.first() else {
            return Ok(false);
        };
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
        // `ext`: the extent of the gathered axis, when it is not a literal.
        let extent = builder.array(&matched.source).map(|array| array.dims[matched.axis]);
        let Some(extent) = extent else {
            return Ok(false);
        };
        if builder.terms.constant_value(extent).is_none() {
            let row =
                builder.child(request, Target::Term(extent), Family::Wire, request.condition)?;
            let Some(first) = row.entries.first() else {
                return Ok(false);
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        Ok(true)
    }

    /// The gadgets of the `β_I` row: per slot the two comparisons and the
    /// `and` of `INDEX_NORMALIZE`'s `ok`, then `or` with the inactive flag
    /// and one `and` per slot beyond the first.
    fn in_range_gadgets(
        &self,
        builder: &Builder<'_>,
        value: &crate::phase0::desc::ValueKey,
    ) -> Vec<(crate::phase1::cost::Primitive, i128)> {
        use crate::phase1::cost::Primitive;
        let Some(matched) = crate::phase1::rules::reading::gather::take_of(builder, value) else {
            return Vec::new();
        };
        let slots: i128 = builder
            .array(&matched.index)
            .map(|array| {
                array
                    .dims
                    .iter()
                    .map(|&dimension| builder.interval(dimension).hi.unwrap_or(1).max(0))
                    .product()
            })
            .unwrap_or(1);
        let width = builder.omega.round(builder.dtype_width(
            builder
                .array(&matched.index)
                .map(|array| array.dtype)
                .unwrap_or(Dtype::I64),
        ));
        vec![
            (Primitive::Cmp { width }, 2 * slots),
            (Primitive::And, slots + (slots - 1).max(0)),
            (Primitive::Or, slots),
        ]
    }

    /// One sub-term's wire, at whichever sort the child node has.
    fn child_wire(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        child: u32,
    ) -> Result<Option<(ReqId, u32)>, Reject> {
        let target = match builder.terms.bool_at(child) {
            Some(formula) => Target::BoolTerm(formula),
            None => match builder.terms.int_at(child) {
                Some(term) => Target::Term(term),
                None => return Ok(None),
            },
        };
        let row = builder.child(request, target, Family::Wire, request.condition)?;
        Ok(row
            .entries
            .first()
            .map(|entry| (entry.requirement, entry.epoch)))
    }
}
