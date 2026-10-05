//! `LEN_SHAPE` (`revision/phase1/rules/scalar_control/len_shape.md`).
//!
//! One axis extent of an array value as a wire. Phase 0 makes `len(a)`,
//! `a.shape[k]` and `a.size` shape scalars whose term is `dims_a[k]`; the row
//! of such a scalar is the row of its term (`rules/README.md` §3.2), so the
//! rule also answers the term requirement other rules raise for a dimension
//! term, and the variable ports for `n_x`, `ℓ` and `c′`, which are dimension
//! terms of the input, the sized value and the `unique` value.
//!
//! Three ways: a constant (`/const`); the extent wire a bundle of `a`
//! already carries, at no cost (`/read`, one node per bundle); the term
//! compiled by `TERM_TO_WIRES` from the wires of its variables, which needs
//! no bundle of `a` (`/term`).

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::phi::VarOrigin;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::{width_of, Builder};
use crate::phase1::graph::Target;
use crate::phase1::cost::Primitive;
use crate::phase1::interface::{Family, Interface};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct LenShape;

/// The families a bundle of `a` can carry the extent of axis `k` on
/// (`len_shape.md`, `/read`).
const BUNDLES: [Family; 3] = [Family::Store, Family::Masked, Family::Packed];

impl Rule for LenShape {
    fn id(&self) -> RuleId {
        RuleId::LenShape
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some((term, array, axis, matched)) = self.matches(builder, request) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();

        // P1 (variant): CONST dims_a[k] under ψ_a.
        let mut probe = Candidate::new(RuleId::LenShape, Some("/const"), request.epoch);
        let psi = builder.psi(&array);
        let condition = builder.child_under_g(&mut probe, request, psi, &array.encode())?;
        let under = builder.under(condition);
        let constant = builder.ask_for(&mut probe, Question::Const(term), request.epoch, under);
        if let Verdict::Constant(value) = constant {
            let mut candidate = probe
                .clone()
                .output(request.target.clone(), Interface::Wire(Dtype::Idx), request.condition)
                .parameter("n", value);
            candidate.matched = matched.clone();
            return Ok(vec![candidate]);
        }

        // `/div`: the inferred axis of a reshape, ℓ = size(a) div p, which
        // is the one variable of that kind with a defining equation
        // (`len_shape.md`, the `/div` variant).
        let divided = self.inferred_axis(builder, request, term, &array, axis, condition)?;

        // `/read`: one node per bundle of `a` that carries an extent wire.
        //
        // Not for the inferred axis of a reshape: every bundle of a reshape
        // takes its own extent wire from this very row — `BUILD_DENSE`'s
        // `shape_k` port and a `Ref`'s `ext_k` are `Represent(ℓ, Wire[Idx])`
        // — so `/read` is the circular route and `/div` is the only provider
        // (`len_shape.md`, `/read`).
        for family in BUNDLES.iter().copied().filter(|_| divided.is_empty()) {
            if !self.carries_extent(builder, &array, family, axis) {
                continue;
            }
            let row = builder.child(request, Target::Value(array.clone()), family, condition)?;
            for entry in &row.entries {
                let mut candidate = Candidate::new(RuleId::LenShape, Some("/read"), request.epoch)
                    .output(
                        request.target.clone(),
                        Interface::Wire(Dtype::Idx),
                        request.condition,
                    )
                    .parameter("axis", axis as i128);
                candidate.matched = matched.clone();
                candidate.inputs.push(entry.requirement);
                candidate.epoch = candidate.epoch.max(entry.epoch);
                candidate.answers = probe.answers.clone();
                out.push(candidate);
            }
        }

        out.extend(divided);

        // `/term`: absent when the term is a variable with no defining term
        // (`n_x`, `ℓ`, `c′`), whose only wires are bundles'.
        if builder.terms.var_of(term).is_none() {
            let row = builder.child(request, Target::Term(term), Family::Wire, condition)?;
            if let Some(entry) = row.entries.first() {
                let mut candidate = Candidate::new(RuleId::LenShape, Some("/term"), request.epoch)
                    .output(
                        request.target.clone(),
                        Interface::Wire(Dtype::Idx),
                        request.condition,
                    )
                    .parameter("axis", axis as i128);
                candidate.matched = matched.clone();
                candidate.inputs.push(entry.requirement);
                candidate.epoch = candidate.epoch.max(entry.epoch);
                candidate.answers = probe.answers.clone();
                out.push(candidate);
            }
        }
        Ok(out)
    }
}

impl LenShape {
    /// `/div`: the inferred axis of `reshape(a, shape)` with a `-1` whose
    /// companion extents are not all constant (`len_shape.md`).
    ///
    /// Phase 0 gives that axis the variable ℓ with `0 ≤ ℓ ≤ size(a)` and the
    /// check `p > 0 ∧ size(a) = ∏ target`, p the product of the other target
    /// extents (`../../phase0/definition.md` §5.6). ℓ has no defining term,
    /// so `/term` does not answer it and `/read` would need a bundle of the
    /// reshape, which is the value being built. The wire is the quotient of
    /// `divmod_d(size(a), p)`: the gadget gives `size(a) = p·q + r` with
    /// `0 ≤ r < p`, the check gives `size(a) = ℓ·p` with `p ≥ 1`, and
    /// Euclidean division is unique, so `q = ℓ` and `r = 0` on every
    /// admitted run. The check itself is `ENFORCE_CHECK`'s.
    ///
    /// `None` when the value is no reshape, when the axis is not the
    /// inferred one, or when either product has two run-time factors and so
    /// no term (`rules/README.md` §6.3).
    fn inferred_axis(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
        term: TermId,
        value: &ValueKey,
        axis: usize,
        condition: crate::phase0::term::BoolId,
    ) -> Result<Vec<Candidate>, Reject> {
        if builder.terms.var_of(term).is_none() {
            return Ok(Vec::new());
        }
        let Some(entry) = builder.entry(value) else {
            return Ok(Vec::new());
        };
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return Ok(Vec::new());
        };
        if !rules.contains(&Phase0Rule::EReshape) {
            return Ok(Vec::new());
        }
        let operands = operands.clone();
        let Some(dims) = builder.array(value).map(|array| array.dims.clone()) else {
            return Ok(Vec::new());
        };
        let Some(source) = operands.iter().find(|key| builder.array(key).is_some()).cloned()
        else {
            return Ok(Vec::new());
        };
        let source_dims = builder.array(&source).map(|array| array.dims.clone()).unwrap_or_default();
        // `p`, the product of the target extents off the inferred axis, and
        // `size(a)`, the product of the source's extents.
        let others: Vec<TermId> = dims
            .iter()
            .enumerate()
            .filter(|&(at, _)| at != axis)
            .map(|(_, &dimension)| dimension)
            .collect();
        let Some(divisor) = product(builder, &others) else {
            return Ok(Vec::new());
        };
        let Some(size) = product(builder, &source_dims) else {
            return Ok(Vec::new());
        };
        let mut candidate = Candidate::new(RuleId::LenShape, Some("/div"), request.epoch);
        candidate.matched = vec![value.clone(), source.clone()];
        // `size` and `p`: one term requirement each, absent where the term is
        // a literal the gadget folds in.
        for subject in [size, divisor] {
            if builder.terms.constant_value(subject).is_some() {
                continue;
            }
            let row = builder.child(request, Target::Term(subject), Family::Wire, condition)?;
            let Some(first) = row.entries.first() else {
                return Ok(Vec::new());
            };
            candidate.inputs.push(first.requirement);
            candidate.epoch = candidate.epoch.max(first.epoch);
        }
        let width = builder.omega.round(width_of(builder.interval(size)));
        if !builder.omega.admits(width) {
            return Ok(Vec::new());
        }
        Ok(vec![candidate
            .emit(Primitive::DivModWire { width }, 1)
            .output(request.target.clone(), Interface::Wire(Dtype::Idx), request.condition)
            .parameter("axis", axis as i128)
            .parameter("w", width as i128)])
    }

    /// `source_pattern`: the scalar `len(a)`, `a.shape[k]` or `a.size`, or a
    /// term requirement for a dimension term.
    ///
    /// Returns the dimension term, the value whose axis it is, the axis, and
    /// the keys the rule matched.
    fn matches(
        &self,
        builder: &Builder<'_>,
        request: &RowRequest,
    ) -> Option<(TermId, ValueKey, usize, Vec<ValueKey>)> {
        match &request.target {
            Target::Value(key) => {
                let entry = builder.entry(key)?;
                let term = entry.desc.as_scalar()?.term.as_int()?;
                // The value names the array it measures: `len(a)`, `a.size`
                // and `a.shape` read it out of their operand.
                if let Producer::Applied { rules, operands, .. } = &entry.producer {
                    if rules.iter().any(|rule| {
                        matches!(
                            rule,
                            Phase0Rule::ELen | Phase0Rule::EShape | Phase0Rule::EProperty
                        )
                    }) {
                        if let Some(array) =
                            operands.iter().find(|key| builder.array(key).is_some())
                        {
                            if let Some(axis) = self.axis_of(builder, array, term) {
                                return Some((
                                    term,
                                    array.clone(),
                                    axis,
                                    vec![key.clone(), array.clone()],
                                ));
                            }
                        }
                    }
                }
                // It does not always: `a.shape[k]` is an `E-Property` tuple
                // and an `E-Composite-Index` into it, and the index's
                // operand is the tuple, not the array. The term is the
                // dimension term all the same, so the variable it is names
                // the array (`rules/README.md` §2).
                let (array, axis) = self.sized_value(builder, term)?;
                Some((term, array.clone(), axis, vec![key.clone(), array]))
            }
            // A dimension term asked directly.
            Target::Term(term) => {
                let (array, axis) = self.sized_value(builder, *term)?;
                Some((*term, array.clone(), axis, vec![array]))
            }
            Target::BoolTerm(_) => None,
        }
    }

    /// The value a dimension term measures and the axis it measures, read
    /// off the variable the term is (`rules/README.md` §2): `n_x` names the
    /// input x, a shape symbol the first input in Δ order that binds it, ℓ
    /// and c′ the value they size.
    ///
    /// `None` when the term is not a bare variable of one of those kinds, so
    /// a data scalar never reaches this rule.
    fn sized_value(
        &self,
        builder: &Builder<'_>,
        term: TermId,
    ) -> Option<(ValueKey, usize)> {
        let variable = builder.terms.var_of(term)?;
        let array = match builder.var_origin(variable).clone() {
            VarOrigin::InputLength { input, .. } => builder.input_key(&input)?.clone(),
            VarOrigin::ValueSized { value } | VarOrigin::UniqueCount { value } => value,
            VarOrigin::ShapeSymbol { .. } => builder.first_input_with_dim(term)?,
            _ => return None,
        };
        // `unique` with `return_inverse` names the Tuple the call produced,
        // which Δ keys as a Composite; the value that carries the extent is
        // the element bound out of it (`rules/README.md` §2).
        let array = match builder.array(&array) {
            Some(_) => array,
            None => builder.array_with_dim(term)?,
        };
        let axis = self.axis_of(builder, &array, term)?;
        Some((array, axis))
    }

    /// The axis of `array` whose dimension term is `term`.
    fn axis_of(&self, builder: &Builder<'_>, array: &ValueKey, term: TermId) -> Option<usize> {
        builder
            .array(array)?
            .dims
            .iter()
            .position(|&dimension| dimension == term)
    }

    /// Whether a bundle of this family carries the extent of axis `k`
    /// (`len_shape.md`, `required_input_ports`).
    fn carries_extent(
        &self,
        builder: &Builder<'_>,
        array: &ValueKey,
        family: Family,
        axis: usize,
    ) -> bool {
        let rank = builder.array(array).map(|array| array.rank()).unwrap_or(0);
        match family {
            // A box `Store` carries one shape wire per axis.
            Family::Store => axis < rank,
            // A rank-1 `Masked` carries ℓ, which is axis 0.
            Family::Masked => rank == 1 && axis == 0,
            // A `Packed` carries ℓ for axis 0 of a rank-1 value and r shape
            // wires at rank ≥ 2 (`rules/README.md` §4.1), the first of which
            // `PACK_BY_KEY` and `BUILD_DENSE` compute.
            Family::Packed => axis < rank,
            _ => false,
        }
    }
}

/// `∏_k dims[k]` as a term, when at most one factor is not a literal
/// (`../../phase0/rules/helpers.rs`, `product_term`: §3.1 has no product of
/// two run-time terms).
fn product(builder: &mut Builder<'_>, dims: &[TermId]) -> Option<TermId> {
    let mut constant: i128 = 1;
    let mut symbolic: Option<TermId> = None;
    for &dimension in dims {
        match builder.terms.constant_value(dimension) {
            Some(value) => constant *= value,
            None if symbolic.is_none() => symbolic = Some(dimension),
            None => return None,
        }
    }
    Some(match symbolic {
        None => builder.terms.constant(constant),
        Some(factor) => builder.terms.mul_const(constant, factor),
    })
}
