//! `MAKE_REF` (`revision/phase1/rules/mapping/make_ref.md`).
//!
//! Answers `Represent(v, Ref)` for a view v of a value u whose map μ is
//! constant: every element of v is the element of u at a coordinate that is
//! an expression over v's coordinates and literals. It allocates no wire and
//! emits no constraint; its effect is the map, which readers fold at compile
//! time. It matches a subscript with basic entries, a transpose, a reshape, a
//! broadcast or a reversal; the same AST nodes with a parameter that is not
//! constant under g go to `MAKE_REF_RUNTIME`, and the CONST premises below
//! decide which of the two instantiates per (v, owner).

use crate::phase0::term::TermId;

use crate::phase1::build::Builder;
use crate::phase1::graph::Target;
use crate::phase1::interface::{
    Capacity, Family, Interface, LayoutKind, Mu, MuForm, OwnerRef, RefIface,
};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

use super::{MapParam, Match, View};

pub(crate) struct MakeRef;

impl Rule for MakeRef {
    fn id(&self) -> RuleId {
        RuleId::MakeRef
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
        let mut probe = Candidate::new(RuleId::MakeRef, None, request.epoch);
        let under = builder.under(request.condition);
        // P1: one CONST question per entry. A parameter that is not constant
        // sends this (v, owner) to `MAKE_REF_RUNTIME`, which reads the same
        // answers out of A.
        let Some(parameters) = super::parameters(builder, &mut probe, &matched, request.epoch, under)
        else {
            return Ok(Vec::new());
        };
        if !parameters.iter().all(|start| start.literal().is_some()) {
            return Ok(Vec::new());
        }
        let literals: Vec<MapParam> = parameters
            .iter()
            .filter_map(|start| start.literal().map(MapParam::Literal))
            .collect();
        build_refs(
            builder,
            request,
            &matched,
            &mut probe,
            &literals,
            &[],
            RuleId::MakeRef,
            /* flatten = */ false,
        )
    }
}

/// One `Ref` per owner bundle, with the map the view prescribes.
///
/// `MAKE_REF` and `MAKE_REF_RUNTIME` differ only in where a parameter comes
/// from — a literal or a wire of π — so the bundle, the strides, the extents
/// and the span are built once here, with `π` given as the map fragment each
/// owner axis starts from.
///
/// `flatten` is `REINDEX_REF`'s: over a `Ref` owner it composes the two maps
/// into one instead of stacking them, so that the bundle is flat.
pub(super) fn build_refs(
    builder: &mut Builder<'_>,
    request: &RowRequest,
    matched: &Match,
    probe: &mut Candidate,
    starts: &[MapParam],
    params: &[TermId],
    rule: RuleId,
    flatten: bool,
) -> Result<Vec<Candidate>, Reject> {
    let owner_psi = builder.psi(&matched.owner);
    let condition = builder.child_under_g(probe, request, owner_psi, &matched.owner.encode())?;
    let mut out = Vec::new();
    for family in super::OWNERS {
        let row = builder.child(request, Target::Value(matched.owner.clone()), family, condition)?;
        'bundle: for bundle in &row.entries {
            let Some(owner) = owner_ref(&bundle.interface, &matched.owner) else {
                continue;
            };
            let over_ref = matches!(bundle.interface, Interface::Ref(_));
            // `MAKE_REF` stacks over a `Ref` owner and `REINDEX_REF`
            // composes, so each takes the owner the other leaves.
            if over_ref != flatten {
                continue;
            }
            // A nested `Ref` is `REINDEX_REF`'s; this rule takes flat owners
            // only, and the span it would reach is bounded by Ω.map_depth.
            let span = match &bundle.interface {
                Interface::Ref(inner) => inner.span + 1,
                _ => 1,
            };
            if span > builder.omega.map_depth {
                continue;
            }
            // The map, with the parameter wires it adds: a view stride that
            // is not constant is R4 of `make_ref_runtime.md`, a wire of π
            // this rule never allows (`MAKE_REF` is the constant map).
            let mut params = params.to_vec();
            let supplied = params.len();
            let Some(map) = map_of(
                builder,
                matched,
                starts,
                &bundle.interface,
                &owner,
                &mut params,
                rule == RuleId::MakeRefRuntime,
            ) else {
                continue;
            };
            // `MAKE_REF_RUNTIME` instantiates for a (v, owner) exactly where
            // a parameter or a stride of the map is not constant; where both
            // are, `MAKE_REF` built the node already.
            let constant_parameters =
                starts.iter().all(|start| matches!(start, MapParam::Literal(_)));
            if rule == RuleId::MakeRefRuntime
                && constant_parameters
                && params.len() == supplied
            {
                continue;
            }
            let mut candidate = Candidate::new(rule, None, request.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched = vec![matched.value.clone(), matched.owner.clone()];
            candidate.inputs.push(bundle.requirement);
            candidate.epoch = candidate.epoch.max(bundle.epoch);
            // ext_k: the literal of CONST dims_v[k], or the wire of the term
            // requirement the node raises (`mapping/README.md` §2.4).
            let ext: Vec<crate::phase1::interface::Extent> = matched
                .dims
                .iter()
                .map(|&dimension| super::extent(builder, dimension))
                .collect();
            // An `ext_k` port with no provider is a required port the
            // library cannot fill, so the bundle is not offered: a `Ref`
            // whose extent has no wire states an extent it cannot produce.
            for extent in &ext {
                if let crate::phase1::interface::Extent::Term(term) = extent {
                    let row =
                        builder.child(request, Target::Term(*term), Family::Wire, request.condition)?;
                    let Some(first) = row.entries.first() else {
                        continue 'bundle;
                    };
                    candidate.inputs.push(first.requirement);
                    candidate.epoch = candidate.epoch.max(first.epoch);
                }
            }
            let (form, expressions) = map;
            // A stride wire is a term requirement like any other parameter,
            // and like `ext_k` it is required.
            for &term in params[supplied..].to_vec().iter() {
                let row = builder.child(request, Target::Term(term), Family::Wire, request.condition)?;
                let Some(first) = row.entries.first() else {
                    continue 'bundle;
                };
                candidate.inputs.push(first.requirement);
                candidate.epoch = candidate.epoch.max(first.epoch);
            }
            let mut bundle_iface = RefIface {
                owner,
                form,
                map: expressions,
                params: params.clone(),
                ext,
                span,
                nested: None,
            };
            if let Interface::Ref(inner) = &bundle.interface {
                // A `Ref` over a `Ref` is nested: its map runs over the owner
                // value's coordinates. `REINDEX_REF` composes the two into
                // one map over the flat owner's keys.
                match flatten {
                    false => bundle_iface.nested = Some(Box::new(inner.clone())),
                    true => {
                        let Some((composed_form, composed_map)) =
                            super::reindex_ref::compose_maps(&bundle_iface, inner)
                        else {
                            continue;
                        };
                        bundle_iface.form = composed_form;
                        bundle_iface.map = composed_map;
                        for term in &inner.params {
                            if !bundle_iface.params.contains(term) {
                                bundle_iface.params.push(*term);
                            }
                        }
                    }
                }
            }
            candidate = candidate
                .output(
                    request.target.clone(),
                    Interface::Ref(bundle_iface),
                    request.condition,
                )
                .parameter("span", span as i128);
            out.push(candidate);
        }
    }
    Ok(out)
}

/// The owner node a bundle is, when the bundle is one a `Ref` may point into
/// (`mapping/README.md` §2.2).
pub(super) fn owner_ref(
    interface: &Interface,
    value: &crate::phase0::desc::ValueKey,
) -> Option<OwnerRef> {
    // A `Ref` owner is flattened by `REINDEX_REF`; the owner named here is
    // the flat one it already points into.
    if let Interface::Ref(inner) = interface {
        return Some(inner.owner.clone());
    }
    crate::phase1::rules::reading::owner_of(interface, value)
}

/// μ for one (view, owner) pair (`make_ref.md`, `produced_output_bundle`).
///
/// `None` when the owner's key strides are not literal, which is R3 and R4 of
/// `make_ref_runtime.md`: a stride that is a product of run-time extents is a
/// wire, and no reader of step 2 evaluates one.
#[allow(clippy::too_many_arguments)]
fn map_of(
    builder: &mut Builder<'_>,
    matched: &Match,
    starts: &[MapParam],
    interface: &Interface,
    owner: &OwnerRef,
    params: &mut Vec<TermId>,
    runtime: bool,
) -> Option<(MuForm, Vec<Mu>)> {
    let dense_key = !matches!(owner.layout, LayoutKind::Box(_));
    match &matched.view {
        View::Subscript(entries) => {
            let lower: Vec<Option<Mu>> = starts
                .iter()
                .map(|start| Some(start.expression()))
                .collect();
            let strides = super::key_strides(builder, interface, &matched.owner_dims)?;
            Some((
                MuForm::Coordinate { strides },
                super::subscript_map(entries, &lower),
            ))
        }
        View::Transpose(permutation) => {
            let strides = super::key_strides(builder, interface, &matched.owner_dims)?;
            let mut map = vec![Mu::Const(0); permutation.len()];
            for (output, &owner_axis) in permutation.iter().enumerate() {
                map[owner_axis] = Mu::Coord(output);
            }
            Some((MuForm::Coordinate { strides }, map))
        }
        View::Reshape => {
            // ℓ(i) = Σ_k S^v_k × i_k, the C-order position of v's coordinate.
            // A stride whose factors are not all literal is a wire of π (R4).
            let view_strides = view_stride_params(builder, &matched.dims, params, runtime)?;
            let linear = linear_form(&view_strides);
            if dense_key {
                // Over an owner whose key is the C-order index, the key form
                // is the linear position itself.
                return Some((MuForm::Key, vec![linear]));
            }
            // Over a box owner the coordinate form writes `unlin` out, which
            // needs literal owner extents beyond axis 0.
            let owner_extents: Vec<Capacity> = matched
                .owner_dims
                .iter()
                .map(|&dimension| builder.terms.constant_value(dimension))
                .collect::<Option<Vec<_>>>()?;
            let owner_strides = super::key_strides(builder, interface, &matched.owner_dims)?;
            let rank = owner_extents.len();
            let mut map = Vec::with_capacity(rank);
            for axis in 0..rank {
                let stride: Capacity = owner_extents[axis + 1..].iter().product();
                let mut expression = linear.clone();
                if axis < rank - 1 {
                    expression = Mu::DivConst(Box::new(expression), stride);
                }
                if axis > 0 {
                    expression = Mu::ModConst(Box::new(expression), owner_extents[axis]);
                }
                map.push(expression);
            }
            Some((MuForm::Coordinate { strides: owner_strides }, map))
        }
        // The diagonal: one output axis onto both owner axes, `i_0` twice.
        // `min(d_0, d_1)` is the extent, so `(i_0, i_0)` is inside the owner
        // on every entry.
        View::Diagonal => {
            let strides = super::key_strides(builder, interface, &matched.owner_dims)?;
            Some((
                MuForm::Coordinate { strides },
                vec![Mu::Coord(0), Mu::Coord(0)],
            ))
        }
        View::Broadcast => {
            let strides = super::key_strides(builder, interface, &matched.owner_dims)?;
            let offset = matched.dims.len() - matched.owner_dims.len();
            let mut map = Vec::with_capacity(matched.owner_dims.len());
            for (axis, &start) in starts.iter().enumerate().take(matched.owner_dims.len()) {
                // P1 decided "is the owner's extent 1" per axis: 1 gives the
                // constant 0, otherwise the output coordinate. A run-time
                // test is R5, whose Boolean wire multiplies the coordinate.
                map.push(match start {
                    MapParam::Literal(1) => Mu::Const(0),
                    MapParam::Literal(_) => Mu::Coord(axis + offset),
                    MapParam::Wire(index) => {
                        Mu::ScaleParam(index, Box::new(Mu::Coord(axis + offset)))
                    }
                });
            }
            Some((MuForm::Coordinate { strides }, map))
        }
    }
}

/// `S^v_k = ∏_{b>k} dims_v[b]` as map fragments (`make_ref.md` P2, key
/// form; `make_ref_runtime.md` R4).
///
/// A stride every factor of which is CONST is the literal. One whose factors
/// include a run-time extent is a wire of π — the term `k·d_b`, which
/// `TERM_TO_WIRES` compiles — and only `MAKE_REF_RUNTIME` admits it, which
/// is what keeps exactly one of the two rules instantiated per (v, owner).
/// Two run-time factors have no term at all (`rules/README.md` §6.3).
fn view_stride_params(
    builder: &mut Builder<'_>,
    dims: &[TermId],
    params: &mut Vec<TermId>,
    runtime: bool,
) -> Option<Vec<MapParam>> {
    let mut strides: Vec<MapParam> = vec![MapParam::Literal(1); dims.len()];
    let mut factors: Vec<TermId> = Vec::new();
    for axis in (0..dims.len().saturating_sub(1)).rev() {
        factors.push(dims[axis + 1]);
        let mut constant: Capacity = 1;
        let mut wire: Option<TermId> = None;
        for &factor in &factors {
            match builder.terms.constant_value(factor) {
                Some(value) => constant *= value,
                None if wire.is_none() => wire = Some(factor),
                None => return None,
            }
        }
        strides[axis] = match wire {
            None => MapParam::Literal(constant),
            Some(factor) if runtime => {
                let term = builder.terms.mul_const(constant, factor);
                let at = match params.iter().position(|&held| held == term) {
                    Some(at) => at,
                    None => {
                        params.push(term);
                        params.len() - 1
                    }
                };
                MapParam::Wire(at)
            }
            Some(_) => return None,
        };
    }
    Some(strides)
}

/// `Σ_k S_k × i_k`, the C-order position of a coordinate, with `π_m × i_k`
/// where the stride is a wire.
fn linear_form(strides: &[MapParam]) -> Mu {
    let mut expression: Option<Mu> = None;
    for (axis, &stride) in strides.iter().enumerate() {
        let term = match stride {
            MapParam::Literal(1) => Mu::Coord(axis),
            MapParam::Literal(value) => Mu::Scale(value, Box::new(Mu::Coord(axis))),
            MapParam::Wire(at) => Mu::ScaleParam(at, Box::new(Mu::Coord(axis))),
        };
        expression = Some(match expression {
            None => term,
            Some(sum) => Mu::Add(Box::new(sum), Box::new(term)),
        });
    }
    expression.unwrap_or(Mu::Const(0))
}
