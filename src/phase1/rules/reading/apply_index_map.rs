//! `APPLY_INDEX_MAP` (`revision/phase1/rules/reading/apply_index_map.md`).
//!
//! Composes the coordinates of a batch with the map from v to the owner that
//! holds v's elements and produces the read request: per entry the address —
//! the key of the owner as `README.md` §3.2 defines it per layout — and the
//! active entry. The map is the μ of a flat `Ref[v; o, μ]` provider of v, or
//! the identity when v has a Store-family provider of its own. One node per
//! such provider.

use crate::phase1::build::Builder;
use crate::phase1::cost::Primitive;
use crate::phase1::interface::{Family, Interface, LayoutKind, Mu, MuForm, RefIface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

use super::Composed;

pub(crate) struct ApplyIndexMap;

impl Rule for ApplyIndexMap {
    fn id(&self) -> RuleId {
        RuleId::ApplyIndexMap
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::ReadReq]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let (Some(value), Some(batch)) = (request.target.as_value(), request.batch.clone()) else {
            return Ok(Vec::new());
        };
        let value = value.clone();
        let Some(dims) = builder.array(&value).map(|array| array.dims.clone()) else {
            return Ok(Vec::new());
        };
        // `coords`: the one node every provider shares.
        let mut probe = Candidate::new(RuleId::ApplyIndexMap, None, request.epoch);
        let coords = builder.child_in_batch(
            request,
            request.target.clone(),
            Family::Coords,
            request.condition,
        )?;
        let Some(coords) = coords.entries.first().cloned() else {
            return Ok(Vec::new());
        };

        let mut out = Vec::new();
        // The identity map over each Store-family provider of v, and the map
        // of each flat `Ref` provider.
        for family in [Family::Store, Family::Masked, Family::Packed, Family::Ref] {
            let row = builder.child(request, request.target.clone(), family, request.condition)?;
            for provider in &row.entries {
                let (owner, reference) = match &provider.interface {
                    // A nested `Ref` is not matched; `REINDEX_REF` flattens it
                    // (`mapping/README.md` §2.6).
                    Interface::Ref(reference) if !reference.is_flat() => continue,
                    Interface::Ref(reference) => (reference.owner.clone(), Some(reference.clone())),
                    other => match super::owner_of(other, &value) {
                        Some(owner) => (owner, None),
                        None => continue,
                    },
                };
                // premises (structural): a ranked or keyed owner has rank 1,
                // and a key-form μ over a box owner is not admitted.
                if !owner.layout.is_dense() && batch.rank() > 1 {
                    continue;
                }
                let owner_dims = match &reference {
                    Some(_) => builder
                        .array(&owner.value)
                        .map(|array| array.dims.clone())
                        .unwrap_or_default(),
                    None => dims.clone(),
                };
                if matches!(reference.as_ref().map(|r| &r.form), Some(MuForm::Key))
                    && matches!(owner.layout, LayoutKind::Box(_))
                {
                    continue;
                }
                let Some(strides) =
                    super::strides_of(builder, reference.as_ref(), &owner, &owner_dims)
                else {
                    continue;
                };
                let mut candidate = Candidate::new(RuleId::ApplyIndexMap, None, request.epoch);
                candidate.answers = probe.answers.clone();
                candidate.matched.push(value.clone());
                candidate.inputs.push(coords.requirement);
                candidate.epoch = candidate.epoch.max(coords.epoch);
                candidate.inputs.push(provider.requirement);
                candidate.epoch = candidate.epoch.max(provider.epoch);

                // The address per entry, and the gates its evaluation costs.
                // The composition is the batch template substituted into μ
                // and asked CONST (`README.md`, "CONST on a composed
                // coordinate"); a map with no term takes every entry's
                // address to a wire.
                let composition = super::Composition::resolve(
                    builder,
                    &mut candidate,
                    reference.as_ref(),
                    &batch,
                    request.epoch,
                    request.condition,
                );
                let mut wires = 0i128;
                match &composition {
                    Some(composition) => {
                        for at in 0..batch.len() {
                            if super::address(&owner, &strides, composition.at(at))
                                == Composed::Wire
                            {
                                wires += 1;
                            }
                        }
                    }
                    None => wires = batch.len(),
                }
                let constant = wires == 0;
                let per_entry = reference
                    .as_ref()
                    .map(|reference| gates_of(builder, reference))
                    .unwrap_or_default();
                for (primitive, times) in per_entry {
                    candidate = candidate.emit(primitive, wires * times);
                }
                candidate = candidate
                    .output(
                        request.target.clone(),
                        Interface::ReadReq {
                            owner: owner.clone(),
                            queries: batch.len(),
                            constant,
                        },
                        request.condition,
                    )
                    .parameter("Q", batch.len())
                    .parameter("B", owner.slots());
                let _ = &probe;
                out.push(candidate);
            }
        }
        // Keep the questions of a rule instance that found no provider in A,
        // attached to the construction rather than to a node.
        let _ = &mut probe;
        Ok(out)
    }
}

/// The gadgets one entry's address costs beyond `lin` (`apply_index_map.md`,
/// emitter): one `mul` per product with a parameter wire, one `divmod_k` per
/// `div` or `mod` by a constant, and one `divmod_d` per division by a
/// parameter wire (`rules/README.md` §4.2, §7).
///
/// The widths are the reader's: a key of the owner lies in `[0, B_tot)`, so
/// `bits(B_tot)` bounds every quotient and remainder of an address.
fn gates_of(builder: &Builder<'_>, reference: &RefIface) -> Vec<(Primitive, i128)> {
    let width = builder
        .omega
        .round(crate::phase1::build::bits_of_capacity(reference.owner.slots()));
    let mut out: Vec<(Primitive, i128)> = Vec::new();
    for expression in &reference.map {
        count(expression, width, &mut out);
    }
    out
}

fn count(expression: &Mu, width: u32, out: &mut Vec<(Primitive, i128)>) {
    let mut charge = |primitive: Primitive| match out.iter_mut().find(|(held, _)| *held == primitive)
    {
        Some(entry) => entry.1 += 1,
        None => out.push((primitive, 1)),
    };
    match expression {
        Mu::Const(_) | Mu::Coord(_) | Mu::Param(_) => {}
        Mu::Add(left, right) | Mu::Sub(left, right) => {
            count(left, width, out);
            count(right, width, out);
        }
        Mu::Scale(_, inner) => count(inner, width, out),
        Mu::ScaleParam(_, inner) => {
            charge(Primitive::Mul);
            count(inner, width, out);
        }
        Mu::DivConst(inner, divisor) | Mu::ModConst(inner, divisor) => {
            charge(Primitive::DivModConst {
                quotient_width: width,
                divisor_width: crate::phase1::build::bits_for(*divisor),
                power_of_two: divisor.count_ones() == 1,
            });
            count(inner, width, out);
        }
        Mu::DivParam(inner, _) | Mu::ModParam(inner, _) => {
            charge(Primitive::DivModWire { width });
            count(inner, width, out);
        }
    }
}
