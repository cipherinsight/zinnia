//! `CUMSUM` (`revision/phase1/rules/reduction/cumsum.md`).
//!
//! Computes a prefix scan: every element of `w = cumsum(v, axis=k)` is the
//! running sum of the elements of v at or before it along axis k; `cumprod`
//! is the running product. The rule reads v as fibers along k, laid out by
//! `EXPAND_FIBERS`, and produces w in the same slots with the validity wires
//! of v aliased: an invalid slot carries the running value, so a slot that is
//! not an element of v neither changes nor breaks the scan.

use crate::phase0::desc::ValueKey;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;

use crate::phase1::build::Builder;
use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::interface::{Family, Interface, LayoutKind, StoreIface};
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Cumsum;

impl Rule for Cumsum {
    fn id(&self) -> RuleId {
        RuleId::Cumsum
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Store, Family::Masked, Family::Packed]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(key) = request.target.as_value() else {
            return Ok(Vec::new());
        };
        let key = key.clone();
        let Some((operand, product)) = self.matches(builder, &key) else {
            return Ok(Vec::new());
        };
        let Some(array) = builder.array(&key).cloned() else {
            return Ok(Vec::new());
        };
        if !super::super::sources::encoded(builder, "CUMSUM", array.dtype) {
            return Ok(Vec::new());
        }
        let Some(node) = builder.node(&key).cloned() else {
            return Ok(Vec::new());
        };
        // `axis` is a constant normalised to [0, r); without one phase 0
        // flattens first, so the scan axis is 0 of the flat value.
        let axis = super::axes_of(&node, array.rank(), 1, "axis");
        let axis = match axis.len() {
            1 => axis[0],
            _ => 0,
        };

        let mut probe = Candidate::new(RuleId::Cumsum, None, request.epoch);
        let groups = builder.groups_port(&mut probe, request, &operand, vec![axis])?;
        let mut out = Vec::new();
        for entry in &groups.entries.clone() {
            let Interface::Groups(fibers) = &entry.interface else {
                continue;
            };
            // ORDER: a scan is order-sensitive for every dtype.
            if !fibers.flags.ordered {
                continue;
            }
            // The output is produced in the interface of the slots v was read
            // in, so a row of another family is another node's.
            let family = match &fibers.slots.layout {
                LayoutKind::Box(_) => Family::Store,
                LayoutKind::Compact(_) => Family::Packed,
                LayoutKind::Ranked { .. } => Family::Masked,
                LayoutKind::Keyed { .. } => continue,
            };
            if family != request.family {
                continue;
            }
            let mut candidate = Candidate::new(RuleId::Cumsum, None, request.epoch);
            candidate.answers = probe.answers.clone();
            candidate.matched = vec![key.clone(), operand.clone()];
            candidate.inputs.push(entry.requirement);
            candidate.epoch = candidate.epoch.max(entry.epoch);

            // emitter: one step per slot, with the conditional update where
            // the fibers are not prefix-valid (CARRY-DROP). An integer scan
            // wraps at every step and a `cumprod` costs one product more; a
            // Float scan is one `fop⟨add_f⟩` — or `fop⟨mul_f⟩` — per slot and
            // no wrap, a Float wire having no width (`rules/README.md` §5).
            let slots = fibers.fibers * fibers.width;
            let conditional = i128::from(!(fibers.flags.prefix || fibers.flags.zeroed));
            if super::super::sources::is_float(array.dtype) {
                let step = if product { FloatOp::Mul } else { FloatOp::Add };
                candidate = candidate
                    .emit(Primitive::Float(FloatOp::Select), slots * conditional)
                    .emit(Primitive::Float(step), slots);
            } else {
                let width = builder.omega.round(builder.dtype_width(array.dtype));
                candidate = candidate
                    .emit(Primitive::Select, slots * conditional)
                    .emit(Primitive::Mul, slots * i128::from(product));
                if width > 0 {
                    candidate = candidate.emit(
                        Primitive::Wrap { width, carry_width: builder.omega.round(1) },
                        slots,
                    );
                }
            }
            let capacity = fibers.fibers * fibers.width;
            let interface = match family {
                Family::Store => Interface::Store(StoreIface {
                    owner: key.clone(),
                    dtype: array.dtype,
                    capacity,
                    lambda: fibers.slots.clone(),
                }),
                Family::Packed => Interface::Packed {
                    dtype: array.dtype,
                    capacity,
                    lambda: fibers.slots.clone(),
                    canon: false,
                },
                _ => Interface::Masked {
                    dtype: array.dtype,
                    capacity,
                    lambda: fibers.slots.clone(),
                },
            };
            candidate = candidate
                .output(request.target.clone(), interface, request.condition)
                .parameter("P", fibers.fibers)
                .parameter("W", fibers.width);
            out.push(candidate);
        }
        Ok(out)
    }
}

impl Cumsum {
    /// `source_pattern`: the producer names E-Cumsum, and the AST node says
    /// whether the scan is a sum or a product.
    fn matches(&self, builder: &Builder<'_>, key: &ValueKey) -> Option<(ValueKey, bool)> {
        let entry = builder.entry(key)?;
        let Producer::Applied { rules, operands, .. } = &entry.producer else {
            return None;
        };
        if !rules.iter().any(|rule| matches!(rule, Phase0Rule::ECumsum)) {
            return None;
        }
        let operand = operands
            .iter()
            .find(|operand| {
                builder.array(operand).map(|array| array.rank() > 0).unwrap_or(false)
            })?
            .clone();
        let member = builder.node(key)?.member().unwrap_or("cumsum");
        Some((operand, member.contains("prod")))
    }
}
