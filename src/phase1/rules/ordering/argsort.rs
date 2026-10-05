//! `ARGSORT` (`revision/phase1/rules/ordering/argsort.md`).
//!
//! `v = argsort(a, axis=k)`: along axis k, the logical indices of the
//! elements in ascending order of value, ties broken by the index. The
//! construction is `SORT`'s with three changes, which
//! [`super::sort::build_sort`] reads off the match: the permuted tuples
//! carry the logical index of each element, the order check is strict on the
//! combined `(value, index)` key and so `b = bits(B)` bits wider, and the
//! index vector is one witness column more.
//!
//! **The second match key.** When Δ also holds `sort(a, axis=k)` — the same
//! operand, the same normalised axis, the same ψ, the dtype of a — the
//! sorted values are not internal but a second bundle of the same wires, and
//! the node is attached to that value's rows too (§3.4). The row of the
//! sorted value then holds both this node and `SORT`'s own, and phase 2
//! chooses.

use crate::phase0::desc::ValueKey;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;

use crate::phase1::build::Builder;
use crate::phase1::interface::Family;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Argsort;

impl Rule for Argsort {
    fn id(&self) -> RuleId {
        RuleId::Argsort
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Packed, Family::Masked, Family::Store]
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
        let Some(matched) = super::sort::matches(builder, &key) else {
            return Ok(Vec::new());
        };
        if !matched.indices {
            return Ok(Vec::new());
        }
        let sorted = self.sorted_sibling(builder, &key, &matched);
        super::sort::build_sort(
            builder,
            request,
            &key,
            &matched,
            RuleId::Argsort,
            sorted.as_ref(),
        )
    }
}

impl Argsort {
    /// The value of Δ that is `sort(a, axis=k)` for the same operand and
    /// axis, when the program computes one (`argsort.md`, `source_pattern`).
    fn sorted_sibling(
        &self,
        builder: &Builder<'_>,
        key: &ValueKey,
        matched: &super::sort::Matched,
    ) -> Option<ValueKey> {
        let psi = builder.psi(key);
        let dtype = builder.array(&matched.operand)?.dtype;
        builder
            .result
            .descriptors
            .iter()
            .find(|entry| {
                if entry.key == *key {
                    return false;
                }
                let Producer::Applied { rules, operands, .. } = &entry.producer else {
                    return false;
                };
                if !rules.contains(&Phase0Rule::ESort) || !operands.contains(&matched.operand) {
                    return false;
                }
                let Some(array) = entry.desc.as_array() else {
                    return false;
                };
                array.dtype == dtype
                    && entry.desc.psi() == Some(psi)
                    && super::sort::matches(builder, &entry.key)
                        .map(|other| !other.indices && other.axis == matched.axis)
                        .unwrap_or(false)
            })
            .map(|entry| entry.key.clone())
    }
}
