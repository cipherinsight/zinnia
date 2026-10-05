//! `COUNT` (`revision/phase1/rules/masks/count.md`).
//!
//! Puts a count variable `c = Count(m)` of Φ on a wire. The wire is the last
//! rank `r_B` of a `PREFIX_RANK` node over a bundle of m, and `COUNT` is
//! that node's second output bundle (`rules/README.md` §3.4, §4.5), not a
//! node of its own: it has no port and no cost beyond `PREFIX_RANK`'s.
//!
//! The rule fills the row of m in the three Store families under ψ_m and
//! offers, per ordered bundle it finds, the very candidate
//! [`super::prefix_rank::node`] offers in that bundle's `Ranks` row. The two
//! offers are one node, so whichever of the two requirements phase 2 selects
//! it through, its cost is counted once.
//!
//! **Which row.** `rules/README.md` §3.2 makes the row of a scalar value of
//! Δ and the row of its term one row, so a count reaches this rule as a term
//! requirement `Represent(c, Wire[Idx])` or as the row of `len(a[m])`,
//! `nonzero(m).shape[0]` and their kin. Both are matched; the node's output
//! names the canonical one of the two ([`Builder::wire_target`]), which is
//! the row the provider table sends every occurrence of c to.

use crate::phase0::desc::ValueKey;
use crate::phase0::phi::VarOrigin;
use crate::phase0::term::TermId;

use crate::phase1::build::Builder;
use crate::phase1::graph::Target;
use crate::phase1::interface::Family;
use crate::phase1::rules::{Candidate, Rule, RuleId};
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

pub(crate) struct Count;

/// The families a bundle of the mask may be held in (`count.md`,
/// `source_pattern`).
const BUNDLES: [Family; 3] = [Family::Store, Family::Masked, Family::Packed];

impl Rule for Count {
    fn id(&self) -> RuleId {
        RuleId::Count
    }

    fn answers(&self) -> &'static [Family] {
        &[Family::Wire]
    }

    fn build(
        &self,
        builder: &mut Builder<'_>,
        request: &RowRequest,
    ) -> Result<Vec<Candidate>, Reject> {
        let Some(mask) = self.matches(builder, request) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        // One node per ordered bundle the row of m holds, exactly as the
        // `Ranks` rows of those bundles build them.
        let mut probe = Candidate::new(RuleId::Count, None, request.epoch);
        let psi = builder.psi(&mask);
        let condition = builder.child_under_g(&mut probe, request, psi, &mask.encode())?;
        for family in BUNDLES {
            let row = builder.child(request, Target::Value(mask.clone()), family, condition)?;
            for entry in &row.entries.clone() {
                let Some(subject) = super::subject_of(&entry.interface, &mask, false) else {
                    continue;
                };
                if let Some(candidate) =
                    super::prefix_rank::node(builder, request, &mask, &subject)?
                {
                    out.push(candidate);
                }
            }
        }
        Ok(out)
    }
}

impl Count {
    /// `source_pattern`: the row is a count variable's wire, and the mask it
    /// counts is the defining key the variable's origin names
    /// (`masks/README.md` §2).
    fn matches(&self, builder: &Builder<'_>, request: &RowRequest) -> Option<ValueKey> {
        let term = match &request.target {
            Target::Term(t) => *t,
            // The row of a scalar of Δ whose term is c is the same row
            // (§3.2): `len(a[m])`, `a[m].size`, `nonzero(m).shape[0]`.
            Target::Value(key) => builder.scalar(key)?.term.as_int()?,
            Target::BoolTerm(_) => return None,
        };
        self.mask_of(builder, term)
    }

    /// The mask a term counts, when the term is the bare count variable.
    fn mask_of(&self, builder: &Builder<'_>, term: TermId) -> Option<ValueKey> {
        let variable = builder.terms.var_of(term)?;
        match builder.var_origin(variable) {
            VarOrigin::Count { mask } => Some(mask.clone()),
            _ => None,
        }
    }
}
