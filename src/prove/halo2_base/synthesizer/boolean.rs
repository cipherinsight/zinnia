//! Structural implications of constrained Boolean conjunctions.
use super::*;

impl Halo2BaseSynthesizer<'_> {
    pub(super) fn conjunction(&self, bit: &Halo2BaseCell) -> Vec<(u64, bool)> {
        self.and_literals.get(&bit.id).cloned().unwrap_or_else(|| {
            vec![self.negations.get(&bit.id).map_or((bit.id, true), |id| (*id, false))]
        })
    }

    pub(super) fn disjoint_bits(&self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> bool {
        if self.constant_truth(a) == Some(false) || self.constant_truth(b) == Some(false) { return true; }
        // Keep the identity of a compound conjunction when testing its own
        // negation: flattening only the positive side would hide this pair.
        if self.negations.get(&a.id) == Some(&b.id) || self.negations.get(&b.id) == Some(&a.id) { return true; }
        let a = self.conjunction(a); let b = self.conjunction(b);
        a.iter().any(|(root, sign)| b.contains(&(*root, !*sign)))
    }

    /// `a && !b == a` when conjunction a contains a literal contradicting
    /// conjunction b. This uses only existing bit gates, not witness values.
    pub(super) fn implies_negation(&self, a: &Halo2BaseCell, not_b: &Halo2BaseCell) -> bool {
        let Some(b) = self.negations.get(&not_b.id) else { return false };
        let a = self.conjunction(a);
        let b = self.and_literals.get(b).cloned().unwrap_or_else(|| vec![(*b, true)]);
        a.iter().any(|(root, sign)| b.contains(&(*root, !*sign)))
    }

    /// A masked sum equals one selected term under its indicator when all
    /// other indicators are disjoint with it. This does not assume that the
    /// indicators cover every case; pairwise disjointness with this term is
    /// sufficient to remove a guard selecting that same value.
    pub(super) fn redundant_indicator_select(&self, condition: &Halo2BaseCell,
        t: &Halo2BaseCell, f: &Halo2BaseCell) -> Option<Halo2BaseCell> {
        for (sum, plain, active) in [(t, f, self.negations.get(&condition.id).copied()), (f, t, Some(condition.id))] {
            let Some(active) = active else { continue };
            let Some(terms) = self.indicator_sums.get(&sum.id) else { continue };
            let Some((index, (chosen, _))) = terms.iter().enumerate().find(|(_, (bit, value))|
                bit.id == active && value.id == plain.id) else { continue };
            if terms.iter().enumerate().all(|(i, (bit, _))| i == index || self.disjoint_bits(chosen, bit)) {
                return Some(*sum);
            }
        }
        None
    }
}
