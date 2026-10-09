//! Pack exact arithmetic trees without moving a fixed-point rounding boundary.
use super::plan::{AffineTerm, AffineTree};
use super::ranges::Interval;
use super::*;

impl Halo2BaseSynthesizer<'_> {
    /// Called once at the entry to an IR arithmetic operation. Internal tree
    /// nodes have only one exact-arithmetic consumer and no provenance reader.
    /// Their placeholders never reach a gate; the root reads the original leaves.
    pub(super) fn planned_affine(&mut self, value: Fp) -> Result<Option<Halo2BaseCell>, ProvingError> {
        if self.plan.affine_links.contains(&self.current_stmt) || self.plan.select_add_links.contains(&self.current_stmt) {
            return Ok(Some(self.cellless(value)));
        }
        let Some(AffineTree { terms }) = self.plan.affine_roots.remove(&self.current_stmt) else {
            return Ok(None);
        };
        let mut pairs = Vec::with_capacity(terms.len());
        let mut range = Some(Interval { lo: 0, hi: 0 });
        for term in terms {
            let get = |s: &Self, vid| s.cell_of_vid.get(&vid).copied()
                .ok_or_else(|| ProvingError::synthesis("affine leaf without a cell"));
            let (left, right) = match term {
                AffineTerm::Linear(vid, coefficient) => {
                    let leaf = get(self, vid)?;
                    if let Some((a, b)) = self.deferred.get(&leaf.id).copied() {
                        if coefficient == 1 {
                            (a, b)
                        } else if self.is_const_cell(&b) {
                            (a, self.const_cell(b.value * kernel::i128_to_fp(coefficient)))
                        } else if self.is_const_cell(&a) {
                            (b, self.const_cell(a.value * kernel::i128_to_fp(coefficient)))
                        } else {
                            let leaf = self.materialize(&leaf);
                            (leaf, self.const_cell(kernel::i128_to_fp(coefficient)))
                        }
                    } else {
                        (leaf, self.const_cell(kernel::i128_to_fp(coefficient)))
                    }
                }
                AffineTerm::Product(a, b) => (get(self, a)?, get(self, b)?),
            };
            if (self.is_const_cell(&left) && left.value == Fp::zero())
                || (self.is_const_cell(&right) && right.value == Fp::zero()) { continue; }
            range = range.and_then(|r| self.proven_range(&left)
                .zip(self.proven_range(&right)).and_then(|(a, b)| a.mul(&b)).and_then(|p| r.add(&p)));
            pairs.push((left, right));
        }
        if pairs.is_empty() { return Ok(Some(self.const_cell(Fp::zero()))); }
        if pairs.len() == 1 && self.is_const_cell(&pairs[0].1) && pairs[0].1.value == Fp::one() {
            return Ok(Some(pairs[0].0));
        }
        // Put a unit coefficient first so inner_product omits the initial zero
        // accumulator. This is exact field reassociation, never a Q rescale.
        if let Some(first) = pairs.iter().position(|(_, b)| self.is_const_cell(b) && b.value == Fp::one()) {
            pairs.swap(0, first);
        }
        let value = pairs.iter().fold(Fp::zero(), |acc, (a, b)| acc + a.value * b.value);
        let assigned = self.gate.inner_product(self.builder.main(0),
            pairs.iter().map(|(a, _)| a.quantum()), pairs.iter().map(|(_, b)| b.quantum()));
        let out = self.wrap(assigned, value);
        let indicators: Option<Vec<_>> = pairs.iter().map(|(a, b)| {
            if self.is_bit(a) { Some((*a, *b)) } else if self.is_bit(b) { Some((*b, *a)) } else { None }
        }).collect();
        if let Some(terms) = indicators { self.indicator_sums.insert(out.id, terms); }
        self.last_output = Some((out.id, self.builder.main(0).advice.len()));
        if let Some(range) = range { self.prove_range(&out, range); }
        Ok(Some(out))
    }
}
