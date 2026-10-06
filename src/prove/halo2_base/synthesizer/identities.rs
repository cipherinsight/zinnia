//! `IR::IdentityCheck`: the identity registered as a second-phase check
//! (`super::super::identities`), reading the operands' first-phase cells.

use std::sync::Arc;

use super::ranges::Interval;
use super::{Halo2BaseCell, Halo2BaseSynthesizer};
use crate::ir_defs::Identity;
use crate::prove::error::ProvingError;
use crate::prove::halo2_base::circuit::SecondPhaseCheck;
use crate::prove::halo2_base::identities::{MatrixProduct, MultisetEquality, PolynomialProduct};
use crate::prove::kernel;

impl Halo2BaseSynthesizer<'_> {
    /// Registers the check of `identity` among `operands` for the second
    /// phase. With a witness, a false identity is noted as unsatisfied (as
    /// a failed assertion is) before any proving work.
    ///
    /// A multiset equality also bounds its right-hand values: in every
    /// accepted witness (except with the check's soundness error) each
    /// right-hand component equals some left-hand one, so it lies in the
    /// hull of the left-hand components' proven ranges. Later gadgets may
    /// size their checks by that range (an asserted order of hinted sorted
    /// keys, `lower::sort::grand_product`); the check is registered before
    /// they are synthesized, as the IR orders them.
    pub(super) fn check_identity(&mut self, identity: Identity, operands: &[Halo2BaseCell]) -> Result<(), ProvingError> {
        let values: Vec<_> = operands.iter().map(|c| c.value).collect();
        if !kernel::identity_holds(identity, &values) {
            self.note_unsat(|| format!("Witness does not satisfy circuit constraints: identity {}", identity.describe()));
        }
        let cells: Vec<_> = operands.iter().map(|c| self.assigned(c)).collect();
        let check: Arc<dyn SecondPhaseCheck> = match identity {
            Identity::MultisetEqual { len, width } => {
                let (len, width) = (len as usize, width as usize);
                if width == 0 {
                    return Err(ProvingError::synthesis("multiset equality of empty tuples"));
                }
                self.bound_by_left_side(operands, len, width);
                let (left, right) = cells.split_at(len * width);
                Arc::new(MultisetEquality { left: left.to_vec(), right: right.to_vec(), width })
            }
            Identity::PolynomialProduct { n, m } => {
                let (n, m) = (n as usize, m as usize);
                if n == 0 || m == 0 {
                    return Err(ProvingError::synthesis("polynomial product of an empty polynomial"));
                }
                Arc::new(PolynomialProduct {
                    a: cells[..n].to_vec(),
                    b: cells[n..n + m].to_vec(),
                    c: cells[n + m..].to_vec(),
                })
            }
            Identity::MatrixProduct { rows, inner, cols } => {
                let (rows, inner, cols) = (rows as usize, inner as usize, cols as usize);
                if rows == 0 || inner == 0 || cols == 0 {
                    return Err(ProvingError::synthesis("matrix product with an empty dimension"));
                }
                let (a, rest) = cells.split_at(rows * inner);
                let (b, c) = rest.split_at(inner * cols);
                Arc::new(MatrixProduct { a: a.to_vec(), b: b.to_vec(), c: c.to_vec(), rows, inner, cols })
            }
        };
        self.second_phase.push(check);
        Ok(())
    }

    /// Component `l` of every right-hand tuple lies in the hull of the
    /// proven ranges of component `l` of the left-hand tuples (when all of
    /// those are known).
    fn bound_by_left_side(&mut self, operands: &[Halo2BaseCell], len: usize, width: usize) {
        let (left, right) = operands.split_at(len * width);
        for l in 0..width {
            let hull = left
                .iter()
                .skip(l)
                .step_by(width)
                .map(|c| self.proven_range(c))
                .try_fold(None::<Interval>, |acc, r| r.map(|r| Some(acc.map_or(r, |a| a.hull(&r)))));
            if let Some(Some(hull)) = hull {
                for cell in right.iter().skip(l).step_by(width) {
                    self.prove_range(cell, hull);
                }
            }
        }
    }
}
