//! Inner products over one-hot indicators (`plan::OneHotChain`): a select
//! chain over the equality bits of one bounded index is `Σ_v [root == v] ·
//! x_v`, with the indicators `[root == v]` shared by every chain on the
//! same root.

use std::collections::HashMap;

use halo2_base::gates::GateInstructions;

use super::{Halo2BaseCell, Halo2BaseSynthesizer};
use crate::prove::error::ProvingError;
use crate::prove::kernel;
use crate::prove::traits::Synthesizer;

/// The indicators built so far, by (root cell id, value).
#[derive(Default)]
pub(super) struct Indicators(HashMap<(u64, i64), Halo2BaseCell>);

impl Halo2BaseSynthesizer<'_> {
    /// `[root == v]`, shared by every chain on the same root.
    pub(super) fn root_indicator(&mut self, root: &Halo2BaseCell, v: i64) -> Result<Halo2BaseCell, ProvingError> {
        if let Some(ind) = self.indicators.0.get(&(root.id, v)) {
            return Ok(*ind);
        }
        let v_cell = self.const_cell(kernel::i64_to_fp(v));
        let ind = self.eq_i(root, &v_cell)?;
        self.indicators.0.insert((root.id, v), ind);
        Ok(ind)
    }

    /// `Σ indicator · value` for indicators exactly one of which is 1 (the
    /// caller's argument): the selected value.
    pub(super) fn one_hot_combination(&mut self, choices: &[(Halo2BaseCell, Halo2BaseCell)]) -> Halo2BaseCell {
        let value = choices.iter().map(|(ind, cell)| ind.value * cell.value).sum();
        self.gate_cell(value, |gate, ctx| {
            gate.inner_product(ctx, choices.iter().map(|(_, cell)| cell.quantum()), choices.iter().map(|(ind, _)| ind.quantum()))
        })
    }
}
