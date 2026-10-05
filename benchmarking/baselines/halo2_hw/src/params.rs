//! Choice of the circuit size `k` and the lookup table width.
//!
//! This mirrors Zinnia's halo2-base backend
//! (`src/prove/halo2_base/mod.rs::estimate_params`) line for line, so both
//! sides of the comparison size their circuits by the same rule:
//!
//! * `lookup_bits = min(8, k - 1)` (at least 1);
//! * the cell counts come from a dry synthesis at `k = 22` for that
//!   `lookup_bits` (counts do not depend on `k`, only on `lookup_bits`);
//! * among `k` in `4..=22` whose usable rows (`2^k - 20`) hold the instance
//!   column and the lookup table, pick the one minimizing
//!   `committed polynomials * 2^k`, with the polynomial count read from a
//!   halo2 `ConstraintSystem` configured for that column layout.
//!
//! If the harness defines a different policy, it passes `--k` and
//! `--lookup-bits` explicitly.

use std::collections::HashMap;

use halo2_base::gates::circuit::{BaseCircuitParams, BaseConfig};
use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
use halo2_base::halo2_proofs::plonk::ConstraintSystem;

use crate::Error;

pub const K_MIN: u32 = 4;
pub const K_MAX: u32 = 22;
pub const DEFAULT_LOOKUP_BITS: u32 = 8;
/// Rows reserved for blinding; Zinnia's `UNUSABLE_ROWS`.
pub const UNUSABLE_ROWS: usize = 20;

/// Cell counts of one synthesized circuit.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct CellStats {
    pub advice: usize,
    pub lookup: usize,
    pub fixed: usize,
    pub instances: usize,
}

/// Zinnia's lookup-width rule for a given `k`.
pub fn lookup_bits_for(k: u32) -> u32 {
    DEFAULT_LOOKUP_BITS.min(k.saturating_sub(1)).max(1)
}

/// Picks `(k, lookup_bits)`. `dry_run(lookup_bits)` must return the cell
/// statistics of the circuit synthesized with that lookup width.
pub fn choose(mut dry_run: impl FnMut(u32) -> Result<CellStats, Error>) -> Result<(u32, u32), Error> {
    let mut cache: HashMap<u32, CellStats> = HashMap::new();
    let mut best: Option<(u128, u32, u32)> = None;
    for k in K_MIN..=K_MAX {
        let lookup_bits = lookup_bits_for(k);
        let stats = match cache.get(&lookup_bits) {
            Some(s) => *s,
            None => {
                let s = dry_run(lookup_bits)?;
                cache.insert(lookup_bits, s);
                s
            }
        };
        if let Some(cost) = cost(k, lookup_bits, &stats) {
            if best.map_or(true, |(c, _, _)| cost < c) {
                best = Some((cost, k, lookup_bits));
            }
        }
    }
    best.map(|(_, k, lb)| (k, lb))
        .ok_or_else(|| Error::new(format!("no k in [{K_MIN}..{K_MAX}] fits the circuit's public instances")))
}

/// Zinnia's cost model; `None` if the circuit does not fit at this `k`.
fn cost(k: u32, lookup_bits: u32, stats: &CellStats) -> Option<u128> {
    let lookup_used = stats.lookup > 0;
    let usable = (1usize << k).saturating_sub(UNUSABLE_ROWS);
    if usable == 0 || stats.instances > usable {
        return None;
    }
    if lookup_used && (1usize << lookup_bits) > usable {
        return None;
    }
    let num_advice = stats.advice.div_ceil(usable).max(1);
    let num_lookup_advice = if lookup_used { stats.lookup.div_ceil(usable) } else { 0 };
    let num_fixed = stats.fixed.div_ceil(usable).max(1);

    let base_params = BaseCircuitParams {
        k: k as usize,
        num_advice_per_phase: vec![num_advice],
        num_fixed,
        num_lookup_advice_per_phase: vec![num_lookup_advice],
        lookup_bits: lookup_used.then_some(lookup_bits as usize),
        num_instance_columns: 1,
    };
    let mut meta = ConstraintSystem::<Fr>::default();
    let _config = BaseConfig::configure(&mut meta, base_params);
    let degree = meta.degree().max(3);
    let perm_chunk = (degree - 2).max(1);
    let perm_columns = meta.permutation().get_columns().len();
    let committed_polys = num_advice
        + num_lookup_advice
        + meta.lookups().len() * 2
        + perm_columns.div_ceil(perm_chunk)
        + (degree - 1)
        + 1;
    Some((committed_polys as u128) << k)
}
