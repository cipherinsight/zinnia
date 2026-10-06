//! The circuit the halo2-base backend keys, proves and verifies: halo2-base's
//! virtual regions, plus checks that use verifier challenges.
//!
//! `BaseCircuitBuilder`'s own `Circuit` implementation lays out the first
//! challenge phase only. Some checks are far cheaper with a random field
//! element the prover cannot predict when it fixes the values being checked
//! (a multiset equality as a grand product, Freivalds' check of a matrix
//! product, a polynomial identity for a convolution). halo2 supports this
//! natively: advice columns are committed phase by phase, and a challenge
//! declared `usable_after(FirstPhase)` is squeezed from the transcript after
//! every first-phase column is committed (Fiat-Shamir, the same assumption
//! halo2's own lookup and permutation arguments rest on).
//!
//! [`Halo2BaseCircuit`] is that two-phase circuit:
//!
//! * **First phase**: the cells the synthesizer builds before proving (a
//!   `BaseCircuitBuilder`, exactly as before, keyed and pinned the same way).
//! * **Challenges**: `n` halo2 challenges, each copied into a second-phase
//!   advice cell that a custom gate constrains to equal it
//!   (`q_j · (cell_j − challenge_j) = 0`). halo2-base's gates cannot read a
//!   challenge expression; from that cell on the challenge is an ordinary
//!   copy-constrained value.
//! * **Second phase**: the cells of every registered [`SecondPhaseCheck`],
//!   built inside `synthesize` (their values depend on the challenges, which
//!   exist only there), in second-phase gate columns, reading first-phase
//!   cells through copy constraints.
//!
//! **API.** A construction registers a check with
//! [`SecondPhase::push`] (the synthesizer exposes it as
//! `Halo2BaseSynthesizer::second_phase`), capturing the first-phase cells it
//! reads. `build` gets a second-phase context, the gate chip and the
//! challenge cells, and adds whatever cells and equalities the check needs.
//! The check must build the same cells whatever the values (the circuit's
//! shape is fixed at key generation), and it must be sound for a challenge
//! drawn uniformly after the first phase is fixed (its doc comment states
//! the Schwartz-Zippel argument).
//!
//! **Soundness of the plumbing.** The challenge cells are second-phase advice
//! pinned by a gate to the challenges; a prover who writes anything else
//! there fails that gate (test `a_wrong_challenge_cell_is_rejected`). The
//! first-phase cells a check reads are copy-constrained to its second-phase
//! cells. A program without checks has no challenge column and no
//! second-phase columns: its constraint system is `BaseCircuitBuilder`'s.
//!
//! **Layout.** The second phase gets its own gate columns
//! (`num_advice_per_phase[1]`), sized from a dry build of the checks
//! ([`SecondPhase::shape`]) by [`layout`]; its constants share the
//! first phase's fixed columns. In the prover stage the first phase is
//! witness-only (pinned break points, as before) and the second phase is
//! witness-only too, at break points recorded by the keygen synthesis
//! ([`Halo2BaseCircuit::break_points`] returns both phases'; the prover's
//! builder is pinned to them as before).

use std::sync::{Arc, Mutex};

use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, BaseConfig, MaybeRangeConfig};
use halo2_base::gates::flex_gate::threads::SinglePhaseCoreManager;
use halo2_base::gates::flex_gate::{MultiPhaseThreadBreakPoints, ThreadBreakPoints};
use halo2_base::gates::flex_gate::GateChip;
use halo2_base::halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    halo2curves::bn256::Fr as Fp,
    plonk::{Advice, Assigned, Challenge, Circuit, Column, ConstraintSystem, Error, FirstPhase, SecondPhase as Phase2, Selector},
    poly::Rotation,
};
use halo2_base::virtual_region::copy_constraints::SharedCopyConstraintManager;
use halo2_base::virtual_region::manager::VirtualRegionManager;
use halo2_base::{AssignedValue, Context};
use serde::{Deserialize, Serialize};

/// Rows reserved for blinding factors when packing cells into columns (the
/// backend's sizing rule; see `super::UNUSABLE_ROWS`).
use super::UNUSABLE_ROWS;

/// A check laid out in the second phase, after the verifier's challenges.
pub trait SecondPhaseCheck: Send + Sync {
    /// How many independent challenges it reads (`challenges[..n]` in
    /// [`Self::build`]).
    fn challenges(&self) -> usize;

    /// Adds the check's cells and constraints to `ctx` (a second-phase
    /// context), reading the first-phase cells it captured and the challenge
    /// cells. Must build the same cells for any values.
    fn build(&self, ctx: &mut Context<Fp>, gate: &GateChip<Fp>, challenges: &[AssignedValue<Fp>]);
}

/// The checks registered for the second phase, in registration order.
#[derive(Clone, Default)]
pub struct SecondPhase {
    checks: Vec<Arc<dyn SecondPhaseCheck>>,
}

/// The second phase's size: advice cells and distinct constants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseShape {
    pub advice: usize,
    pub constants: usize,
}

impl SecondPhase {
    pub fn push(&mut self, check: Arc<dyn SecondPhaseCheck>) {
        self.checks.push(check);
    }

    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }

    /// The number of challenges the circuit must declare (the most any
    /// check reads).
    pub fn challenges(&self) -> usize {
        self.checks.iter().map(|c| c.challenges()).max().unwrap_or(0)
    }

    /// Builds every check into `manager` (a second-phase manager) with the
    /// given challenge cells.
    fn build_into(&self, manager: &mut SinglePhaseCoreManager<Fp>, challenges: &[AssignedValue<Fp>]) {
        let gate = GateChip::<Fp>::default();
        for check in &self.checks {
            check.build(manager.main(), &gate, challenges);
        }
    }

    /// The cells and constants of the second phase, from a build with
    /// placeholder challenges in a scratch manager (the shape does not
    /// depend on values).
    pub fn shape(&self) -> PhaseShape {
        if self.is_empty() {
            return PhaseShape::default();
        }
        let copy: SharedCopyConstraintManager<Fp> = Default::default();
        let challenges: Vec<AssignedValue<Fp>> = (0..self.challenges())
            .map(|j| copy.lock().unwrap().mock_external_assigned(Fp::from(j as u64 + 2)))
            .collect();
        let mut manager = SinglePhaseCoreManager::new(false, copy.clone()).in_phase(1);
        self.build_into(&mut manager, &challenges);
        let constants = {
            let cm = copy.lock().unwrap();
            let mut values: Vec<Fp> = cm.constant_equalities.iter().map(|(c, _)| *c).collect();
            values.sort();
            values.dedup();
            values.len()
        };
        let shape = PhaseShape { advice: manager.total_advice(), constants };
        discard(&copy);
        shape
    }
}

/// Empties a copy manager that will not be laid out (halo2-base warns when
/// one is dropped with pending equalities).
fn discard(copy: &SharedCopyConstraintManager<Fp>) {
    copy.lock().unwrap().clear();
}

/// The circuit's parameters: halo2-base's column layout and the number of
/// challenges (0: a one-phase circuit, `BaseCircuitBuilder`'s).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CircuitParams {
    pub base: BaseCircuitParams,
    #[serde(default)]
    pub challenges: usize,
}

/// The challenges and the second-phase cells pinned to them.
#[derive(Clone, Debug)]
pub struct ChallengeConfig {
    challenges: Vec<Challenge>,
    column: Column<Advice>,
    /// `selectors[j]` enables `cell − challenge_j = 0` on row `j`.
    selectors: Vec<Selector>,
}

impl ChallengeConfig {
    fn configure(meta: &mut ConstraintSystem<Fp>, n: usize) -> Self {
        let column = meta.advice_column_in(Phase2);
        meta.enable_equality(column);
        let challenges: Vec<Challenge> = (0..n).map(|_| meta.challenge_usable_after(FirstPhase)).collect();
        let selectors: Vec<Selector> = (0..n).map(|_| meta.selector()).collect();
        for (challenge, selector) in challenges.iter().zip(&selectors) {
            meta.create_gate("challenge cell", |v| {
                let q = v.query_selector(*selector);
                let cell = v.query_advice(column, Rotation::cur());
                vec![q * (cell - v.query_challenge(*challenge))]
            });
        }
        ChallengeConfig { challenges, column, selectors }
    }
}

#[derive(Clone, Debug)]
pub struct Halo2BaseConfig {
    pub base: BaseConfig<Fp>,
    pub challenges: Option<ChallengeConfig>,
}

/// halo2-base's first-phase virtual regions plus the second-phase checks
/// (see the module documentation).
pub struct Halo2BaseCircuit {
    pub builder: BaseCircuitBuilder<Fp>,
    pub second_phase: SecondPhase,
    /// The second phase's break points, recorded when a non-prover stage
    /// lays it out (see [`Self::break_points`]).
    second_break_points: Mutex<Option<ThreadBreakPoints>>,
    /// Added to every challenge before it is written to its cell: a
    /// cheating prover, for tests only (the gate must reject it).
    #[cfg(test)]
    pub challenge_offset: Fp,
}

impl Halo2BaseCircuit {
    pub fn new(builder: BaseCircuitBuilder<Fp>, second_phase: SecondPhase) -> Self {
        Halo2BaseCircuit {
            builder,
            second_phase,
            second_break_points: Mutex::new(None),
            #[cfg(test)]
            challenge_offset: Fp::zero(),
        }
    }

    /// The thread break points of every phase, after a keygen or mock
    /// synthesis: what a prover-stage circuit is pinned to
    /// (`BaseCircuitBuilder::prover`), the second phase's included.
    pub fn break_points(&self) -> MultiPhaseThreadBreakPoints {
        let mut points = self.builder.break_points();
        points.truncate(1);
        if let Some(second) = self.second_break_points.lock().unwrap().clone() {
            points.push(second);
        }
        points
    }

    /// The layout of the second phase's challenge cells: one row per
    /// challenge.
    fn assign_challenges(
        &self,
        config: &ChallengeConfig,
        layouter: &mut impl Layouter<Fp>,
        copy: &SharedCopyConstraintManager<Fp>,
    ) -> Result<Vec<AssignedValue<Fp>>, Error> {
        #[cfg(test)]
        let offset = self.challenge_offset;
        #[cfg(not(test))]
        let offset = Fp::zero();
        let values: Vec<Value<Fp>> =
            config.challenges.iter().map(|c| layouter.get_challenge(*c).map(|v| v + offset)).collect();
        let cells = layouter.assign_region(
            || "challenges",
            |mut region| {
                let mut cells = Vec::with_capacity(values.len());
                for (row, (selector, value)) in config.selectors.iter().zip(&values).enumerate() {
                    selector.enable(&mut region, row)?;
                    cells.push(region.assign_advice(|| "challenge", config.column, row, || value.map(Assigned::Trivial))?);
                }
                Ok(cells)
            },
        )?;
        let mut copy = copy.lock().unwrap();
        Ok(cells.into_iter().map(|cell| copy.load_external_assigned(cell)).collect())
    }
}

impl Circuit<Fp> for Halo2BaseCircuit {
    type Config = Halo2BaseConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = CircuitParams;

    fn params(&self) -> CircuitParams {
        CircuitParams { base: self.builder.config_params.clone(), challenges: self.second_phase.challenges() }
    }

    fn without_witnesses(&self) -> Self {
        unimplemented!("keys are generated from a witness-less synthesis instead")
    }

    fn configure(_: &mut ConstraintSystem<Fp>) -> Halo2BaseConfig {
        unreachable!("configure_with_params")
    }

    fn configure_with_params(meta: &mut ConstraintSystem<Fp>, params: CircuitParams) -> Halo2BaseConfig {
        let k = params.base.k;
        let mut base = BaseConfig::configure(meta, params.base);
        let challenges = (params.challenges > 0).then(|| ChallengeConfig::configure(meta, params.challenges));
        // The challenge column may add blinding rows.
        base.set_usable_rows((1 << k) - meta.minimum_rows());
        Halo2BaseConfig { base, challenges }
    }

    fn synthesize(&self, config: Halo2BaseConfig, mut layouter: impl Layouter<Fp>) -> Result<(), Error> {
        let builder = &self.builder;
        config.base.initialize(&mut layouter);
        let usable = config.base.gate().max_rows;
        layouter.assign_region(
            || "first phase",
            |mut region| {
                builder.core().phase_manager[0].assign_raw(&(config.base.gate().basic_gates[0].clone(), usable), &mut region);
                if let MaybeRangeConfig::WithRange(range) = &config.base.base {
                    builder.assign_lookups_in_phase(range, &mut region, 0);
                }
                Ok(())
            },
        )?;

        // The second phase keeps its own copy manager: it is rebuilt on every
        // call of `synthesize` and must not accumulate in the first phase's.
        // A prover-stage circuit lays it out as witnesses only, at the break
        // points the keygen synthesis recorded (pinned with the first phase's).
        let second_copy: SharedCopyConstraintManager<Fp> = Default::default();
        let mut second = SinglePhaseCoreManager::new(builder.witness_gen_only(), second_copy.clone()).in_phase(1);
        if builder.witness_gen_only() {
            let pinned = builder.core().phase_manager.get(1).and_then(|pm| pm.break_points.borrow().clone());
            *second.break_points.borrow_mut() = pinned;
        }
        if let Some(challenge_config) = &config.challenges {
            let challenges = self.assign_challenges(challenge_config, &mut layouter, &second_copy)?;
            self.second_phase.build_into(&mut second, &challenges);
            layouter.assign_region(
                || "second phase",
                |mut region| {
                    second.assign_raw(&(config.base.gate().basic_gates[1].clone(), usable), &mut region);
                    Ok(())
                },
            )?;
            if !builder.witness_gen_only() {
                *self.second_break_points.lock().unwrap() = second.break_points.borrow().clone();
            }
        }

        if builder.witness_gen_only() {
            // The prover lays out values only; equalities are in the keys.
            discard(&second_copy);
        } else {
            let first_copy = &builder.core().copy_manager;
            layouter.assign_region(
                || "copy constraints",
                |mut region| {
                    first_copy.assign_raw(config.base.constants(), &mut region);
                    Ok(())
                },
            )?;
            if config.challenges.is_some() {
                // The second phase's equalities name first-phase cells too.
                let first_cells = first_copy.lock().unwrap().assigned_advices.clone();
                second_copy.lock().unwrap().assigned_advices.extend(first_cells);
                layouter.assign_region(
                    || "second-phase copy constraints",
                    |mut region| {
                        second_copy.assign_raw(config.base.constants(), &mut region);
                        Ok(())
                    },
                )?;
            }
        }
        builder.assign_instances(&config.base.instance, layouter.namespace(|| "public values"));
        Ok(())
    }
}

/// Sets the circuit's column layout at its `k`: halo2-base's for the first
/// phase (`calculate_params`, reserving `UNUSABLE_ROWS` rows per column, the
/// constants' fixed columns sized like the others), plus the second phase's
/// gate columns and constants from [`SecondPhase::shape`].
pub fn layout(circuit: &mut Halo2BaseCircuit) {
    let builder = &mut circuit.builder;
    let mut params = builder.calculate_params(Some(UNUSABLE_ROWS));
    let usable = (1usize << params.k).saturating_sub(UNUSABLE_ROWS).max(1);
    let first_constants = builder.statistics().gate.total_fixed;
    let second = circuit.second_phase.shape();
    if !circuit.second_phase.is_empty() {
        params.num_advice_per_phase.truncate(1);
        params.num_advice_per_phase.push(second.advice.div_ceil(usable).max(1));
        params.num_lookup_advice_per_phase.truncate(1);
    }
    params.num_fixed = if second.constants == 0 {
        first_constants.div_ceil(usable).max(params.num_fixed)
    } else {
        // The two phases' constants are laid out in two regions of the same
        // fixed columns, one below the other.
        let mut num_fixed = params.num_fixed.max(1);
        while first_constants.div_ceil(num_fixed) + second.constants.div_ceil(num_fixed) > usable {
            num_fixed += 1;
        }
        num_fixed
    };
    builder.set_params(params);
}

#[cfg(test)]
mod tests;
