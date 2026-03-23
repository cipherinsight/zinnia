//! The halo2 `Circuit` implementation for Zinnia IR programs.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use pasta_curves::Fp;

use crate::ir::IRGraph;
use crate::prove::halo2::config::ZinniaConfig;
use crate::prove::halo2::synthesizer::Halo2Synthesizer;
use crate::prove::interpreter::interpret_ir;
use crate::prove::types::{ProvingParams, WitnessInput};

/// A halo2 circuit constructed from a Zinnia IR program.
///
/// During keygen, `witness` is `None` — the synthesizer will use zero values.
/// During proving, `witness` contains the concrete input values.
#[derive(Clone)]
pub struct ZinniaCircuit {
    pub ir: IRGraph,
    pub witness: Option<WitnessInput>,
    pub params: ProvingParams,
}

impl Circuit<Fp> for ZinniaCircuit {
    type Config = ZinniaConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            ir: self.ir.clone(),
            witness: None,
            params: self.params.clone(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> ZinniaConfig {
        ZinniaConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: ZinniaConfig,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        // Phase 1: Record operations by interpreting the IR.
        let mut synth = Halo2Synthesizer::new(
            config.clone(),
            self.witness.clone(),
            self.params.clone(),
        );

        interpret_ir(&self.ir, &mut synth).map_err(|e| {
            eprintln!("Zinnia synthesis error: {}", e);
            Error::Synthesis
        })?;

        // Phase 2: Replay recorded operations into the halo2 region.
        let assigned = layouter.assign_region(
            || "zinnia_main",
            |mut region| synth.replay_into_region(&mut region),
        )?;

        // Phase 3: Expose public outputs to instance column.
        synth.expose_instances(&mut layouter, &assigned)?;

        Ok(())
    }
}
