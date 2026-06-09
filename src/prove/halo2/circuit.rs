//! The halo2 `Circuit` implementation for Zinnia IR programs.

use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};

use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::halo2::config::ZinniaConfig;
use crate::prove::halo2::synthesizer::Halo2Synthesizer;
use crate::prove::interpreter::interpret_ir;
use crate::prove::types::ProvingParams;

#[derive(Clone, Copy, Debug, Default)]
pub struct ZinniaCircuitParams {
    pub wide_select_many: bool,
}

pub fn params_for_ir(ir: &IRGraph) -> ZinniaCircuitParams {
    let wide_select_many = ir.stmts.iter().any(|stmt| match stmt.ir {
        IR::SelectManyI { len, .. } | IR::SelectManyF { len, .. } => len >= 5,
        _ => false,
    });
    ZinniaCircuitParams { wide_select_many }
}

/// A halo2 circuit constructed from a Zinnia IR program.
///
/// During keygen, `witness` is `None` — the synthesizer will use zero values.
/// During proving, `witness` contains the concrete input values.
#[derive(Clone)]
pub struct ZinniaCircuit {
    pub ir: IRGraph,
    pub witness: Option<ResolvedWitness>,
    pub params: ProvingParams,
}

impl Circuit<Fp> for ZinniaCircuit {
    type Config = ZinniaConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ZinniaCircuitParams;

    fn without_witnesses(&self) -> Self {
        Self {
            ir: self.ir.clone(),
            witness: None,
            params: self.params.clone(),
        }
    }

    fn configure_with_params(
        meta: &mut ConstraintSystem<Fp>,
        params: ZinniaCircuitParams,
    ) -> ZinniaConfig {
        ZinniaConfig::configure_with_wide_select_many(meta, params.wide_select_many)
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> ZinniaConfig {
        ZinniaConfig::configure(meta)
    }

    fn params(&self) -> Self::Params {
        params_for_ir(&self.ir)
    }

    fn synthesize(
        &self,
        config: ZinniaConfig,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        // Phase 1: Record operations by interpreting the IR.
        let mut synth =
            Halo2Synthesizer::new(config.clone(), self.witness.clone(), self.params.clone());

        interpret_ir(&self.ir, &mut synth).map_err(|e| {
            eprintln!("Zinnia synthesis error: {}", e);
            Error::Synthesis
        })?;

        // Phase 2: Replay recorded operations into the halo2 region.
        synth.load_range_table(&mut layouter)?;
        let assigned = layouter.assign_region(
            || "zinnia_main",
            |mut region| synth.replay_into_region(&mut region),
        )?;

        // Phase 3: Expose public outputs to instance column.
        synth.expose_instances(&mut layouter, &assigned)?;

        Ok(())
    }
}
