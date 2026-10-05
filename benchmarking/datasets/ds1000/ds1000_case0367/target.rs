// DS-1000 case 367 — Halo2 baseline.
// Kernel: `copy.deepcopy(array_of_arrays)`. `pairs` is unused.
// Mirrors Zinnia IR after DCE: 3× ReadInt (for array_of_arrays only)
// + 3× ExposePublicII + 2× AllocateMemoryIR.
// `pairs` is unused by the kernel; Zinnia v7 DCE eliminates the dead
// ReadIntegerIR ops, so HW skips the loads too.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub pairs: Vec<String>, pub array_of_arrays: Vec<String> }

fn deepcopy_id<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    let mut load = |ctx: &mut halo2_base::Context<F>, s: &str| {
        let abs = ctx.load_witness(F::from_u128(s.trim_start_matches('-').parse::<u128>().unwrap()));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };
    assert_eq!(input.pairs.len(), 6);
    // `pairs` is unused by the kernel; Zinnia v7 DCE eliminates the dead
    // ReadIntegerIR ops, so HW skips the loads too.
    for s in &input.array_of_arrays { make_public.push(load(ctx, s)); }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(deepcopy_id, args); }
