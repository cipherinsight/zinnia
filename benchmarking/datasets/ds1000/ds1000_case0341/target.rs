// DS-1000 case 340 — Halo2 baseline. Flatten `X.flat` on Integer[5, 6].
// Mirrors Zinnia IR: 30× ReadInt + 30× ExposePublicII (identity-like flatten).
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub x: Vec<String> }  // 30 ints, row-major

fn flatten_circuit<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    assert_eq!(input.x.len(), 30);
    for s in &input.x {
        let abs = ctx.load_witness(F::from_u128(s.trim_start_matches('-').parse::<u128>().unwrap()));
        let v = if s.starts_with('-') { gate.neg(ctx, abs) } else { abs };
        make_public.push(v);
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(flatten_circuit, args); }
