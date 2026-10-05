// DS-1000 case 344 — Halo2 baseline.
// Kernel: `np.array(list(mystr), dtype=int)` on Integer scalar → echoes the input.
// Mirrors Zinnia IR (3 stmts): ReadInt + AllocateMemory + ExposePublicII.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub mystr: String }

fn trivial<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    let abs = ctx.load_witness(F::from_u128(
        input.mystr.trim_start_matches('-').parse::<u128>().expect("int")));
    let v = if input.mystr.starts_with('-') { gate.neg(ctx, abs) } else { abs };
    make_public.push(v);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(trivial, args); }
