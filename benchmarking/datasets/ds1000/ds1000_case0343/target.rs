// DS-1000 case 343 — Halo2 baseline. `X.T.flat` on Integer[5, 6].
// Emits cells in column-major order: X[0][0], X[1][0], ..., X[5][0], X[0][1], ...
// IR: 30× ReadInt + 30× ExposePublicII.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub x: Vec<String> }

fn flatten_t<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    assert_eq!(input.x.len(), 30);
    let cells: Vec<AssignedValue<F>> = input.x.iter().map(|s| {
        let abs = ctx.load_witness(F::from_u128(s.trim_start_matches('-').parse::<u128>().unwrap()));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    }).collect();
    // X is 5x6 (5 rows, 6 cols). X.T.flat iterates cols then rows.
    for j in 0..6usize {
        for i in 0..5usize {
            make_public.push(cells[i * 6 + j]);
        }
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(flatten_t, args); }
