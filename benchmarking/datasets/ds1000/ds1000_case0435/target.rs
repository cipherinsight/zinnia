// DS-1000 case 435 — Halo2 baseline. `a[1, :] = 0; a[:, 0] = 0` on Integer[4, 4].
// Zinnia uses zkram for the in-place writes (24× WriteMemoryIR). The handwritten
// halo2 baseline can compute the result directly without zkram since the
// indices are static.
//
// Output: 16 cells where row 1 and col 0 are zero; others pass through.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<String> }

fn static_zero_rows_cols<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 16);
    let cells: Vec<AssignedValue<F>> = input.a.iter().map(|s| {
        let abs = ctx.load_witness(F::from_u128(s.trim_start_matches('-').parse::<u128>().unwrap()));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    }).collect();
    let zero = ctx.load_constant(F::ZERO);
    for i in 0..4usize {
        for j in 0..4usize {
            // Row 1 → 0; col 0 → 0; else pass through.
            let cell = if i == 1 || j == 0 { zero } else { cells[i * 4 + j] };
            make_public.push(cell);
        }
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(static_zero_rows_cols, args); }
