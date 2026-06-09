// DS-1000 case 433 — Halo2 baseline.
// Kernel: `a[zero_rows, :] = 0; a[:, zero_cols] = 0` on Integer[4, 4] with
// zero_rows, zero_cols: scalar Integer (runtime). Mirrors Zinnia IR (98 stmts).
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<String>, pub zero_rows: String, pub zero_cols: String }

fn scalar_zero_idx<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    let load = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(s.trim_start_matches('-').parse::<u128>().unwrap()));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };
    let cells: Vec<AssignedValue<F>> = input.a.iter().map(|s| load(ctx, s)).collect();
    let zr = load(ctx, &input.zero_rows);
    let zc = load(ctx, &input.zero_cols);
    let zero = ctx.load_constant(F::ZERO);
    for i in 0..4usize {
        let i_cell = ctx.load_constant(F::from(i as u64));
        let row_match = gate.is_equal(ctx, i_cell, zr);
        for j in 0..4usize {
            let j_cell = ctx.load_constant(F::from(j as u64));
            let col_match = gate.is_equal(ctx, j_cell, zc);
            let any = gate.or(ctx, row_match, col_match);
            let out = gate.select(ctx, zero, cells[i * 4 + j], any);
            make_public.push(out);
        }
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(scalar_zero_idx, args); }
