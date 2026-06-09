// DS-1000 case 434 — Halo2 baseline.
// Kernel: `a[zero_rows, :] = 0; a[:, zero_cols] = 0` on Integer[4, 4] with
// zero_rows, zero_cols: Integer[2] (runtime). Mirrors Zinnia IR (103 stmts):
// 16× ReadInt (a) + 4× ReadInt (zero_rows ∪ zero_cols) + per-cell zkram
// + 16× ExposePublicII. We compute directly without zkram by select-chaining
// row/col membership.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,           // 16 ints
    pub zero_rows: Vec<String>,   // 2 ints
    pub zero_cols: Vec<String>,   // 2 ints
}

fn array_indexed_zero<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    let load = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(s.trim_start_matches('-').parse::<u128>().unwrap()));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };
    assert_eq!(input.a.len(), 16);
    assert_eq!(input.zero_rows.len(), 2);
    assert_eq!(input.zero_cols.len(), 2);
    let cells: Vec<AssignedValue<F>> = input.a.iter().map(|s| load(ctx, s)).collect();
    let zrs: Vec<AssignedValue<F>> = input.zero_rows.iter().map(|s| load(ctx, s)).collect();
    let zcs: Vec<AssignedValue<F>> = input.zero_cols.iter().map(|s| load(ctx, s)).collect();
    let zero = ctx.load_constant(F::ZERO);
    for i in 0..4usize {
        let i_cell = ctx.load_constant(F::from(i as u64));
        let i_in_zr_0 = gate.is_equal(ctx, i_cell, zrs[0]);
        let i_in_zr_1 = gate.is_equal(ctx, i_cell, zrs[1]);
        let row_zero = gate.or(ctx, i_in_zr_0, i_in_zr_1);
        for j in 0..4usize {
            let j_cell = ctx.load_constant(F::from(j as u64));
            let j_in_zc_0 = gate.is_equal(ctx, j_cell, zcs[0]);
            let j_in_zc_1 = gate.is_equal(ctx, j_cell, zcs[1]);
            let col_zero = gate.or(ctx, j_in_zc_0, j_in_zc_1);
            let any_zero = gate.or(ctx, row_zero, col_zero);
            let out = gate.select(ctx, zero, cells[i * 4 + j], any_zero);
            make_public.push(out);
        }
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(array_indexed_zero, args); }
