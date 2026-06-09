// DS-1000 case 345 — Halo2 baseline.
// Kernel: a[:, col-1] *= multiply_number; result = cumsum(a[:, col-1])
// On Float[8, 6] with runtime col, multiply_number. Output: 8 floats.
// Approach: select a[i][col-1] via 6-way select chain per row, scale, cumsum.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,
    pub col: String,
    pub multiply_number: String,
}

fn col_scale_cumsum<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 48);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let col_abs = ctx.load_witness(F::from_u128(input.col.trim_start_matches('-').parse::<u128>().unwrap()));
    let col = if input.col.starts_with('-') { gate.neg(ctx, col_abs) } else { col_abs };
    let one = ctx.load_constant(F::ONE);
    let col_idx = gate.sub(ctx, col, one);  // col - 1
    let mn_abs = ctx.load_witness(F::from_u128(input.multiply_number.trim_start_matches('-').parse::<u128>().unwrap()));
    let mn = if input.multiply_number.starts_with('-') { gate.neg(ctx, mn_abs) } else { mn_abs };
    // multiply_number lifted to fp.
    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let mn_fp = fpc.qmul(ctx, mn, one_fp);
    // Per row: select a[i][col_idx] via match on j=0..6.
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    let mut row_vals: Vec<AssignedValue<F>> = Vec::with_capacity(8);
    for i in 0..8usize {
        let mut chosen = zero_fp;
        for j in 0..6usize {
            let j_cell = ctx.load_constant(F::from(j as u64));
            let is_j = gate.is_equal(ctx, col_idx, j_cell);
            chosen = gate.select(ctx, cells[i * 6 + j], chosen, is_j);
        }
        let scaled = fpc.qmul(ctx, chosen, mn_fp);
        row_vals.push(scaled);
    }
    // Cumsum of 8 cells.
    let mut acc = row_vals[0];
    make_public.push(acc);
    for k in 1..8usize {
        acc = fpc.qadd(ctx, acc, row_vals[k]);
        make_public.push(acc);
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(col_scale_cumsum, args); }
