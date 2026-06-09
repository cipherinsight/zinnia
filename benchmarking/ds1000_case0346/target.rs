// DS-1000 case 346 — Halo2 baseline.
// Kernel: a[row-1, :] *= multiply_number; result = cumsum(a[row-1, :])
// On Float[8, 6] with runtime row, multiply_number. Output: 6 floats.
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
    pub row: String,
    pub multiply_number: String,
}

fn row_scale_cumsum<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 48);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let row_abs = ctx.load_witness(F::from_u128(input.row.trim_start_matches('-').parse::<u128>().unwrap()));
    let row = if input.row.starts_with('-') { gate.neg(ctx, row_abs) } else { row_abs };
    let one = ctx.load_constant(F::ONE);
    let row_idx = gate.sub(ctx, row, one);
    let mn_abs = ctx.load_witness(F::from_u128(input.multiply_number.trim_start_matches('-').parse::<u128>().unwrap()));
    let mn = if input.multiply_number.starts_with('-') { gate.neg(ctx, mn_abs) } else { mn_abs };
    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let mn_fp = fpc.qmul(ctx, mn, one_fp);
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    // Per col j: select a[row_idx][j] via match on i=0..8.
    let mut col_vals: Vec<AssignedValue<F>> = Vec::with_capacity(6);
    for j in 0..6usize {
        let mut chosen = zero_fp;
        for i in 0..8usize {
            let i_cell = ctx.load_constant(F::from(i as u64));
            let is_i = gate.is_equal(ctx, row_idx, i_cell);
            chosen = gate.select(ctx, cells[i * 6 + j], chosen, is_i);
        }
        let scaled = fpc.qmul(ctx, chosen, mn_fp);
        col_vals.push(scaled);
    }
    let mut acc = col_vals[0];
    make_public.push(acc);
    for k in 1..6usize {
        acc = fpc.qadd(ctx, acc, col_vals[k]);
        make_public.push(acc);
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(row_scale_cumsum, args); }
