// DS-1000 case 414 — Halo2 baseline.
// Kernel: data[:(N//bs)*bs].reshape(-1, bs).mean(axis=1) on Float[8], bs=2 → (4,).
// High-level: for each bin, accumulate sum via qadd, then qdiv by bin_size.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub data: Vec<f64>, pub bin_size: u32 }

fn bin_mean_1d<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.data.len(), 8);
    let bs = input.bin_size as usize;
    let n_bins = input.data.len() / bs;

    let cells: Vec<AssignedValue<F>> = input.data.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let bs_fp = ctx.load_witness(fpc.quantization(bs as f64));

    for bin in 0..n_bins {
        let mut sum = cells[bin * bs];
        for j in 1..bs {
            sum = fpc.qadd(ctx, sum, cells[bin * bs + j]);
        }
        let mean = fpc.qdiv(ctx, sum, bs_fp);
        make_public.push(mean);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(bin_mean_1d, args); }
