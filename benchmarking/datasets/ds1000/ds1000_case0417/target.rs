// DS-1000 case 417 — Halo2 baseline.
// Kernel: data[::-1] then bin-mean with bs=2 on Float[8] → (4,).
// Reverse is a compile-time index permutation; only the bin-mean is computed.

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

fn rev_bin_mean_1d<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 8;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.data.len(), N);
    let bs = input.bin_size as usize;
    let n_bins = N / bs;

    let cells: Vec<AssignedValue<F>> = input.data.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    // new_data[k] = data[N-1-k]
    let rev: Vec<AssignedValue<F>> = (0..N).map(|k| cells[N - 1 - k]).collect();
    let bs_fp = ctx.load_witness(fpc.quantization(bs as f64));

    for bin in 0..n_bins {
        let mut sum = rev[bin * bs];
        for j in 1..bs {
            sum = fpc.qadd(ctx, sum, rev[bin * bs + j]);
        }
        let mean = fpc.qdiv(ctx, sum, bs_fp);
        make_public.push(mean);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(rev_bin_mean_1d, args); }
