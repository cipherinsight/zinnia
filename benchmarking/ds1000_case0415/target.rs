// DS-1000 case 415 — Halo2 baseline.
// Kernel: data[:(N//bs)*bs].reshape(-1, bs).max(axis=1) on Float[8], bs=2 → (4,).
// High-level: for each bin, running max via is_less_than + select.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub data: Vec<f64>, pub bin_size: u32 }

fn bin_max_1d<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.data.len(), 8);
    let bs = input.bin_size as usize;
    let n_bins = input.data.len() / bs;

    let cells: Vec<AssignedValue<F>> = input.data.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    for bin in 0..n_bins {
        let mut best = cells[bin * bs];
        for j in 1..bs {
            let cand = cells[bin * bs + j];
            // cand > best ⇔ best < cand
            let is_new = rc.is_less_than(ctx, best, cand, 128);
            best = gate.select(ctx, cand, best, is_new);
        }
        make_public.push(best);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(bin_max_1d, args); }
