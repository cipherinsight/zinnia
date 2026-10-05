// DS-1000 case 460 — Halo2 baseline.
// Kernel: `np.mean(NA.astype(float), axis=0)` returning a single mean of NA.
// A: Float[5] is declared but unused (witnessed to mirror Zinnia IR).
// Identical structure to case 459.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,      // 5 floats (unused — mirrors Zinnia witnessing them)
    pub na: Vec<String>,  // 5 decimal integers
}

fn mean_cast<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 5);
    assert_eq!(input.na.len(), 5);
    // `a` is unused by the kernel; Zinnia v7 DCE-eliminates the dead
    // ReadFloatIR ops, so HW skips the loads too.
    // Cast NA to fp.
    let na_fp: Vec<AssignedValue<F>> = input.na.iter().map(|s| {
        let v: i64 = s.parse().expect("decimal int");
        ctx.load_witness(fpc.quantization(v as f64))
    }).collect();
    let mut sum = na_fp[0];
    for k in 1..5 { sum = fpc.qadd(ctx, sum, na_fp[k]); }
    let n = ctx.load_witness(fpc.quantization(5.0));
    let mean = fpc.qdiv(ctx, sum, n);
    make_public.push(mean);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(mean_cast, args); }
