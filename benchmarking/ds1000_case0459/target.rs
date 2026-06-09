// DS-1000 case 459 — Halo2 baseline.
// Kernel: `np.mean(NA.astype(float), axis=0)` returning a single mean of NA.
// (A is declared but unused.) Mirrors Zinnia IR: 5× ReadFloatIR (A, unused)
// + 5× ReadIntegerIR (NA) + 5× FloatCastIR + 4× AddFIR + 1× DivFIR
// + 1× ExposePublicFIR.

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
    pub a: Vec<f64>,    // 5 floats (unused — mirrors Zinnia witnessing them)
    pub na: Vec<String>,  // 5 decimal integers
}

fn mean_cast<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 5);
    assert_eq!(input.na.len(), 5);
    // `a` is unused by the kernel; Zinnia v7 DCE-eliminates the dead
    // ReadFloatIR ops, so HW skips the loads too.
    // Cast NA to fp.
    let na_fp: Vec<AssignedValue<F>> = input.na.iter().map(|s| {
        let v: i64 = s.parse().expect("decimal int");
        ctx.load_witness(fixed_point_chip.quantization(v as f64))
    }).collect();
    let mut sum = na_fp[0];
    for k in 1..5 { sum = fixed_point_chip.qadd(ctx, sum, na_fp[k]); }
    let n = ctx.load_witness(fixed_point_chip.quantization(5.0));
    let mean = fixed_point_chip.qdiv(ctx, sum, n);
    make_public.push(mean);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(mean_cast, args); }
