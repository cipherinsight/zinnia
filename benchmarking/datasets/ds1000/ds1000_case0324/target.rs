// DS-1000 case 324 — Halo2 baseline. cos(deg2rad(degree)).
// Mirrors Zinnia IR: ReadInt + FloatCast + ConstantF(π/180) + MulF + CosFIR + ExposeF.
// Uses halo2-graph's qcos (single primitive) — matches Zinnia's single CosFIR.

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
pub struct CircuitInput { pub degree: String }

fn cos_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    let degree_native: f64 = input.degree.parse::<i64>().expect("integer degree") as f64;
    let degree_fp = ctx.load_witness(fixed_point_chip.quantization(degree_native));
    let pi_over_180 = ctx.load_witness(fixed_point_chip.quantization(std::f64::consts::PI / 180.0));
    let rad = fixed_point_chip.qmul(ctx, degree_fp, pi_over_180);
    let cos_val = fixed_point_chip.qcos(ctx, rad);
    make_public.push(cos_val);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(cos_circuit, args); }
