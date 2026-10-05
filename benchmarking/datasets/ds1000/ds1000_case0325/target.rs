// DS-1000 case 325 — Halo2 baseline.
// Kernel: `int(sin(number) > sin(deg2rad(number)))`. Mirrors Zinnia IR
// (9 stmts): ReadInt + FloatCast + ConstantF(π/180) + MulF + 2× SinFIR
// + GreaterThanF + IntCast + ExposePublicII.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::RangeInstructions;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub number: String }

fn sin_compare<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    let number_native: f64 = input.number.parse::<i64>().expect("integer number") as f64;
    let n_fp = ctx.load_witness(fpc.quantization(number_native));
    let pi_over_180 = ctx.load_witness(fpc.quantization(std::f64::consts::PI / 180.0));
    let rad = fpc.qmul(ctx, n_fp, pi_over_180);
    let sin_deg = fpc.qsin(ctx, rad);   // sin(deg2rad(n))
    let sin_n = fpc.qsin(ctx, n_fp);     // sin(n)
    // result = (sin_n > sin_deg) as int (0 or 1).
    let bit = rc.is_less_than(ctx, sin_deg, sin_n, 128);
    make_public.push(bit);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(sin_compare, args); }
