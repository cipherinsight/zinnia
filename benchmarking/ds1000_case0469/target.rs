// DS-1000 case 469 — Halo2 baseline.
// Kernel: `np.real(s)/n + 1j*np.imag(s)/n` with s = np.sum(a) on Float[3].
// For real inputs np.imag(s) = 0; result = (mean, 0).
// Exposes (real, imag) as two public floats.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }  // 3 floats

fn complex_mean<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 3);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    // s_re = a[0] + a[1] + a[2]; s_im = 0.
    let s01 = fpc.qadd(ctx, cells[0], cells[1]);
    let s_re = fpc.qadd(ctx, s01, cells[2]);
    let s_im = ctx.load_witness(fpc.quantization(0.0));
    // n = 3.
    let n = ctx.load_witness(fpc.quantization(3.0));
    let res_re = fpc.qdiv(ctx, s_re, n);
    let res_im = fpc.qdiv(ctx, s_im, n);
    make_public.push(res_re);
    make_public.push(res_im);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(complex_mean, args); }
