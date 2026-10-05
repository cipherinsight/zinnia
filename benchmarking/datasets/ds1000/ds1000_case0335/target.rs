// DS-1000 case 335 — Halo2 baseline.
// Kernel: `np.max([a, b, c], axis=0)` on three Float[50] arrays.
// Mirrors Zinnia IR: 150× ReadFloatIR + 100× GreaterThanFIR + 100× SelectFIR + 50× ExposePublicFIR.
// Per-cell pattern: max(max(a, b), c) via 2 GreaterThan + 2 Select.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
}

fn max_of_three<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 50);
    assert_eq!(input.b.len(), 50);
    assert_eq!(input.c.len(), 50);

    for k in 0..50usize {
        let a_k = ctx.load_witness(fixed_point_chip.quantization(input.a[k]));
        let b_k = ctx.load_witness(fixed_point_chip.quantization(input.b[k]));
        let c_k = ctx.load_witness(fixed_point_chip.quantization(input.c[k]));

        // max(a, b)
        let b_gt_a = range_chip.is_less_than(ctx, a_k, b_k, 128);
        let ab_max = gate.select(ctx, b_k, a_k, b_gt_a);
        // max(_, c)
        let c_gt_ab = range_chip.is_less_than(ctx, ab_max, c_k, 128);
        let final_max = gate.select(ctx, c_k, ab_max, c_gt_ab);

        make_public.push(final_max);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(max_of_three, args);
}
