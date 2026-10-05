// DS-1000 case 436 — Halo2 baseline.
// Kernel: `(a.max(axis=1, keepdims=1) == a)` on Float[10, 5] → Bool[10, 5].
// Mirrors Zinnia IR: 50× ReadFloatIR + 40× GreaterThanFIR + 40× SelectFIR
// + 50× EqualFIR + 50× ExposePublicIIR.
// Per row: row_max via 4 GT + 4 Select; then 5 element-vs-max equality bits.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }  // 50 cells

fn max_mask<F: ScalarField>(
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
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fixed_point_chip.quantization(*v)))
        .collect();
    let eps_pos = Constant(fixed_point_chip.quantization(0.001));
    let eps_neg = Constant(fixed_point_chip.quantization(-0.001));
    for i in 0..10usize {
        // running row max
        let mut row_max = cells[i * 5];
        for j in 1..5usize {
            let cur = cells[i * 5 + j];
            let cur_gt = range_chip.is_less_than(ctx, row_max, cur, 128);
            row_max = gate.select(ctx, cur, row_max, cur_gt);
        }
        // element-wise equality (float-equality pattern)
        for j in 0..5usize {
            let diff = fixed_point_chip.qsub(ctx, cells[i * 5 + j], row_max);
            let lt_p = range_chip.is_less_than(ctx, diff, eps_pos, 128);
            let gt_n = range_chip.is_less_than(ctx, eps_neg, diff, 128);
            let eq = gate.and(ctx, lt_p, gt_n);
            make_public.push(eq);
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(max_mask, args); }
