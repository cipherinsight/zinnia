// DS-1000 case 310 — Halo2 baseline.
// Kernel: `a.argmin()` on Float[8, 6].
// Mirrors Zinnia IR: 48× ReadFloatIR + 47× LessThanFIR + 46× SelectFIR
// + 46× SelectIIR + 46× ConstantIntIR + 1× ExposePublicIIR.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,  // 48 floats, row-major
}

fn argmin_8x6<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 48);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fixed_point_chip.quantization(*v)))
        .collect();

    let mut best_val = cells[0];
    let mut best_idx = ctx.load_constant(F::ZERO);
    for k in 1..48usize {
        // LessThanFIR(cells[k], best_val): cells[k] < best_val.
        let is_new_min = range_chip.is_less_than(ctx, cells[k], best_val, 128);
        best_val = gate.select(ctx, cells[k], best_val, is_new_min);
        let idx_k = Constant(F::from(k as u64));
        best_idx = gate.select(ctx, idx_k, best_idx, is_new_min);
    }
    make_public.push(best_idx);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(argmin_8x6, args);
}
