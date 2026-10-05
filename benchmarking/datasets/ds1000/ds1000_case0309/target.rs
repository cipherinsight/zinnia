// DS-1000 case 309 — Halo2 baseline (apples-to-apples with Zinnia).
//
// Mirrors Zinnia's IR exactly (235 stmts, op breakdown):
//   48 ReadFloatIR        → load_witness(quantization(x))
//   47 GreaterThanFIR     → range_chip.is_less_than(rhs, lhs, 128)
//   46 SelectFIR          → gate.select(true, false, cond)         (value)
//   46 SelectIIR          → gate.select(true, false, cond)         (index)
//   46 ConstantIntIR      → Constant(F::from(k))
//    1 ExposePublicIIR    → make_public.push(best_idx)
//
// No claim assertion — the computed argmax is exposed as public output
// directly, matching ExposePublicIIR's lowering.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{
    AssignedValue,
    QuantumCell::Constant,
};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,  // 48 floats, row-major
}

fn argmax_8x6<F: ScalarField>(
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
        // GreaterThanFIR (cells[k] > best_val) → is_less_than(best_val, cells[k]).
        let is_new_max = range_chip.is_less_than(ctx, best_val, cells[k], 128);
        // SelectFIR (running max value).
        best_val = gate.select(ctx, cells[k], best_val, is_new_max);
        // SelectIIR (running argmax index). ConstantIntIR for the candidate index.
        let idx_k = Constant(F::from(k as u64));
        best_idx = gate.select(ctx, idx_k, best_idx, is_new_max);
    }

    // ExposePublicIIR.
    make_public.push(best_idx);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(argmax_8x6, args);
}
