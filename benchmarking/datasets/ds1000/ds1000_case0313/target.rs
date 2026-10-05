// DS-1000 case 313 — Halo2 baseline.
// Kernel: a.argmax() on Float[8, 6] → flat C-order index in [0, 48).
//
// Standard running-argmax pattern: scan all 48 cells, keep best_val and
// best_idx by select on greater-than.

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

fn argmax_flat<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 48);

    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    let mut best_val = cells[0];
    let mut best_idx = ctx.load_constant(F::ZERO);
    for k in 1..48usize {
        let is_new_max = range_chip.is_less_than(ctx, best_val, cells[k], 128);
        best_val = gate.select(ctx, cells[k], best_val, is_new_max);
        let idx_k = Constant(F::from(k as u64));
        best_idx = gate.select(ctx, idx_k, best_idx, is_new_max);
    }

    make_public.push(best_idx);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(argmax_flat, args);
}
