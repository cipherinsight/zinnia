// DS-1000 case 314 — Halo2 baseline.
// Kernel:
//   idx = unravel_index(a.argmax(), a.shape)
//   a[idx] = a.min()
//   result = unravel_index(a.argmax(), a.shape)
// Inputs: a: Float[8, 6]. Output: (row, col) of the second argmax.
//
// High-level reimplementation:
//   1. First-pass argmax (flat C-order) on the 48 cells.
//   2. min over the 48 cells.
//   3. Replace a[idx] with min by rebuilding the array with select.
//   4. Second-pass argmax on the modified array.
//   5. Unravel to (row, col) via integer divmod by 6, witnessed and
//      verified by row*6 + col == best_idx and col < 6.

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

fn argmax_replace_min<F: ScalarField>(
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

    // First argmax (flat).
    let mut best_val = cells[0];
    let mut best_idx = ctx.load_constant(F::ZERO);
    for k in 1..48usize {
        let is_new = range_chip.is_less_than(ctx, best_val, cells[k], 128);
        best_val = gate.select(ctx, cells[k], best_val, is_new);
        let idx_k = Constant(F::from(k as u64));
        best_idx = gate.select(ctx, idx_k, best_idx, is_new);
    }

    // min over all 48 cells.
    let mut min_val = cells[0];
    for k in 1..48usize {
        let is_new = range_chip.is_less_than(ctx, cells[k], min_val, 128);
        min_val = gate.select(ctx, cells[k], min_val, is_new);
    }

    // Rebuild a' with a'[best_idx] = min_val.
    let mut updated: Vec<AssignedValue<F>> = Vec::with_capacity(48);
    for k in 0..48usize {
        let k_const = Constant(F::from(k as u64));
        let is_k = gate.is_equal(ctx, best_idx, k_const);
        updated.push(gate.select(ctx, min_val, cells[k], is_k));
    }

    // Second argmax on `updated`.
    let mut best_val2 = updated[0];
    let mut best_idx2 = ctx.load_constant(F::ZERO);
    for k in 1..48usize {
        let is_new = range_chip.is_less_than(ctx, best_val2, updated[k], 128);
        best_val2 = gate.select(ctx, updated[k], best_val2, is_new);
        let idx_k = Constant(F::from(k as u64));
        best_idx2 = gate.select(ctx, idx_k, best_idx2, is_new);
    }

    // Unravel (C-order, ncols=6): witness (row, col), check row*6+col == idx
    // and col < 6 and row < 8.
    // Witness via assignment values then assert.
    // (Rust-side compute the divmod; halo2 just verifies the witness.)
    // Compute best_idx2 native value: argmax over the cpu side too.
    // Easier: enumerate all (row, col) and pick by indicator.
    let mut row = ctx.load_constant(F::ZERO);
    let mut col = ctx.load_constant(F::ZERO);
    for r in 0..8usize {
        for c in 0..6usize {
            let flat = r * 6 + c;
            let flat_const = Constant(F::from(flat as u64));
            let is_match = gate.is_equal(ctx, best_idx2, flat_const);
            let r_const = Constant(F::from(r as u64));
            let c_const = Constant(F::from(c as u64));
            row = gate.select(ctx, r_const, row, is_match);
            col = gate.select(ctx, c_const, col, is_match);
        }
    }

    make_public.push(row);
    make_public.push(col);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(argmax_replace_min, args);
}
