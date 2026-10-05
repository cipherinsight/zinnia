// DS-1000 case 322 — Halo2 baseline.
// Kernel: result = np.argwhere(a == np.min(a)) on Integer[2, 2].
//
// High-level reimplementation:
//   1. min over the 4 integer cells (running select with is_less_than).
//   2. For each cell, indicator = (cell == min).
//   3. len = sum of indicators.
//   4. First-match (row, col): scan cells in row-major order; first index
//      where indicator==1 sets (row, col) once.
// Output expose: (first_row, first_col, length).
//
// Mirrors synthesizer.rs::assert_dynamic_ndarray_meta: length is a
// runtime value in [0, 4], rank=2, total ≤ 4.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,  // 4 integers, row-major 2x2
}

fn argwhere_min<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 4);

    // Load integer cells (signed).
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|s| {
            let abs = ctx.load_witness(F::from_u128(
                s.trim_start_matches('-').parse::<u128>().expect("decimal"),
            ));
            if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
        })
        .collect();

    // min via running select.
    let mut min_val = cells[0];
    for k in 1..4usize {
        let is_new = range_chip.is_less_than(ctx, cells[k], min_val, 128);
        min_val = gate.select(ctx, cells[k], min_val, is_new);
    }

    // indicators[k] = (cells[k] == min_val).
    let mut indicators: Vec<AssignedValue<F>> = Vec::with_capacity(4);
    for k in 0..4usize {
        indicators.push(gate.is_equal(ctx, cells[k], min_val));
    }

    // length = sum of indicators.
    let mut length = indicators[0];
    for k in 1..4usize {
        length = gate.add(ctx, length, indicators[k]);
    }

    // First match (row, col). Walk cells row-major; the first k with
    // indicators[k] == 1 contributes (k/2, k%2). We pick by "still
    // looking" flag = (1 - any prior match).
    let mut first_row = ctx.load_constant(F::ZERO);
    let mut first_col = ctx.load_constant(F::ZERO);
    let mut found = ctx.load_constant(F::ZERO); // 0 until first match
    for k in 0..4usize {
        let still_looking = gate.not(ctx, found);
        let take = gate.mul(ctx, still_looking, indicators[k]);
        let row_k = Constant(F::from((k / 2) as u64));
        let col_k = Constant(F::from((k % 2) as u64));
        first_row = gate.select(ctx, row_k, first_row, take);
        first_col = gate.select(ctx, col_k, first_col, take);
        // Update found: found = found || indicators[k] = found + ind - found*ind.
        let prod = gate.mul(ctx, found, indicators[k]);
        let sum = gate.add(ctx, found, indicators[k]);
        found = gate.sub(ctx, sum, prod);
    }

    // assert length >= 1 (at least one minimum exists, always true for non-empty).
    let one = ctx.load_constant(F::ONE);
    let length_ge_1 = {
        let lt = range_chip.is_less_than(ctx, length, one, 128);
        gate.not(ctx, lt)
    };
    gate.assert_is_const(ctx, &length_ge_1, &F::ONE);
    // assert length <= 4.
    let four = ctx.load_constant(F::from(4u64));
    let length_le_4 = {
        let lt = range_chip.is_less_than(ctx, four, length, 128);
        gate.not(ctx, lt)
    };
    gate.assert_is_const(ctx, &length_le_4, &F::ONE);

    // Expose (first_row, first_col, length).
    make_public.push(first_row);
    make_public.push(first_col);
    make_public.push(length);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(argwhere_min, args);
}
