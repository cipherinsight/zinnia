// DS-1000 case 364 — Halo2 baseline.
// Kernel: np.insert(a, pos, element, axis=0) on Float[100, 10] with element
// Float[10]. Output: Float[101, 10].
//
// Mirrors kernel.py's unrolled per-row insertion: for each output row i in
// 0..101 and column j in 0..10,
//   out[i, j] = a[i, j]          if i < pos
//             = element[j]        if i == pos
//             = a[i - 1, j]       if i > pos

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,        // 100 * 10 = 1000 floats row-major
    pub pos: String,        // signed decimal integer
    pub element: Vec<f64>,  // 10 floats
}

fn insert_row_2d<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 1000);
    assert_eq!(input.element.len(), 10);

    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let element: Vec<AssignedValue<F>> = input.element.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    let pos_abs = ctx.load_witness(F::from_u128(
        input.pos.trim_start_matches('-').parse::<u128>().unwrap()));
    let pos = if input.pos.starts_with('-') { gate.neg(ctx, pos_abs) } else { pos_abs };

    // Bounds-check semantic (mirrors Zinnia's `discharge_index_in_range`
    // for `np.insert`): pos must lie in [0, 101) for `np.insert(a, pos, ...)
    // on Float[100, 10] with axis=0`. Without this an out-of-range pos
    // silently falls through HW's per-row mux (no `is_pos` ever triggers,
    // so the output equals the un-shifted input — wrong semantic).
    range_chip.check_less_than_safe(ctx, pos, 101);

    for i in 0..101usize {
        let i_cell = ctx.load_constant(F::from(i as u64));
        let is_pre = range_chip.is_less_than(ctx, i_cell, pos, 128);
        let is_pos = gate.is_equal(ctx, i_cell, pos);
        for j in 0..10usize {
            let pre_val = if i < 100 { a[i * 10 + j] } else { a[99 * 10 + j] };
            let post_val = if i >= 1 { a[(i - 1) * 10 + j] } else { a[j] };
            let pos_or_post = gate.select(ctx, element[j], post_val, is_pos);
            let chosen = gate.select(ctx, pre_val, pos_or_post, is_pre);
            make_public.push(chosen);
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(insert_row_2d, args); }
