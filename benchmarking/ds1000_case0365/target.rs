// DS-1000 case 365 — Halo2 baseline (same kernel as 363).
// Kernel: np.insert(a, pos, element) on Float[100] with Integer pos & element.
// Output: Float[101].
//
// Mirrors kernel.py's unrolled insertion: for each output index i in 0..101,
//   out[i] = a[i]          if i < pos
//          = element        if i == pos
//          = a[i - 1]       if i > pos
// We `gate.is_equal`/`is_less_than` against an assigned pos and select.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,        // 100 floats
    pub pos: String,        // signed decimal integer
    pub element: String,    // signed decimal integer
}

fn insert_1d<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 100);
    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    // Load pos (integer) and element (integer cast to fp).
    let pos_abs = ctx.load_witness(F::from_u128(
        input.pos.trim_start_matches('-').parse::<u128>().unwrap()));
    let pos = if input.pos.starts_with('-') { gate.neg(ctx, pos_abs) } else { pos_abs };
    // Full numpy semantics: negative positions wrap (np.insert with
    // pos = -1 inserts before the last slot). In-circuit sign test on
    // the signed-64 window, then all comparisons use the normalized
    // (nonnegative) position; the final bound check enforces
    // -N <= pos <= N (numpy IndexError otherwise).
    let shift_c = ctx.load_constant(F::from(1u64 << 63));
    let shifted = gate.add(ctx, pos, shift_c);
    range_chip.range_check(ctx, shifted, 64);
    let is_neg = range_chip.is_less_than(ctx, shifted, shift_c, 64);
    let n_c = ctx.load_constant(F::from(100u64));
    let wrapped = gate.add(ctx, pos, n_c);
    let pos_n = gate.select(ctx, wrapped, pos, is_neg);
    let n1_c = ctx.load_constant(F::from(101u64));
    range_chip.check_less_than(ctx, pos_n, n1_c, 64);


    let elem_abs = ctx.load_witness(F::from_u128(
        input.element.trim_start_matches('-').parse::<u128>().unwrap()));
    let elem_int = if input.element.starts_with('-') { gate.neg(ctx, elem_abs) } else { elem_abs };
    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let elem_fp = fpc.qmul(ctx, elem_int, one_fp);

    // For each output index i in 0..101.
    for i in 0..101usize {
        let i_cell = ctx.load_constant(F::from(i as u64));
        // pre = (i < pos): means i is before insertion → use a[i].
        // is_pos = (i == pos): use element.
        // else: use a[i - 1] (only valid when i >= 1).
        let is_pre = range_chip.is_less_than(ctx, i_cell, pos_n, 128);
        let is_pos = gate.is_equal(ctx, i_cell, pos_n);

        // Source choices: pre = a[i] (only when i < 100), post = a[i-1] (only when i >= 1).
        // If i is 0 we never take "post" (cannot have i > pos with i=0 unless pos<0 — out of domain).
        // If i is 100 we never take "pre" (would need i<pos with pos>100 — out of domain).
        let pre_val = if i < 100 { a[i] } else { a[99] };  // dummy when unreachable
        let post_val = if i >= 1 { a[i - 1] } else { a[0] };  // dummy when unreachable

        // Layered select: chosen = is_pre ? pre_val : (is_pos ? element : post_val).
        let pos_or_post = gate.select(ctx, elem_fp, post_val, is_pos);
        let chosen = gate.select(ctx, pre_val, pos_or_post, is_pre);
        make_public.push(chosen);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(insert_1d, args); }
