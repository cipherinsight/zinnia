// DS-1000 case 304 — Halo2 baseline.
// Kernel:
//   col = (A.shape[0] // ncol) * ncol
//   B = A[len(A) - col:][::-1]
//   B = np.reshape(B, (-1, ncol))
// Inputs: A: Float[23], ncol: Integer.
//
// Output B has runtime shape (rows, ncol) where rows = 23 // ncol and
// total = rows * ncol. We mirror Zinnia's dynamic-ndarray metadata
// assertion (synthesizer.rs::assert_dynamic_ndarray_meta): assert each
// runtime dim is in [0, axis_bound] and the product is ≤ total_bound.
//
// Static upper bound is 23 cells. We emit 23 padded outputs (zero past
// col_keep) so the output stream length matches Zinnia's flat layout.

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
    pub a: Vec<f64>,    // 23 floats
    pub ncol: String,   // integer in (0, 23]
}

fn reshape_trim_rev<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 23);

    // Load `a` (note: this is a flat 23-cell array).
    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    // ncol: signed integer (asserted > 0 below).
    let ncol_abs = ctx.load_witness(F::from_u128(
        input.ncol.trim_start_matches('-').parse::<u128>().expect("decimal"),
    ));
    let ncol = if input.ncol.starts_with('-') { gate.neg(ctx, ncol_abs) } else { ncol_abs };

    // Witness rows = 23 // ncol and col_keep = rows * ncol.
    let ncol_native: i64 = input.ncol.parse().expect("decimal");
    assert!(ncol_native > 0 && ncol_native <= 23);
    let rows_native: i64 = 23 / ncol_native;
    let col_keep_native: i64 = rows_native * ncol_native;
    let rows = ctx.load_witness(F::from(rows_native as u64));
    let col_keep = ctx.load_witness(F::from(col_keep_native as u64));

    // Verify rows * ncol == col_keep and (rows + 1) * ncol > 23.
    let rec_col_keep = gate.mul(ctx, rows, ncol);
    let ck_eq = gate.is_equal(ctx, rec_col_keep, col_keep);
    gate.assert_is_const(ctx, &ck_eq, &F::ONE);
    let one_int = ctx.load_constant(F::ONE);
    let rows_plus_1 = gate.add(ctx, rows, one_int);
    let next_block = gate.mul(ctx, rows_plus_1, ncol);
    let twenty_three = ctx.load_constant(F::from(23u64));
    // (rows+1)*ncol > 23  ↔  23 < next_block.
    let div_tight = range_chip.is_less_than(ctx, twenty_three, next_block, 128);
    gate.assert_is_const(ctx, &div_tight, &F::ONE);
    // col_keep ≤ 23.
    let ck_le_23 = {
        let lt = range_chip.is_less_than(ctx, twenty_three, col_keep, 128);
        gate.not(ctx, lt)
    };
    gate.assert_is_const(ctx, &ck_le_23, &F::ONE);

    // For each output slot k in 0..23, compute:
    //   in_range = (k < col_keep)
    //   out[k] = a[22 - k] if in_range else 0.0
    // Note: src index 22-k is a compile-time constant for each k.
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    for k in 0..23usize {
        let k_const = Constant(F::from(k as u64));
        let in_range = range_chip.is_less_than(ctx, k_const, col_keep, 128);
        let src_val = a[22 - k];
        let out_k = gate.select(ctx, src_val, zero_fp, in_range);
        make_public.push(out_k);
    }

    // Mirror DynamicNDArray metadata: rank=2, axes=(rows, ncol), product <= 23.
    // Assert non-negative (already non-negative by construction) and per-axis
    // bounds: rows ≤ 23, ncol ≤ 23. Product = col_keep ≤ 23 (checked above).
    let rows_in_axis = {
        let lt = range_chip.is_less_than(ctx, twenty_three, rows, 128);
        gate.not(ctx, lt)
    };
    gate.assert_is_const(ctx, &rows_in_axis, &F::ONE);
    let ncol_in_axis = {
        let lt = range_chip.is_less_than(ctx, twenty_three, ncol, 128);
        gate.not(ctx, lt)
    };
    gate.assert_is_const(ctx, &ncol_in_axis, &F::ONE);

    // Expose the shape metadata (rows, ncol, total, rank=2).
    make_public.push(rows);
    make_public.push(ncol);
    make_public.push(col_keep);
    let two = ctx.load_constant(F::from(2u64));
    make_public.push(two);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(reshape_trim_rev, args);
}
