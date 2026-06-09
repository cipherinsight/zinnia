// DS-1000 case 368 — Halo2 baseline.
// Kernel: `np.isclose(a, a[0], atol=0).all()` on Integer[3, 3].
// Mirrors Zinnia IR (65 stmts): 9× ReadInt + 9× SubI + 12× AbsI + 12× FloatCast
// + 3× MulF + 9× LessThanOrEqualF + 8× LogicalAnd + 1× ExposePublicII
// + (1× ConstantFloatIR for rtol = 1e-5).
//
// np.isclose semantics with atol=0, rtol=1e-5: |a - b| <= rtol * |b|.
// Then .all() reduces 9 booleans to 1 via AND chain.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,  // 9 decimal integers, row-major
}

fn isclose_all<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 9);
    // Load 9 integer cells.
    let cells: Vec<AssignedValue<F>> = input.a.iter().map(|s| {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int"),
        ));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    }).collect();

    let rtol = ctx.load_witness(fixed_point_chip.quantization(1e-5));
    let zero_fp = ctx.load_constant(F::ZERO);
    let one_fp = ctx.load_witness(fixed_point_chip.quantization(1.0));

    // |a[0][j]| as a float, then rtol * |a[0][j]| — one per column j ∈ {0,1,2}.
    let mut rtol_abs_b: Vec<AssignedValue<F>> = Vec::with_capacity(3);
    for j in 0..3 {
        // FloatCast of a[0][j] via qmul with 1.0 (lifts int→fp)
        let b_fp = fixed_point_chip.qmul(ctx, cells[j], one_fp);
        let b_lt_zero = range_chip.is_less_than(ctx, b_fp, zero_fp, 128);
        let neg_b = gate.neg(ctx, b_fp);
        let abs_b = gate.select(ctx, neg_b, b_fp, b_lt_zero);
        rtol_abs_b.push(fixed_point_chip.qmul(ctx, abs_b, rtol));
    }

    // For each cell (i, j): diff = a[i][j] - a[0][j]; |diff| as fp; check <= rtol*|b|.
    let mut all_close: Option<AssignedValue<F>> = None;
    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;
            let diff_int = gate.sub(ctx, cells[idx], cells[j]);
            let diff_fp = fixed_point_chip.qmul(ctx, diff_int, one_fp);
            let d_lt_zero = range_chip.is_less_than(ctx, diff_fp, zero_fp, 128);
            let neg_d = gate.neg(ctx, diff_fp);
            let abs_d = gate.select(ctx, neg_d, diff_fp, d_lt_zero);

            // |diff| <= rtol*|b|: equivalent to !(rtol*|b| < |diff|).
            let strictly_gt = range_chip.is_less_than(ctx, rtol_abs_b[j], abs_d, 128);
            let leq = gate.not(ctx, strictly_gt);

            all_close = Some(match all_close {
                None => leq,
                Some(prev) => gate.and(ctx, prev, leq),
            });
        }
    }
    make_public.push(all_close.unwrap());
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(isclose_all, args);
}
