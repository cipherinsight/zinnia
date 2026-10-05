// DS-1000 case 369 — Halo2 baseline.
// Kernel: `np.isclose(a, a[:, 0].reshape(-1, 1), atol=0).all()` on Integer[3, 3].
// Compares each cell to its row's first column. Mirrors Zinnia IR
// (66 stmts): 9× ReadInt + 12× FloatCast + 12× AbsI + 3× MulF + 9× SubI
// + 9× LessThanOrEqualF + 8× LogicalAnd + 1× ExposePublicII.

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
pub struct CircuitInput { pub a: Vec<String> }  // 9 ints

fn isclose_col_broadcast<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 9);
    let cells: Vec<AssignedValue<F>> = input.a.iter().map(|s| {
        let abs = ctx.load_witness(F::from_u128(s.trim_start_matches('-').parse::<u128>().unwrap()));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    }).collect();
    let zero_fp = ctx.load_constant(F::ZERO);
    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let rtol = ctx.load_witness(fpc.quantization(1e-5));
    let abs_fp = |ctx: &mut halo2_base::Context<F>, x: &AssignedValue<F>| -> AssignedValue<F> {
        let x_fp = fpc.qmul(ctx, *x, one_fp);
        let lt = rc.is_less_than(ctx, x_fp, zero_fp, 128);
        let neg = gate.neg(ctx, x_fp);
        gate.select(ctx, neg, x_fp, lt)
    };
    let mut rtol_abs_col0: Vec<AssignedValue<F>> = Vec::with_capacity(3);
    for i in 0..3 {
        let abs_b = abs_fp(ctx, &cells[i * 3]);
        rtol_abs_col0.push(fpc.qmul(ctx, abs_b, rtol));
    }
    let mut acc: Option<AssignedValue<F>> = None;
    for i in 0..3 {
        for j in 0..3 {
            let diff_i = gate.sub(ctx, cells[i * 3 + j], cells[i * 3]);
            let diff_fp = fpc.qmul(ctx, diff_i, one_fp);
            let lt = rc.is_less_than(ctx, diff_fp, zero_fp, 128);
            let neg = gate.neg(ctx, diff_fp);
            let abs_d = gate.select(ctx, neg, diff_fp, lt);
            let strict_gt = rc.is_less_than(ctx, rtol_abs_col0[i], abs_d, 128);
            let leq = gate.not(ctx, strict_gt);
            acc = Some(match acc { None => leq, Some(p) => gate.and(ctx, p, leq) });
        }
    }
    make_public.push(acc.unwrap());
}
fn main() { env_logger::init(); let args = Cli::parse(); run(isclose_col_broadcast, args); }
