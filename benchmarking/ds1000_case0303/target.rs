// DS-1000 case 303 — Halo2 baseline.
// Kernel: `B = A[:col].reshape(-1, ncol)` where col = 23 // ncol * ncol.
// Output is a (col/ncol, ncol) view of the first `col` cells of A.
// Mirrors Zinnia IR (257 stmts): 23× ReadFloat + ReadInt(ncol)
// + FloorDiv/Mod/Div for col + per-cell BoolCast/Select (keep vs zero)
// + 23× ExposePublicF + 4× ExposePublicI (shape metadata).
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
pub struct CircuitInput { pub a: Vec<f64>, pub ncol: String }

fn reshape_trim<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 23);
    let cells: Vec<AssignedValue<F>> = input.a.iter().map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let ncol_abs = ctx.load_witness(F::from_u128(input.ncol.trim_start_matches('-').parse::<u128>().unwrap()));
    let ncol = if input.ncol.starts_with('-') { gate.neg(ctx, ncol_abs) } else { ncol_abs };
    let twenty_three = ctx.load_constant(F::from(23u64));
    let (rows, _rem) = rc.div_mod_var(ctx, twenty_three, ncol, 128, 128);
    let col_keep = gate.mul(ctx, rows, ncol);  // = 23 // ncol * ncol
    // For each i in 0..23: keep if i < col_keep, else 0.
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    for (i, c) in cells.iter().enumerate() {
        let i_cell = ctx.load_constant(F::from(i as u64));
        let in_range = rc.is_less_than(ctx, i_cell, col_keep, 128);
        let out = gate.select(ctx, *c, zero_fp, in_range);
        make_public.push(out);
    }
    // Shape metadata: (rows, ncol, total=col_keep, rank=2).
    make_public.push(rows);
    make_public.push(ncol);
    make_public.push(col_keep);
    let rank = ctx.load_constant(F::from(2u64));
    make_public.push(rank);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(reshape_trim, args); }
