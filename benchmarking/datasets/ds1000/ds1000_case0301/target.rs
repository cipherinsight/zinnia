// DS-1000 case 301 — Halo2 baseline.
// Kernel: `np.reshape(A, (-1, ncol))` on Float[20] with runtime `ncol`.
// Layout-only — the 20 cell values are unchanged; only the shape changes.
// Mirrors Zinnia IR (117 stmts): 20× ReadF + ReadI(ncol) + DivI (rows)
// + ModI + asserts (divisibility, non-zero) + 20× ExposeF + 4× ExposeII (shape).
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

fn reshape_neg1_ncol<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 20);
    // 20 float cells (data is layout-only).
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    // ncol as int. Compute rows = 20 / ncol; assert 20 % ncol == 0 and ncol != 0.
    let ncol_abs = ctx.load_witness(F::from_u128(input.ncol.trim_start_matches('-').parse::<u128>().unwrap()));
    let ncol = if input.ncol.starts_with('-') { gate.neg(ctx, ncol_abs) } else { ncol_abs };
    let twenty = ctx.load_constant(F::from(20u64));
    let (rows, rem) = rc.div_mod_var(ctx, twenty, ncol, 128, 128);
    let zero = ctx.load_constant(F::ZERO);
    let rem_zero = gate.is_equal(ctx, rem, zero);
    gate.assert_is_const(ctx, &rem_zero, &F::ONE);
    let nc_zero = gate.is_equal(ctx, ncol, zero);
    let nc_nonzero = gate.not(ctx, nc_zero);
    gate.assert_is_const(ctx, &nc_nonzero, &F::ONE);
    // Expose 20 floats then shape metadata (rows, ncol, total=20, rank=2).
    for c in &cells { make_public.push(*c); }
    make_public.push(rows);
    make_public.push(ncol);
    let total = ctx.load_constant(F::from(20u64));
    let rank = ctx.load_constant(F::from(2u64));
    make_public.push(total);
    make_public.push(rank);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(reshape_neg1_ncol, args); }
