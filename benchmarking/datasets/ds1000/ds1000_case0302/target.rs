// DS-1000 case 302 — Halo2 baseline.
// Kernel: `np.reshape(A, (nrow, -1))` on Float[20] with runtime `nrow`.
// Same shape and constraints as 0301 (just labels nrow / -1 swapped).
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
pub struct CircuitInput { pub a: Vec<f64>, pub nrow: String }

fn reshape_nrow_neg1<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 20);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let nrow_abs = ctx.load_witness(F::from_u128(input.nrow.trim_start_matches('-').parse::<u128>().unwrap()));
    let nrow = if input.nrow.starts_with('-') { gate.neg(ctx, nrow_abs) } else { nrow_abs };
    let twenty = ctx.load_constant(F::from(20u64));
    let (ncol, rem) = rc.div_mod_var(ctx, twenty, nrow, 128, 128);
    let zero = ctx.load_constant(F::ZERO);
    let rem_zero = gate.is_equal(ctx, rem, zero);
    gate.assert_is_const(ctx, &rem_zero, &F::ONE);
    let nr_zero = gate.is_equal(ctx, nrow, zero);
    let nr_nonzero = gate.not(ctx, nr_zero);
    gate.assert_is_const(ctx, &nr_nonzero, &F::ONE);
    for c in &cells { make_public.push(*c); }
    make_public.push(nrow);
    make_public.push(ncol);
    let total = ctx.load_constant(F::from(20u64));
    let rank = ctx.load_constant(F::from(2u64));
    make_public.push(total);
    make_public.push(rank);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(reshape_nrow_neg1, args); }
