// DS-1000 case 478 — Halo2 baseline.
// Kernel:
//     result = np.einsum('ii->i', a)   # diagonal of 4×4
//     save = result.copy()
//     a[...] = 0
//     result[...] = save                # no-op restore
//     return result
// Final output is the diagonal [a[0,0], a[1,1], a[2,2], a[3,3]].

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<String> }  // 16 decimal ints, row-major (4×4)

fn diag_einsum<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 16);
    let load = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int"),
        ));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };
    let mut a: Vec<AssignedValue<F>> = input.a.iter().map(|s| load(ctx, s)).collect();

    // result = diagonal of a (4 cells).
    let mut result: Vec<AssignedValue<F>> = (0..4).map(|i| a[i * 4 + i]).collect();
    // save = result.copy()
    let save: Vec<AssignedValue<F>> = result.clone();
    // a[...] = 0  (rewrite each cell of a with the constant 0).
    let zero = ctx.load_constant(F::ZERO);
    for i in 0..16 { a[i] = zero; }
    // result[...] = save  (restore — no-op).
    for i in 0..4 { result[i] = save[i]; }

    // Silence unused-variable warnings: reference `a` to keep its rewrite live.
    let _ = a;

    for k in 0..4 { make_public.push(result[k]); }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(diag_einsum, args); }
