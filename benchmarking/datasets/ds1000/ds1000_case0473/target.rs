// DS-1000 case 473 — Halo2 baseline.
// Kernel: any(np.array_equal(c, x) for x in CNTS)
// c is Integer[4, 1, 2] (8 cells), CNTS is Integer[3] (3 scalars).
// np.array_equal between an 8-element ndarray and a scalar is always False
// (shape mismatch). Zinnia v7 evaluates this to a static False at trace
// time and DCE-eliminates the per-cell ReadIntegerIR + equality ops; HW
// mirrors that by emitting only the constant False output.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub c: Vec<String>, pub cnts: Vec<String> }

fn any_array_equal<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let ctx = b.main(0);
    assert_eq!(input.c.len(), 8);
    assert_eq!(input.cnts.len(), 3);
    // `c` and `cnts` are unused: np.array_equal(8-array, scalar) is False by
    // shape mismatch, so the per-cell equality work is dead. Zinnia v7 DCE
    // eliminates the ReadIntegerIR ops and the comparison chain, so HW skips
    // the loads too and emits the constant False result directly.
    let result = ctx.load_constant(F::ZERO);
    make_public.push(result);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(any_array_equal, args); }
