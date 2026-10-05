// DS-1000 case 291 — Halo2 baseline.
// Kernel: `a.shape` on Integer[2, 5, 5, 2].
// Exposes the literal Python result `(2, 5, 5, 2)`.
// `a` is unused by the kernel; Zinnia v7 DCE eliminates the dead
// ReadIntegerIR ops, so HW skips the loads too.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,  // 100 decimal integers (unused; see note)
}

fn shape_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 100);
    // `a` is unused by the kernel; Zinnia v7 DCE eliminates the dead
    // ReadIntegerIR ops, so HW skips the loads too.

    let two = ctx.load_witness(F::from(2u64));
    let five = ctx.load_witness(F::from(5u64));
    make_public.push(two);
    make_public.push(five);
    make_public.push(five);
    make_public.push(two);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(shape_circuit, args);
}
