// DS-1000 case 451 — Halo2 baseline. `np.zeros((20, 10, 10, 2))` — 4000 zeros.
// Zinnia compiles to just 3 stmts: 1× AllocateMemoryIR + 1× ConstantIntIR + 1× ExposePublicIIR.
// It represents the zero-constant output as a single public value rather than
// 4000 individual exposes. We mirror that: emit one expose of 0.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { }

fn zeros_const<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, _input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let ctx = b.main(0);
    let zero = ctx.load_witness(F::ZERO);
    make_public.push(zero);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(zeros_const, args); }
