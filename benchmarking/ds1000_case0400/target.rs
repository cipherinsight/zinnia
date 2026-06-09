// DS-1000 case 400 — Halo2 baseline. `np.array([])` — empty result.
// Zinnia emits 0 IR stmts; the minimal-eligibility predicate accepts this
// (no advice rows, no instance rows). Trivial circuit.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { }

fn empty_circuit<F: ScalarField>(
    _b: &mut BaseCircuitBuilder<F>, _input: CircuitInput, _make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    // No reads, no constants, no exposes.
}
fn main() { env_logger::init(); let args = Cli::parse(); run(empty_circuit, args); }
