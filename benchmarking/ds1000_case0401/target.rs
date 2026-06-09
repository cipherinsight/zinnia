// DS-1000 case 401 — Halo2 baseline. `np.array([[], [], []])` — empty 3×0 result.
// Same as 0400: 0 IR stmts.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { }

fn empty_3x0<F: ScalarField>(
    _b: &mut BaseCircuitBuilder<F>, _input: CircuitInput, _make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField { }
fn main() { env_logger::init(); let args = Cli::parse(); run(empty_3x0, args); }
