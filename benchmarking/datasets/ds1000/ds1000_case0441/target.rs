// DS-1000 case 441 — Halo2 baseline.
// Kernel: `number in a` on Integer[4].
// Mirrors Zinnia IR: 5× ReadIntegerIR + 4× EqualIIR + 3× LogicalOrIR + 1× ExposePublicIIR.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::GateInstructions;
use halo2_base::gates::GateChip;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,    // 4 decimal integers
    pub number: String,    // 1 decimal integer
}

fn contains_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = builder.main(0);

    let load_int = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int"),
        ));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };

    assert_eq!(input.a.len(), 4);
    let cells: Vec<AssignedValue<F>> = input.a.iter().map(|s| load_int(ctx, s)).collect();
    let number = load_int(ctx, &input.number);

    let mut acc: Option<AssignedValue<F>> = None;
    for k in 0..4 {
        let eq = gate.is_equal(ctx, cells[k], number);
        acc = Some(match acc {
            None => eq,
            Some(prev) => gate.or(ctx, prev, eq),
        });
    }
    make_public.push(acc.unwrap());
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(contains_circuit, args);
}
