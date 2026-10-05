// DS-1000 case 382 — Halo2 baseline.
// Kernel: np.argsort(a) on Float[100] — ascending argsort indices.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,  // 100 floats
}

fn argsort_asc<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 100;
    let gate = GateChip::<F>::default();
    let rc = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), N);

    let mut vals: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();
    let mut idxs: Vec<AssignedValue<F>> = (0..N)
        .map(|k| ctx.load_constant(F::from(k as u64)))
        .collect();

    // Selection-sort ascending on (val, idx) pairs.
    for i in 0..N {
        for j in (i + 1)..N {
            let lt = rc.is_less_than(ctx, vals[j], vals[i], 128);
            let nv_i = gate.select(ctx, vals[j], vals[i], lt);
            let nv_j = gate.select(ctx, vals[i], vals[j], lt);
            let ni_i = gate.select(ctx, idxs[j], idxs[i], lt);
            let ni_j = gate.select(ctx, idxs[i], idxs[j], lt);
            vals[i] = nv_i;
            vals[j] = nv_j;
            idxs[i] = ni_i;
            idxs[j] = ni_j;
        }
    }

    for k in 0..N {
        make_public.push(idxs[k]);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(argsort_asc, args);
}
