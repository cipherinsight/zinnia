// DS-1000 case 375 — Halo2 baseline.
// Kernel: xs = sort(grades); ys = arange(1, N+1)/N;
//         t = (ys > threshold).argmax();
//         low = xs[0]; high = xs[t]; return grades.
// low/high are computed but the returned value is `grades` unchanged.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub grades: Vec<f64>,   // 50 floats
    pub threshold: f64,
}

fn ecdf_bounds<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 50;
    let gate = GateChip::<F>::default();
    let rc = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.grades.len(), N);

    let grades: Vec<AssignedValue<F>> = input.grades.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();
    let threshold = ctx.load_witness(fpc.quantization(input.threshold));

    // Sort grades → xs (selection sort network).
    let mut xs = grades.clone();
    for i in 0..N {
        for j in (i + 1)..N {
            let lt = rc.is_less_than(ctx, xs[j], xs[i], 128);
            let new_i = gate.select(ctx, xs[j], xs[i], lt);
            let new_j = gate.select(ctx, xs[i], xs[j], lt);
            xs[i] = new_i;
            xs[j] = new_j;
        }
    }

    // ys[k] = (k+1)/N.
    let n_fp = ctx.load_witness(fpc.quantization(N as f64));
    let ys: Vec<AssignedValue<F>> = (0..N).map(|k| {
        let num = ctx.load_witness(fpc.quantization((k + 1) as f64));
        fpc.qdiv(ctx, num, n_fp)
    }).collect();

    // t = (ys > threshold).argmax() — first index k with ys[k] > threshold.
    let mut found = Constant(F::ZERO);
    let mut t = ctx.load_constant(F::ZERO);
    for k in 0..N {
        let hit = rc.is_less_than(ctx, threshold, ys[k], 128);
        let not_found = gate.not(ctx, found);
        let commit = gate.and(ctx, hit, not_found);
        let k_const = Constant(F::from(k as u64));
        t = gate.select(ctx, k_const, t, commit);
        let found_av = gate.or(ctx, found, hit);
        found = halo2_base::QuantumCell::Existing(found_av);
    }

    // low = xs[0]; high = xs[t] (gather).
    let low = xs[0];
    let mut high = xs[0];
    for k in 1..N {
        let is_k = gate.is_equal(ctx, t, Constant(F::from(k as u64)));
        high = gate.select(ctx, xs[k], high, is_k);
    }
    // low/high computed but discarded by the kernel.
    let _ = (low, high);

    // Expose grades unchanged.
    for k in 0..N {
        make_public.push(grades[k]);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(ecdf_bounds, args);
}
