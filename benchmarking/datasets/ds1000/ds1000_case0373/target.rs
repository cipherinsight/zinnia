// DS-1000 case 373 — Halo2 baseline.
// Kernel: ecdf_result(grades) with len = 50. xs = sort(grades);
// ys = arange(1, 51) / 50.0; return ys.
//
// xs values are computed (sort network) but unused — the output ys depends
// only on len, which is constant. We still emit the sort to mirror the
// kernel.py algorithm faithfully.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub grades: Vec<f64>,  // 50 floats
}

fn ecdf_y<F: ScalarField>(
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

    // Load grades.
    let cells: Vec<AssignedValue<F>> = input.grades.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    // Sort via selection-sort network: in-place running-min selection at
    // each output position. Same circuit semantics as the SP1 insertion
    // sort — produces xs ascending.
    let mut xs = cells.clone();
    for i in 0..N {
        for j in (i + 1)..N {
            // If xs[j] < xs[i], swap.
            let lt = rc.is_less_than(ctx, xs[j], xs[i], 128);
            let new_i = gate.select(ctx, xs[j], xs[i], lt);
            let new_j = gate.select(ctx, xs[i], xs[j], lt);
            xs[i] = new_i;
            xs[j] = new_j;
        }
    }
    let _ = &xs;  // Sort produced but ys depends only on length.

    // ys[i] = (i+1)/N for i in 0..N.
    let n_fp = ctx.load_witness(fpc.quantization(N as f64));
    for i in 0..N {
        let num_fp = ctx.load_witness(fpc.quantization((i + 1) as f64));
        let y = fpc.qdiv(ctx, num_fp, n_fp);
        make_public.push(y);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(ecdf_y, args);
}
