// DS-1000 case 505 — Halo2 baseline.
// Kernel:
//     arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//     result = np.sum(a) - np.sum(arr)
// a: Float[13, 13, 5], b: Integer[13, 13]. Output: scalar.
// Compute sum(a) over all 845 cells, sum(arr) via 5-way gather, return diff.

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
    pub a: Vec<f64>, // 13*13*5 = 845 floats
    pub b: Vec<u64>, // 13*13 = 169 indices
}

fn take_along_axis_complement<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 13 * 13 * 5);
    assert_eq!(input.b.len(), 13 * 13);

    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();
    let b: Vec<AssignedValue<F>> = input.b.iter()
        .map(|v| ctx.load_witness(F::from(*v)))
        .collect();
    // Soundness: assert each idx ∈ {0..4}. See 0503 for rationale.
    for &idx in &b {
        rc.check_less_than_safe(ctx, idx, 5);
    }

    // sum_a = sum over all 845 cells of a.
    let mut sum_a = a[0];
    for k in 1..(13 * 13 * 5) {
        sum_a = fpc.qadd(ctx, sum_a, a[k]);
    }

    // sum_arr = sum_{i,j} a[i][j][b[i][j]] via 5-way gather.
    let mut sum_arr = ctx.load_witness(fpc.quantization(0.0));
    for i in 0..13usize {
        for j in 0..13usize {
            let idx = b[i * 13 + j];
            let mut out = a[(i * 13 + j) * 5 + 4];
            for k in (0..4usize).rev() {
                let k_const = Constant(F::from(k as u64));
                let is_k = gate.is_equal(ctx, idx, k_const);
                let candidate = a[(i * 13 + j) * 5 + k];
                out = gate.select(ctx, candidate, out, is_k);
            }
            sum_arr = fpc.qadd(ctx, sum_arr, out);
        }
    }

    let result = fpc.qsub(ctx, sum_a, sum_arr);
    make_public.push(result);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(take_along_axis_complement, args);
}
