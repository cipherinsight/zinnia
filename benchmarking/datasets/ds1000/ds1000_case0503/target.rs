// DS-1000 case 503 — Halo2 baseline.
// Kernel: `np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]`
// a: Float[13, 13, 5], b: Integer[13, 13] (values in {0..4}).
// 5-way gather per cell: build masks `(b == k)` and form
// out[i][j] = sum_{k=0..4} (b[i][j] == k) * a[i][j][k].

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>, // 13*13*5 = 845 floats, row-major
    pub b: Vec<u64>, // 13*13 = 169 indices
}

fn take_along_axis_3d5<F: ScalarField>(
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
    // Soundness: assert each idx ∈ {0..4}. Without this, the select chain
    // below defaults to a[..4] when idx is out of range (e.g. idx=99),
    // which the verifier would still accept. check_less_than_safe emits a
    // range-check lookup proving idx < 5.
    for &idx in &b {
        rc.check_less_than_safe(ctx, idx, 5);
    }

    for i in 0..13usize {
        for j in 0..13usize {
            let idx = b[i * 13 + j];
            // Chain gate.select: start with a[i][j][4], walk back peeling off equalities.
            // out = (idx == 0) ? a[..0] : ((idx == 1) ? a[..1] : ... a[..4]).
            let mut out = a[(i * 13 + j) * 5 + 4];
            for k in (0..4usize).rev() {
                let k_const = Constant(F::from(k as u64));
                let is_k = gate.is_equal(ctx, idx, k_const);
                let candidate = a[(i * 13 + j) * 5 + k];
                out = gate.select(ctx, candidate, out, is_k);
            }
            make_public.push(out);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(take_along_axis_3d5, args);
}
