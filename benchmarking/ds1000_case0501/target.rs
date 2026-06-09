// DS-1000 case 501 — Halo2 baseline.
// Kernel: `np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]`
// a: Float[13, 13, 2], b: Integer[13, 13] (values in {0, 1}).
// Per-cell select between a[i,j,0] and a[i,j,1] based on b[i,j].

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>, // 13*13*2 = 338 floats, row-major
    pub b: Vec<u64>, // 13*13 = 169 indices
}

fn take_along_axis_3d2<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 13 * 13 * 2);
    assert_eq!(input.b.len(), 13 * 13);

    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();
    let b: Vec<AssignedValue<F>> = input.b.iter()
        .map(|v| ctx.load_witness(F::from(*v)))
        .collect();
    // Soundness: assert each idx ∈ {0, 1}. Without this constraint, a
    // malicious prover could pass idx=5 and `gate.select` would compute
    // 5*v1 + (1-5)*v0 = 5v1−4v0 — a different output the verifier would
    // still accept. `assert_bit` enforces idx*(idx-1)==0 in a single row.
    for &idx in &b {
        gate.assert_bit(ctx, idx);
    }

    for i in 0..13usize {
        for j in 0..13usize {
            let idx = b[i * 13 + j];
            let v0 = a[(i * 13 + j) * 2 + 0];
            let v1 = a[(i * 13 + j) * 2 + 1];
            let out = gate.select(ctx, v1, v0, idx);
            make_public.push(out);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(take_along_axis_3d2, args);
}
