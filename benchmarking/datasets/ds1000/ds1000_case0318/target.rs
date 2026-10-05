// DS-1000 case 318 — Halo2 baseline.
// Kernel:
//   c = empty_like(permutation)
//   c[permutation] = arange(len(permutation))   # inverse permutation
//   a = a[:, c]                                   # reorder columns by c
// Inputs: a: Float[8, 6], permutation: Integer[6].
//
// High-level reimplementation:
//   1. Build inverse c: for each k in 0..6, find j such that permutation[j] == k.
//      Equivalently c[k] = sum_j j * (permutation[j] == k).
//   2. For each output cell out[i][j] = a[i][c[j]]. Gather via
//      indicator-vector inner product over the j-th row.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,             // 48 floats, row-major (8 x 6)
    pub permutation: Vec<String>,// 6 integers, a permutation of 0..6
}

fn inv_perm_cols<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 48);
    assert_eq!(input.permutation.len(), 6);

    // Load a (row-major).
    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    // Load permutation as signed integers.
    let perm: Vec<AssignedValue<F>> = input.permutation.iter()
        .map(|s| {
            let abs = ctx.load_witness(F::from_u128(
                s.trim_start_matches('-').parse::<u128>().expect("decimal"),
            ));
            if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
        })
        .collect();

    // c[k] = sum_j j * (perm[j] == k).
    let mut c: Vec<AssignedValue<F>> = Vec::with_capacity(6);
    for k in 0..6usize {
        let k_const = Constant(F::from(k as u64));
        let mut sum = ctx.load_constant(F::ZERO);
        for j in 0..6usize {
            let is_eq = gate.is_equal(ctx, perm[j], k_const);
            let j_const = Constant(F::from(j as u64));
            let contrib = gate.mul(ctx, j_const, is_eq);
            sum = gate.add(ctx, sum, contrib);
        }
        c.push(sum);
    }

    // out[i][j] = a[i][c[j]]. Gather via indicator inner product on row i.
    for i in 0..8usize {
        for j in 0..6usize {
            // Indicator vector over the 6 columns of row i.
            let mut acc = ctx.load_constant(F::ZERO);
            for col in 0..6usize {
                let col_const = Constant(F::from(col as u64));
                let is_col = gate.is_equal(ctx, c[j], col_const);
                let contrib = gate.mul(ctx, a[i * 6 + col], is_col);
                acc = gate.add(ctx, acc, contrib);
            }
            make_public.push(acc);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(inv_perm_cols, args);
}
