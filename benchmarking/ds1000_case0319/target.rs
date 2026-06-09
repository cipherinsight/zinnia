// DS-1000 case 319 — Halo2 baseline.
// Kernel:
//   c = empty_like(permutation)
//   c[permutation] = arange(len(permutation))   # inverse permutation
//   result = a[c, :, :]                          # reorder rows by c
// Inputs: a: Float[8, 6, 8], permutation: Integer[8].
//
// High-level reimplementation:
//   1. Build inverse c (length 8): c[k] = sum_j j * (perm[j] == k).
//   2. For each output cell result[i][j][k] = a[c[i]][j][k]. The first
//      axis is gathered via an indicator inner product across the 8
//      possible source planes.

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
    pub a: Vec<f64>,              // 8*6*8 = 384 floats (i, j, k C-order)
    pub permutation: Vec<String>, // 8 integers
}

fn inv_perm_rows_3d<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 8 * 6 * 8);
    assert_eq!(input.permutation.len(), 8);

    // Load a (C-order: i*48 + j*8 + k).
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
    let mut c: Vec<AssignedValue<F>> = Vec::with_capacity(8);
    for k in 0..8usize {
        let k_const = Constant(F::from(k as u64));
        let mut sum = ctx.load_constant(F::ZERO);
        for j in 0..8usize {
            let is_eq = gate.is_equal(ctx, perm[j], k_const);
            let j_const = Constant(F::from(j as u64));
            let contrib = gate.mul(ctx, j_const, is_eq);
            sum = gate.add(ctx, sum, contrib);
        }
        c.push(sum);
    }

    // result[i][j][k] = a[c[i]][j][k]. Gather first axis via indicator
    // inner product over the 8 possible source planes.
    for i in 0..8usize {
        // Precompute the 8 indicator bits for c[i].
        let mut indicators: Vec<AssignedValue<F>> = Vec::with_capacity(8);
        for src in 0..8usize {
            let src_const = Constant(F::from(src as u64));
            indicators.push(gate.is_equal(ctx, c[i], src_const));
        }
        for j in 0..6usize {
            for k in 0..8usize {
                let mut acc = ctx.load_constant(F::ZERO);
                for src in 0..8usize {
                    let off = src * 48 + j * 8 + k;
                    let contrib = gate.mul(ctx, a[off], indicators[src]);
                    acc = gate.add(ctx, acc, contrib);
                }
                make_public.push(acc);
            }
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(inv_perm_rows_3d, args);
}
