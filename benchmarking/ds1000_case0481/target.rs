// DS-1000 case 481 — Halo2 baseline.
// Kernel:
//     idx_list = ((x == a) & (y == b))
//     result = idx_list.nonzero()[0]
// Dynamic-length output on Integer[9]. Padded buffer + runtime length:
//   - claimed_len ∈ [0, 9]
//   - claimed[k] for k < claimed_len holds the k-th index where mask=1
//   - claimed[k] for k >= claimed_len is 0 (padding)
// We expose the runtime length and 9-slot buffer as public outputs after
// recomputing the mask and the nonzero indices and asserting they match.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub x: Vec<String>,           // 9 decimal ints
    pub y: Vec<String>,           // 9 decimal ints
    pub a: String,
    pub b_val: String,
    pub claimed_len: String,      // decimal int in [0, 9]
    pub claimed: Vec<String>,     // 9 decimal ints (padded with 0s)
}

fn nonzero_and<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = b.main(0);
    const N: usize = 9;
    assert_eq!(input.x.len(), N);
    assert_eq!(input.y.len(), N);
    assert_eq!(input.claimed.len(), N);
    let load = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int"),
        ));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };
    let xs: Vec<AssignedValue<F>> = input.x.iter().map(|s| load(ctx, s)).collect();
    let ys: Vec<AssignedValue<F>> = input.y.iter().map(|s| load(ctx, s)).collect();
    let a = load(ctx, &input.a);
    let b_cell = load(ctx, &input.b_val);
    let claimed_len = load(ctx, &input.claimed_len);
    let claimed: Vec<AssignedValue<F>> = input.claimed.iter().map(|s| load(ctx, s)).collect();

    // mask[k] = (x[k] == a) AND (y[k] == b)
    let mut mask: Vec<AssignedValue<F>> = Vec::with_capacity(N);
    for k in 0..N {
        let xa = gate.is_equal(ctx, xs[k], a);
        let yb = gate.is_equal(ctx, ys[k], b_cell);
        mask.push(gate.and(ctx, xa, yb));
    }

    // Build the (cumulative) running pointer and write each nonzero index
    // into a 9-slot output buffer. We use a select chain: when mask[k] = 1
    // we set out[pos] = k and pos += 1.
    //
    // To keep this constant-shape we update every cell of the 9-slot buffer
    // each step: out[j] = (mask[k] & (pos == j)) ? k : out[j].
    let zero = ctx.load_constant(F::ZERO);
    let mut out: Vec<AssignedValue<F>> = vec![zero; N];
    let mut pos = zero;
    for k in 0..N {
        let k_cell = ctx.load_constant(F::from(k as u64));
        for j in 0..N {
            let j_eq_pos = gate.is_equal(ctx, pos, Constant(F::from(j as u64)));
            let write = gate.and(ctx, mask[k], j_eq_pos);
            out[j] = gate.select(ctx, k_cell, out[j], write);
        }
        // pos += mask[k]  (integer addition; mask is 0/1).
        pos = gate.add(ctx, pos, mask[k]);
    }

    // Assert computed length matches claimed length.
    let len_eq = gate.is_equal(ctx, pos, claimed_len);
    gate.assert_is_const(ctx, &len_eq, &F::ONE);
    // Assert each output slot matches claimed (full-length compare; padding is 0).
    for k in 0..N {
        let eq = gate.is_equal(ctx, out[k], claimed[k]);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }

    // Expose runtime length + padded buffer.
    make_public.push(pos);
    for k in 0..N { make_public.push(out[k]); }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(nonzero_and, args); }
