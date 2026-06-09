// DS-1000 case 353 — Halo2 baseline.
// Kernel:
//   dims = np.maximum(B.max(0), A.max(0)) + 1
//   output = A[~np.in1d(np.ravel_multi_index(A.T, dims),
//                       np.ravel_multi_index(B.T, dims))]
// A: Integer[2,3], B: Integer[7,3]. Output is dynamic-shape, so we
// emit a fixed-shape result: for each of the 2 rows in A we expose
// (keep_mask_bit, row[0], row[1], row[2]). A row is "kept" when its
// ravel index is NOT present in any row of B.
//
// Mirrors kernel.py's algorithm: per-column max across A and B → dims;
// ravel each row; in1d via O(N*M) equality check.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,  // 6 signed integers (2x3 row-major)
    pub b: Vec<String>,  // 21 signed integers (7x3 row-major)
}

fn setdiff_rows<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 6);
    assert_eq!(input.b.len(), 21);

    let mut a: Vec<AssignedValue<F>> = Vec::with_capacity(6);
    for s in &input.a {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().unwrap()));
        a.push(if s.starts_with('-') { gate.neg(ctx, abs) } else { abs });
    }
    let mut b: Vec<AssignedValue<F>> = Vec::with_capacity(21);
    for s in &input.b {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().unwrap()));
        b.push(if s.starts_with('-') { gate.neg(ctx, abs) } else { abs });
    }

    // dims[k] = max(max_i a[i][k], max_i b[i][k]) + 1 for k in 0..3.
    let one_int = ctx.load_constant(F::ONE);
    let mut dims: [AssignedValue<F>; 3] = [a[0]; 3];
    for k in 0..3usize {
        // running max across A column k and B column k.
        let mut m = a[0 * 3 + k];
        for i in 1..2usize {
            let v = a[i * 3 + k];
            let lt = range_chip.is_less_than(ctx, m, v, 128);
            m = gate.select(ctx, v, m, lt);
        }
        for i in 0..7usize {
            let v = b[i * 3 + k];
            let lt = range_chip.is_less_than(ctx, m, v, 128);
            m = gate.select(ctx, v, m, lt);
        }
        dims[k] = gate.add(ctx, m, one_int);
    }

    // ravel_multi_index of A.T with shape dims: for each row r in A,
    // idx = ((a[r,0] * dims[1]) + a[r,1]) * dims[2] + a[r,2].
    let mut a_idx: [AssignedValue<F>; 2] = [a[0]; 2];
    for r in 0..2usize {
        let t0 = gate.mul(ctx, a[r * 3 + 0], dims[1]);
        let t1 = gate.add(ctx, t0, a[r * 3 + 1]);
        let t2 = gate.mul(ctx, t1, dims[2]);
        a_idx[r] = gate.add(ctx, t2, a[r * 3 + 2]);
    }
    let mut b_idx: [AssignedValue<F>; 7] = [b[0]; 7];
    for r in 0..7usize {
        let t0 = gate.mul(ctx, b[r * 3 + 0], dims[1]);
        let t1 = gate.add(ctx, t0, b[r * 3 + 1]);
        let t2 = gate.mul(ctx, t1, dims[2]);
        b_idx[r] = gate.add(ctx, t2, b[r * 3 + 2]);
    }

    // in1d: in_b[r] = OR over s of (a_idx[r] == b_idx[s]). keep[r] = !in_b[r].
    let zero = ctx.load_constant(F::ZERO);
    for r in 0..2usize {
        let mut in_b = zero;
        for s in 0..7usize {
            let eq = gate.is_equal(ctx, a_idx[r], b_idx[s]);
            in_b = gate.or(ctx, in_b, eq);
        }
        let keep = gate.not(ctx, in_b);
        // Expose (keep_mask, row[0], row[1], row[2]) for each row.
        make_public.push(keep);
        for j in 0..3usize {
            make_public.push(a[r * 3 + j]);
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(setdiff_rows, args); }
