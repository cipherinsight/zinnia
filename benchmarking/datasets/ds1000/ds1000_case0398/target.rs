// DS-1000 case 398 — Halo2 baseline. 2-tap IIR recurrence.
// Kernel:
//   B = np.empty(len(A))
//   for k in range(0, len(B)):
//       if k == 0:    B[k] = a*A[k]
//       else:         B[k] = a*A[k] + b*B[k-1]
// Sample: N=8, a=b=0.5, A=[4,2,1,3,2,0,1,4] → B=[2,2,1.5,2.25,2.125,1.0625,1.03125,2.515625].
//
// Zinnia compiles `np.empty(len(A))` as a DynamicNDArrayMeta, so the
// synthesizer allocates a zkram region (id 10000+array_id, max_length).
// In the handwritten baseline, the access pattern is statically known
// (sequential write + read of B[k-1]), so we materialise B as a Rust Vec
// of AssignedValue. We mirror the synthesizer's shape-bound assertions:
// witness the runtime shape, assert each dim is in [0, axis_bound] and
// the runtime length product is ≤ total_bound.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a_arr: Vec<f64>,   // length N (sample N=8)
    pub a_scalar: f64,
    pub b_scalar: f64,
    pub claimed: Vec<f64>, // length N
}

fn iir_2tap<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 8;
    // DynamicNDArrayMeta bounds (mirror Zinnia's `Float[Array, "N<=64 M<=64 K<=64"]`).
    const MAX_LENGTH: u64 = 64;
    const AXIS_BOUND: u64 = 64;

    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);

    assert_eq!(input.a_arr.len(), N);
    assert_eq!(input.claimed.len(), N);

    let a_cells: Vec<AssignedValue<F>> = input.a_arr.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let a_scalar = ctx.load_witness(fpc.quantization(input.a_scalar));
    let b_scalar = ctx.load_witness(fpc.quantization(input.b_scalar));
    let claimed: Vec<AssignedValue<F>> = input.claimed.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    // DynamicNDArrayMeta shape assertion (mirrors synthesizer assert_dynamic_ndarray_meta
    // at src/prove/halo2/synthesizer.rs:2515). Runtime shape = [N] (rank 1 — only len(A)
    // is used by the recurrence body).
    let n_cell = ctx.load_witness(F::from(N as u64));
    let axis_bound = Constant(F::from(AXIS_BOUND));
    let total_bound = Constant(F::from(MAX_LENGTH));
    // dim >= 0  ⇔  !(dim < 0)  ⇔  !(dim < 0_const). We compare against a 0 constant.
    let zero_const = Constant(F::ZERO);
    let n_lt_zero = rc.is_less_than(ctx, n_cell, zero_const, 128);
    let n_non_neg = gate.not(ctx, n_lt_zero);
    gate.assert_is_const(ctx, &n_non_neg, &F::ONE);
    // dim <= axis_bound  ⇔  !(axis_bound < dim).
    let n_gt_bound = rc.is_less_than(ctx, axis_bound, n_cell, 128);
    let n_in_axis = gate.not(ctx, n_gt_bound);
    gate.assert_is_const(ctx, &n_in_axis, &F::ONE);
    // runtime_len = n; runtime_len <= total_bound.
    let n_gt_total = rc.is_less_than(ctx, total_bound, n_cell, 128);
    let n_in_total = gate.not(ctx, n_gt_total);
    gate.assert_is_const(ctx, &n_in_total, &F::ONE);

    // Recurrence: B[k] = a*A[k] for k=0, else a*A[k] + b*B[k-1].
    // "zkram" allocation is conceptually `vec![0.0; MAX_LENGTH]`; we use a Rust Vec
    // since the access pattern is static (write B[k], read B[k-1]).
    let mut b_arr: Vec<AssignedValue<F>> = Vec::with_capacity(N);
    for k in 0..N {
        let a_ak = fpc.qmul(ctx, a_scalar, a_cells[k]);
        let bk = if k == 0 {
            a_ak
        } else {
            let b_prev = b_arr[k - 1];
            let b_bprev = fpc.qmul(ctx, b_scalar, b_prev);
            fpc.qadd(ctx, a_ak, b_bprev)
        };
        b_arr.push(bk);
    }

    // Assert equality against claimed and expose.
    let eps_pos = Constant(fpc.quantization(0.001));
    let eps_neg = Constant(fpc.quantization(-0.001));
    for k in 0..N {
        let diff = fpc.qsub(ctx, b_arr[k], claimed[k]);
        let lt_p = rc.is_less_than(ctx, diff, eps_pos, 128);
        let gt_n = rc.is_less_than(ctx, eps_neg, diff, 128);
        gate.assert_is_const(ctx, &lt_p, &F::ONE);
        gate.assert_is_const(ctx, &gt_n, &F::ONE);
        make_public.push(b_arr[k]);
    }
    // Shape metadata (matches Zinnia's DynamicNDArray exposure: rank, total length).
    make_public.push(n_cell);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(iir_2tap, args); }
