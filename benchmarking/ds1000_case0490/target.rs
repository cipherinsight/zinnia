// DS-1000 case 490 — Halo2 baseline.
// Kernel: per-slab threshold replacement on Float[N<=64, M<=64, K<=64] with
// int thresholds n1, n2 (length 5 each):
//     for a, t1, t2 in zip(arr, n1, n2):
//         temp = a.copy()
//         a[np.where(temp < t1)] = 0
//         a[np.where(temp >= t2)] = 30
//         a[np.logical_and(temp >= t1, temp < t2)] += 5
//     return arr
// Sample shape: N=5, M=4, K=4 — 80 floats out.
//
// arr is a DynamicNDArray; we mirror the synthesizer's
// assert_dynamic_ndarray_meta (synthesizer.rs:2515): witness the runtime
// shape, assert each dim is in [0, axis_bound] and the runtime length
// product is ≤ total_bound.
// In-place semantics modelled via snapshot (temp) + element-wise rewrite.

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
    pub arr: Vec<Vec<Vec<f64>>>, // 80 floats (5×4×4), row-major after flatten
    pub n1: Vec<String>,      // 5 decimal ints
    pub n2: Vec<String>,      // 5 decimal ints
}

fn threshold_slabs<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 5;
    const M: usize = 4;
    const K: usize = 4;
    // DynamicNDArrayMeta bounds (mirror Zinnia's `Float[Array, "N<=64 M<=64 K<=64"]`).
    const AXIS_BOUND: u64 = 64;
    const MAX_LENGTH: u64 = 64 * 64 * 64;  // total upper bound when shape is bare

    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);

    assert_eq!(input.arr.len(), N);
    let flat_arr: Vec<f64> = input.arr.iter()
        .flat_map(|slab| slab.iter().flat_map(|row| row.iter().copied()))
        .collect();
    assert_eq!(flat_arr.len(), N * M * K);
    assert_eq!(input.n1.len(), N);
    assert_eq!(input.n2.len(), N);

    // Load arr cells as fp.
    let mut arr: Vec<AssignedValue<F>> = flat_arr.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    // Load n1, n2 ints, then promote to fp (multiply by 1.0 = lift int→fp).
    let load_int = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int"),
        ));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };
    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let n1_fp: Vec<AssignedValue<F>> = input.n1.iter()
        .map(|s| { let v = load_int(ctx, s); fpc.qmul(ctx, v, one_fp) }).collect();
    let n2_fp: Vec<AssignedValue<F>> = input.n2.iter()
        .map(|s| { let v = load_int(ctx, s); fpc.qmul(ctx, v, one_fp) }).collect();

    // DynamicNDArrayMeta shape assertion — mirror synthesizer.rs:2515.
    // rank = 3, dims = [N, M, K].
    let dims: [u64; 3] = [N as u64, M as u64, K as u64];
    let axis_bound = Constant(F::from(AXIS_BOUND));
    let total_bound = Constant(F::from(MAX_LENGTH));
    let mut runtime_len = ctx.load_constant(F::ONE);
    for &d in dims.iter() {
        let dim_cell = ctx.load_witness(F::from(d));
        // dim >= 0 (always true for witnessed unsigned value, but mirror the check).
        let zero = ctx.load_constant(F::ZERO);
        let neg = rc.is_less_than(ctx, dim_cell, zero, 128);
        let non_neg = gate.not(ctx, neg);
        gate.assert_is_const(ctx, &non_neg, &F::ONE);
        // dim <= axis_bound  ⇔  !(axis_bound < dim)
        let over = rc.is_less_than(ctx, axis_bound, dim_cell, 128);
        let in_axis = gate.not(ctx, over);
        gate.assert_is_const(ctx, &in_axis, &F::ONE);
        runtime_len = gate.mul(ctx, runtime_len, dim_cell);
    }
    let total_over = rc.is_less_than(ctx, total_bound, runtime_len, 128);
    let total_in = gate.not(ctx, total_over);
    gate.assert_is_const(ctx, &total_in, &F::ONE);

    // Per-slab loop (i = 0..N). In-place rewrite of arr.
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    let thirty_fp = ctx.load_witness(fpc.quantization(30.0));
    let five_fp = ctx.load_witness(fpc.quantization(5.0));

    for i in 0..N {
        let t1 = n1_fp[i];
        let t2 = n2_fp[i];
        // temp = a.copy() — snapshot before rewrites.
        let base = i * M * K;
        let temp: Vec<AssignedValue<F>> = (0..M * K).map(|x| arr[base + x]).collect();
        // a[temp < t1] = 0
        for x in 0..M * K {
            let lt_t1 = rc.is_less_than(ctx, temp[x], t1, 128);
            arr[base + x] = gate.select(ctx, zero_fp, arr[base + x], lt_t1);
        }
        // a[temp >= t2] = 30
        for x in 0..M * K {
            let lt_t2 = rc.is_less_than(ctx, temp[x], t2, 128);
            let ge_t2 = gate.not(ctx, lt_t2);
            arr[base + x] = gate.select(ctx, thirty_fp, arr[base + x], ge_t2);
        }
        // a[(temp >= t1) & (temp < t2)] += 5
        for x in 0..M * K {
            let lt_t1 = rc.is_less_than(ctx, temp[x], t1, 128);
            let ge_t1 = gate.not(ctx, lt_t1);
            let lt_t2 = rc.is_less_than(ctx, temp[x], t2, 128);
            let mid = gate.and(ctx, ge_t1, lt_t2);
            let plus = fpc.qadd(ctx, arr[base + x], five_fp);
            arr[base + x] = gate.select(ctx, plus, arr[base + x], mid);
        }
    }

    for c in arr.iter() { make_public.push(*c); }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(threshold_slabs, args); }
