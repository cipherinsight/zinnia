// DS-1000 case 300 — Halo2 baseline.
// Kernel: result = np.percentile(a, p) on Float[20] with Integer p.
//
// High-level reimplementation: linear-interpolation percentile.
//   rank = (n - 1) * p / 100
//   lo = floor(rank), hi = min(lo + 1, n - 1)
//   result = a_sorted[lo] + (rank - lo) * (a_sorted[hi] - a_sorted[lo])
//
// We sort via selection sort (O(N^2)) with float comparisons, witness lo
// as a runtime integer and verify floor(rank) via range checks, then
// gather a_sorted[lo] / a_sorted[hi] via indicator-vector dot product.

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
    pub a: Vec<f64>,    // 20 floats
    pub p: String,      // integer percentile in [0, 100]
}

fn percentile_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 20);

    // Load `a`.
    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    // Selection sort (ascending).
    let mut s: Vec<AssignedValue<F>> = a.clone();
    for i in 0..20usize {
        for j in (i + 1)..20usize {
            let j_lt_i = range_chip.is_less_than(ctx, s[j], s[i], 128);
            let new_i = gate.select(ctx, s[j], s[i], j_lt_i);
            let new_j = gate.select(ctx, s[i], s[j], j_lt_i);
            s[i] = new_i;
            s[j] = new_j;
        }
    }

    // Read p as a signed integer, promote to float.
    let p_abs = ctx.load_witness(F::from_u128(
        input.p.trim_start_matches('-').parse::<u128>().expect("decimal"),
    ));
    let p_int = if input.p.starts_with('-') { gate.neg(ctx, p_abs) } else { p_abs };
    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let p_fp = fpc.qmul(ctx, p_int, one_fp);

    // rank = (N - 1) * p / 100.
    let n_minus_1 = ctx.load_witness(fpc.quantization(19.0));
    let hundred = ctx.load_witness(fpc.quantization(100.0));
    let num = fpc.qmul(ctx, n_minus_1, p_fp);
    let rank = fpc.qdiv(ctx, num, hundred);

    // Witness lo = floor(rank) as an integer in [0, 19]; verify
    // lo_fp <= rank < lo_fp + 1.
    let p_native: i64 = input.p.parse().expect("decimal");
    let rank_native: f64 = 19.0 * (p_native as f64) / 100.0;
    let lo_native: u64 = rank_native.floor().max(0.0) as u64;
    let lo_int = ctx.load_witness(F::from(lo_native));
    let lo_fp = fpc.qmul(ctx, lo_int, one_fp);
    let lo_plus_1 = {
        let one_int = ctx.load_constant(F::ONE);
        let s = gate.add(ctx, lo_int, one_int);
        fpc.qmul(ctx, s, one_fp)
    };
    // floor: lo_fp <= rank → !(rank < lo_fp).
    let rank_lt_lo = range_chip.is_less_than(ctx, rank, lo_fp, 128);
    let ok_lo = gate.not(ctx, rank_lt_lo);
    gate.assert_is_const(ctx, &ok_lo, &F::ONE);
    // floor: rank < lo + 1.
    let ok_hi = range_chip.is_less_than(ctx, rank, lo_plus_1, 128);
    gate.assert_is_const(ctx, &ok_hi, &F::ONE);
    // lo must be in [0, 19].
    let nineteen = ctx.load_constant(F::from(19u64));
    let lo_le_19 = {
        let lt = range_chip.is_less_than(ctx, nineteen, lo_int, 128);
        gate.not(ctx, lt)
    };
    gate.assert_is_const(ctx, &lo_le_19, &F::ONE);

    // hi = min(lo + 1, 19). hi_int = lo + 1 if lo < 19 else lo.
    let lo_eq_19 = gate.is_equal(ctx, lo_int, nineteen);
    let lo_plus_1_int = {
        let one_int = ctx.load_constant(F::ONE);
        gate.add(ctx, lo_int, one_int)
    };
    let hi_int = gate.select(ctx, lo_int, lo_plus_1_int, lo_eq_19);

    // Gather s[lo] and s[hi] via indicator-vector inner product.
    let select_indexed = |gate: &GateChip<F>, ctx: &mut halo2_base::Context<F>,
                          arr: &[AssignedValue<F>], idx: AssignedValue<F>| -> AssignedValue<F> {
        let mut sum = ctx.load_constant(F::ZERO);
        for k in 0..arr.len() {
            let k_const = Constant(F::from(k as u64));
            let is_k = gate.is_equal(ctx, idx, k_const);
            let contrib = gate.mul(ctx, arr[k], is_k);
            sum = gate.add(ctx, sum, contrib);
        }
        sum
    };
    let s_lo = select_indexed(&gate, ctx, &s, lo_int);
    let s_hi = select_indexed(&gate, ctx, &s, hi_int);

    // result = s_lo + (rank - lo_fp) * (s_hi - s_lo).
    let frac = fpc.qsub(ctx, rank, lo_fp);
    let diff = fpc.qsub(ctx, s_hi, s_lo);
    let term = fpc.qmul(ctx, frac, diff);
    let result = fpc.qadd(ctx, s_lo, term);

    make_public.push(result);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(percentile_circuit, args);
}
