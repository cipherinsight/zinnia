// DS-1000 case 358 — Halo2 baseline.
// Kernel:
//   index = np.argsort(a.sum(axis=(1, 2)))   # 3 indices
//   result = b[index, :, :]
// Float[3,3,3] a, Integer[3,3,3] b. Output Integer[3,3,3].
//
// Mirrors kernel.py: sum each (j, k) plane of a into 3 scalar sums,
// run a 3-element compare-and-swap argsort, then gather b along axis 0.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,     // 27 floats (3x3x3 row-major)
    pub b: Vec<String>,  // 27 signed integers (3x3x3 row-major)
}

fn argsort_sum_axis12<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 27);
    assert_eq!(input.b.len(), 27);

    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let mut b: Vec<AssignedValue<F>> = Vec::with_capacity(27);
    for s in &input.b {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().unwrap()));
        b.push(if s.starts_with('-') { gate.neg(ctx, abs) } else { abs });
    }

    // sums[i] = sum_{j,k} a[i,j,k]  for i in 0..3.
    let mut sums: [AssignedValue<F>; 3] = [a[0]; 3];
    for i in 0..3usize {
        let mut s = a[i * 9 + 0 * 3 + 0];
        for j in 0..3usize {
            for k in 0..3usize {
                if j == 0 && k == 0 { continue; }
                s = fpc.qadd(ctx, s, a[i * 9 + j * 3 + k]);
            }
        }
        sums[i] = s;
    }

    // 3-element compare-and-swap argsort on sums (ascending).
    let zero_i = ctx.load_constant(F::ZERO);
    let one_i = ctx.load_constant(F::ONE);
    let two_i = ctx.load_constant(F::from(2u64));
    let mut x = sums;
    let mut idx = [zero_i, one_i, two_i];
    let pairs = [(0usize, 1usize), (1, 2), (0, 1)];
    for (lo, hi) in pairs.iter().copied() {
        let swap = range_chip.is_less_than(ctx, x[hi], x[lo], 128);
        let new_lo_x = gate.select(ctx, x[hi], x[lo], swap);
        let new_hi_x = gate.select(ctx, x[lo], x[hi], swap);
        let new_lo_i = gate.select(ctx, idx[hi], idx[lo], swap);
        let new_hi_i = gate.select(ctx, idx[lo], idx[hi], swap);
        x[lo] = new_lo_x; x[hi] = new_hi_x;
        idx[lo] = new_lo_i; idx[hi] = new_hi_i;
    }

    // result[i, j, k] = b[idx[i], j, k]. Gather over assigned index idx[i].
    for i in 0..3usize {
        let s_idx = idx[i];
        for j in 0..3usize {
            for k in 0..3usize {
                let mut chosen = b[0 * 9 + j * 3 + k];
                for ii in 1..3usize {
                    let ii_cell = ctx.load_constant(F::from(ii as u64));
                    let is_ii = gate.is_equal(ctx, s_idx, ii_cell);
                    chosen = gate.select(ctx, b[ii * 9 + j * 3 + k], chosen, is_ii);
                }
                make_public.push(chosen);
            }
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(argsort_sum_axis12, args); }
