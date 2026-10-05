// DS-1000 case 355 — Halo2 baseline.
// Kernel:
//   sort_indices = np.argsort(a, axis=0)            # 3x3x3 indices
//   static_indices = np.indices(a.shape)
//   c = b[sort_indices, static_indices[1], static_indices[2]]
// Float[3,3,3] a, Integer[3,3,3] b. For every (j,k), argsort the 3 elements
// a[:,j,k] ascending; then c[i,j,k] = b[sort_indices[i,j,k], j, k].
//
// Mirrors kernel.py + the SP1 reference: 3-element compare-and-swap network
// over (value, index) pairs for each of the 9 (j,k) pairs, then gather b
// via a select chain over the resulting permutation index.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
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

fn argsort_axis0_gather<F: ScalarField>(
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

    let zero_i = ctx.load_constant(F::ZERO);
    let mut sort_indices: [[[AssignedValue<F>; 3]; 3]; 3] = [[[zero_i; 3]; 3]; 3];

    for j in 0..3usize {
        for k in 0..3usize {
            let v0 = a[0 * 9 + j * 3 + k];
            let v1 = a[1 * 9 + j * 3 + k];
            let v2 = a[2 * 9 + j * 3 + k];
            let one_i = ctx.load_constant(F::ONE);
            let two_i = ctx.load_constant(F::from(2u64));
            let mut x = [v0, v1, v2];
            let mut idx = [zero_i, one_i, two_i];

            // Compare-and-swap network for 3 elements: (0,1), (1,2), (0,1).
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

            for i in 0..3usize {
                sort_indices[i][j][k] = idx[i];
            }
        }
    }

    // c[i,j,k] = b[sort_indices[i,j,k], j, k]. Select chain over the assigned index.
    for i in 0..3usize {
        for j in 0..3usize {
            for k in 0..3usize {
                let s_idx = sort_indices[i][j][k];
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

fn main() { env_logger::init(); let args = Cli::parse(); run(argsort_axis0_gather, args); }
