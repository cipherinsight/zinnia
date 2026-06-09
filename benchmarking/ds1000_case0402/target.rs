// DS-1000 case 402 — Halo2 baseline. `np.ravel_multi_index(index, dims, order='F')`.
// Fortran ordering: linear = i0 + d0*i1 + d0*d1*i2 + d0*d1*d2*i3 + d0*d1*d2*d3*i4
// Sample: dims=[8,9,8,8,8], index=[2,3,4,5,6] → 30842.
// The array `a` of shape (8,9,8,8,8) is part of the signature and witnessed
// (~36864 cells) but unused by the computation, matching Zinnia's emission
// of 8*9*8*8*8 ReadFloatIR statements.

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
    pub dims: Vec<String>,   // 5 ints
    pub a: Vec<f64>,         // 8*9*8*8*8 = 36864 floats (unused)
    pub index: Vec<String>,  // 5 ints
    pub claimed: String,     // expected linear index
}

fn ravel_multi_index_f<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const TOTAL: usize = 8 * 9 * 8 * 8 * 8;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);

    assert_eq!(input.dims.len(), 5);
    assert_eq!(input.index.len(), 5);
    assert_eq!(input.a.len(), TOTAL);

    // `a` is unused by the kernel; Zinnia v7 DCE-eliminates the dead
    // ReadFloatIR ops, so HW skips the loads too (otherwise HW carries
    // ~36 864 dead advice cells the comparison is biased on).

    let dims: Vec<AssignedValue<F>> = input.dims.iter()
        .map(|s| ctx.load_witness(F::from_u128(s.parse::<u128>().unwrap()))).collect();
    let index: Vec<AssignedValue<F>> = input.index.iter()
        .map(|s| ctx.load_witness(F::from_u128(s.parse::<u128>().unwrap()))).collect();
    let claimed = ctx.load_witness(F::from_u128(input.claimed.parse::<u128>().unwrap()));

    // Bounds-check semantic: index[k] must lie in [0, dims[k]).
    // Mirrors Zinnia's per-index `0 <= index[k] < dims[k]` discharge
    // (5 GE + 5 LT + 5 AND + 5 Assert in the Zinnia IR). Without this
    // a malicious prover could supply out-of-range index values that
    // still satisfy the Horner+assert chain below but yield a nonsense
    // linear-index public output.
    for k in 0..5 {
        rc.check_less_than(ctx, index[k], dims[k], 32);
    }

    // Fortran order: linear = sum_k index[k] * prod_{j<k} dims[j].
    // result = index[0]
    //        + dims[0] * index[1]
    //        + dims[0]*dims[1] * index[2]
    //        + dims[0]*dims[1]*dims[2] * index[3]
    //        + dims[0]*dims[1]*dims[2]*dims[3] * index[4]
    let mut stride = ctx.load_constant(F::ONE);
    let mut acc = gate.mul(ctx, index[0], stride);  // index[0] * 1
    for k in 1..5 {
        stride = gate.mul(ctx, stride, dims[k - 1]);
        let term = gate.mul(ctx, index[k], stride);
        acc = gate.add(ctx, acc, term);
    }

    // Assert equality.
    let ok = gate.is_equal(ctx, acc, claimed);
    gate.assert_is_const(ctx, &ok, &F::ONE);
    make_public.push(acc);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(ravel_multi_index_f, args); }
