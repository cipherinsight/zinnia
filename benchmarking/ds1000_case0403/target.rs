// DS-1000 case 403 — Halo2 baseline. `np.ravel_multi_index(index, dims, order='C')`.
// C ordering: linear = ((((i0*d1 + i1)*d2 + i2)*d3 + i3)*d4 + i4).
// Sample: dims=[8,9,8,8,8], index=[2,3,4,5,6] → 11054.
// As in case 0402, `a` is witnessed but unused.

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
    pub dims: Vec<String>,
    pub a: Vec<f64>,         // 36864 unused
    pub index: Vec<String>,
    pub claimed: String,
}

fn ravel_multi_index_c<F: ScalarField>(
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
    // ReadFloatIR ops, so HW skips the loads too.

    let dims: Vec<AssignedValue<F>> = input.dims.iter()
        .map(|s| ctx.load_witness(F::from_u128(s.parse::<u128>().unwrap()))).collect();
    let index: Vec<AssignedValue<F>> = input.index.iter()
        .map(|s| ctx.load_witness(F::from_u128(s.parse::<u128>().unwrap()))).collect();
    let claimed = ctx.load_witness(F::from_u128(input.claimed.parse::<u128>().unwrap()));

    // Bounds-check semantic (mirrors Zinnia's per-index discharge):
    // index[k] must lie in [0, dims[k]). Without this a malicious prover
    // could supply out-of-range index values that satisfy the Horner
    // chain below but yield a nonsense linear-index public output.
    // Honest 64-bit integer domain (same convention as Zinnia): every
    // witness integer that feeds a comparison is range-checked to 64 bits
    // once, and the comparison itself runs at 64-bit width.
    for k in 0..5 {
        rc.range_check(ctx, index[k], 64);
        rc.range_check(ctx, dims[k], 64);
        rc.check_less_than(ctx, index[k], dims[k], 64);
    }

    // C order Horner: acc = ((((i0*d1 + i1)*d2 + i2)*d3 + i3)*d4 + i4).
    let mut acc = index[0];
    for k in 1..5 {
        let scaled = gate.mul(ctx, acc, dims[k]);
        acc = gate.add(ctx, scaled, index[k]);
    }

    let ok = gate.is_equal(ctx, acc, claimed);
    gate.assert_is_const(ctx, &ok, &F::ONE);
    make_public.push(acc);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(ravel_multi_index_c, args); }
