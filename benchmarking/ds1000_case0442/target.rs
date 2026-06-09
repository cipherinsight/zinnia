// DS-1000 case 442 — Halo2 baseline.
// Kernel: C = A[~np.in1d(A, B)] on Integer[20] × Integer[3].
// Dynamic-length output: lower as fixed-size 20-cell masked buffer + length.
// keep_mask[i] = !any_j(A[i] == B[j])  (3-way OR over equality checks).
// Expose: keep_mask[i] * A[i] for each i, plus the total kept count.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<i64>, pub b: Vec<i64> }

fn in1d_neg<F: ScalarField>(
    bld: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    // FixedPointChip is constructed only to align with the project's harness; not used.
    let _fpc = FixedPointChip::<F, P>::default(bld);
    let ctx = bld.main(0);
    assert_eq!(input.a.len(), 20);
    assert_eq!(input.b.len(), 3);

    let a_cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(F::from(*v as u64))).collect();
    let b_cells: Vec<AssignedValue<F>> = input.b.iter()
        .map(|v| ctx.load_witness(F::from(*v as u64))).collect();

    let mut count = ctx.load_constant(F::ZERO);
    let zero = ctx.load_constant(F::ZERO);
    for i in 0..20 {
        // any_eq = (A[i] == B[0]) ∨ (A[i] == B[1]) ∨ (A[i] == B[2])
        let e0 = gate.is_equal(ctx, a_cells[i], b_cells[0]);
        let e1 = gate.is_equal(ctx, a_cells[i], b_cells[1]);
        let e2 = gate.is_equal(ctx, a_cells[i], b_cells[2]);
        let or01 = gate.or(ctx, e0, e1);
        let any_eq = gate.or(ctx, or01, e2);
        let keep = gate.not(ctx, any_eq);  // 1 - any_eq
        let kept_val = gate.select(ctx, a_cells[i], zero, keep);
        make_public.push(kept_val);
        count = gate.add(ctx, count, keep);
    }
    // Expose the kept count as well (length of the dynamic output).
    make_public.push(count);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(in1d_neg, args); }
