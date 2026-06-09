// DS-1000 case 412 — Halo2 baseline. `x[x >= 0]` on Float[10].
// Filter: keep only non-negative entries. Output is a fixed-size 10-cell
// buffer with kept values; positions of negatives get masked. Mirrors
// Zinnia IR (95 stmts): 10× ReadF + 10× GreaterThanOrEqualF + 10× SelectF
// + zkram writes + 10× ExposePublicF.
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
pub struct CircuitInput { pub x: Vec<f64> }

fn filter_nonneg<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.x.len(), 10);
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    let cells: Vec<AssignedValue<F>> = input.x.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    // For each cell: ge_zero = !(x < 0); keep = ge_zero ? x : 0.
    for x in &cells {
        let lt = rc.is_less_than(ctx, *x, zero_fp, 128);
        let ge = gate.not(ctx, lt);
        let kept = gate.select(ctx, *x, zero_fp, ge);
        make_public.push(kept);
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(filter_nonneg, args); }
