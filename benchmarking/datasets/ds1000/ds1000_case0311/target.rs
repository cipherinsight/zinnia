// DS-1000 case 311 — Halo2 baseline.
// Kernel: `np.unravel_index(a.argmax(), a.shape, order='F')` on Float[8, 6].
// 'F' order: idx → (idx % rows, idx // rows), with rows=8.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }  // 48 cells

fn argmax_unravel_f<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 48);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let mut best_val = cells[0];
    let mut best_idx = ctx.load_constant(F::ZERO);
    for k in 1..48usize {
        let is_new = rc.is_less_than(ctx, best_val, cells[k], 128);
        best_val = gate.select(ctx, cells[k], best_val, is_new);
        let idx_k = Constant(F::from(k as u64));
        best_idx = gate.select(ctx, idx_k, best_idx, is_new);
    }
    // 'F'-order unravel for an 8×6 array: row = idx % 8, col = idx / 8.
    let eight = ctx.load_constant(F::from(8u64));
    let (col, row) = rc.div_mod_var(ctx, best_idx, eight, 128, 128);
    make_public.push(row);
    make_public.push(col);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(argmax_unravel_f, args); }
