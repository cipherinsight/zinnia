// DS-1000 case 347 — Halo2 baseline.
// Kernel:
//   a[row-1, :] /= divide_number
//   result = np.multiply.reduce(a[row-1, :])
// On Float[8, 6] with runtime integer `row` and `divide_number`. Output: scalar float.
// Mirror kernel.py: pick the row via select over i=0..8, divide each cell by
// the (int → fp) divisor, then multiply all 6 results together.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,            // 48 floats row-major (8 x 6)
    pub row: String,            // signed decimal integer
    pub divide_number: String,  // signed decimal integer
}

fn rowdiv_product<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);

    assert_eq!(input.a.len(), 48);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    // Load `row` and `divide_number` as signed integers.
    let row_abs = ctx.load_witness(F::from_u128(
        input.row.trim_start_matches('-').parse::<u128>().unwrap()));
    let row = if input.row.starts_with('-') { gate.neg(ctx, row_abs) } else { row_abs };
    let one = ctx.load_constant(F::ONE);
    let row_idx = gate.sub(ctx, row, one);

    let dn_abs = ctx.load_witness(F::from_u128(
        input.divide_number.trim_start_matches('-').parse::<u128>().unwrap()));
    let dn_int = if input.divide_number.starts_with('-') { gate.neg(ctx, dn_abs) } else { dn_abs };
    // FloatCast: divide_number int → fp. This is a plain field multiply by
    // Q; using qmul here would divide by Q again and leave the raw integer.
    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let dn_fp = gate.mul(ctx, dn_int, one_fp);

    let zero_fp = ctx.load_witness(fpc.quantization(0.0));

    // Full numpy semantics: negative indices wrap (row = 0 selects the
    // last row). Dual-equality indicators, hoisted across columns:
    // ind[i] = (idx == i) + (idx == i - 8); membership sum == 1 enforces
    // -8 <= idx < 8 (numpy IndexError otherwise).
    let mut inds = Vec::with_capacity(8);
    let mut hits = ctx.load_constant(F::ZERO);
    for i in 0..8usize {
        let i_cell = ctx.load_constant(F::from(i as u64));
        let in_cell = ctx.load_constant(-F::from((8 - i) as u64));
        let is_i = gate.is_equal(ctx, row_idx, i_cell);
        let is_in = gate.is_equal(ctx, row_idx, in_cell);
        let ind = gate.add(ctx, is_i, is_in);
        hits = gate.add(ctx, hits, ind);
        inds.push(ind);
    }
    let one_c = ctx.load_constant(F::ONE);
    let hits_ok = gate.is_equal(ctx, hits, one_c);
    gate.assert_is_const(ctx, &hits_ok, &F::ONE);
    let mut row_vals: Vec<AssignedValue<F>> = Vec::with_capacity(6);
    for j in 0..6usize {
        let mut chosen = zero_fp;
        for i in 0..8usize {
            chosen = gate.select(ctx, cells[i * 6 + j], chosen, inds[i]);
        }
        // Divide cell by divide_number.
        let divided = fpc.qdiv(ctx, chosen, dn_fp);
        row_vals.push(divided);
    }

    // np.multiply.reduce: chain qmul over the 6 divided values.
    let mut prod = row_vals[0];
    for k in 1..6usize {
        prod = fpc.qmul(ctx, prod, row_vals[k]);
    }
    make_public.push(prod);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(rowdiv_product, args); }
