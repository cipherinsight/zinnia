// DS-1000 case 450 — Halo2 baseline.
// Kernel: `a[:, second.reshape(-1,1), third]` with a: Float[7,8,9],
// second: Integer[2], third: Integer[2]. Result: Float[7, 2, 2] where
// result[i][p][q] = a[i][second[p]][third[q]].
//
// Halo2 idiom: dynamic gather of a[i][j][k] picks (j, k) from runtime arrays.
// We collapse the (j, k) gather into a 72-way (8×9) match-select per (i,p,q).

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
pub struct CircuitInput {
    pub a: Vec<f64>,         // 7*8*9 = 504 floats, row-major
    pub second: Vec<String>, // 2 decimal integers
    pub third: Vec<String>,  // 2 decimal integers
}

fn fancy_index<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 7 * 8 * 9);
    assert_eq!(input.second.len(), 2);
    assert_eq!(input.third.len(), 2);

    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let second: Vec<AssignedValue<F>> = input.second.iter().map(|s| {
        let v: i64 = s.parse().expect("decimal int");
        ctx.load_witness(F::from(v as u64))
    }).collect();
    let third: Vec<AssignedValue<F>> = input.third.iter().map(|s| {
        let v: i64 = s.parse().expect("decimal int");
        ctx.load_witness(F::from(v as u64))
    }).collect();

    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    // Result is Float[7, 2, 2] = 28 cells, exposed in C-order.
    for i in 0..7usize {
        for p in 0..2usize {
            for q in 0..2usize {
                let mut chosen = zero_fp;
                // 8-way match on second[p]; 9-way match on third[q] — flatten to one chain.
                for j in 0..8usize {
                    let j_const = ctx.load_constant(F::from(j as u64));
                    let is_j = gate.is_equal(ctx, second[p], j_const);
                    for k in 0..9usize {
                        let k_const = ctx.load_constant(F::from(k as u64));
                        let is_k = gate.is_equal(ctx, third[q], k_const);
                        let pick = gate.and(ctx, is_j, is_k);
                        let v = cells[i * 72 + j * 9 + k];
                        chosen = gate.select(ctx, v, chosen, pick);
                    }
                }
                make_public.push(chosen);
            }
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(fancy_index, args); }
