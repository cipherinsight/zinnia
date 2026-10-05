// DS-1000 case 463 — Halo2 baseline.
// Kernel:
//   selection = np.ones((len(a), 1), dtype=bool)
//   selection[1:] = a[1:] != a[:-1]
//   selection &= a != 0
//   result = a[selection].reshape(-1, 1)
// Integer[20, 1] in; dynamic-length out, exposed as padded buffer (len 20)
// plus the actual keep-count.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,        // 20 decimal integers (the [20,1] array flattened)
    pub claimed_len: String,   // actual output length (<= 20)
    pub claimed: Vec<String>,  // 20 decimal integers (padded with 0)
}

fn dedup_nonzero<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let ctx = builder.main(0);
    const N: usize = 20;
    assert_eq!(input.a.len(), N);
    assert_eq!(input.claimed.len(), N);

    let load_int = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int"),
        ));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };

    let a: Vec<AssignedValue<F>> = input.a.iter().map(|s| load_int(ctx, s)).collect();
    let claimed_len = load_int(ctx, &input.claimed_len);
    let claimed: Vec<AssignedValue<F>> = input.claimed.iter().map(|s| load_int(ctx, s)).collect();

    let one = ctx.load_constant(F::ONE);
    let zero_int = ctx.load_constant(F::ZERO);

    let mut selection: Vec<AssignedValue<F>> = Vec::with_capacity(N);
    selection.push(one);
    for i in 1..N {
        let eq = gate.is_equal(ctx, a[i], a[i - 1]);
        let ne = gate.not(ctx, eq);
        selection.push(ne);
    }
    let mut keep: Vec<AssignedValue<F>> = Vec::with_capacity(N);
    for i in 0..N {
        let eq_zero = gate.is_equal(ctx, a[i], zero_int);
        let nz = gate.not(ctx, eq_zero);
        let k = gate.and(ctx, selection[i], nz);
        keep.push(k);
    }

    let mut rank_after: Vec<AssignedValue<F>> = Vec::with_capacity(N);
    let mut running = zero_int;
    for i in 0..N {
        running = gate.add(ctx, running, keep[i]);
        rank_after.push(running);
    }
    let total = rank_after[N - 1];

    let mut out: Vec<AssignedValue<F>> = Vec::with_capacity(N);
    for k in 0..N {
        let target = ctx.load_constant(F::from((k + 1) as u64));
        let mut chosen = zero_int;
        for i in 0..N {
            let rank_eq = gate.is_equal(ctx, rank_after[i], target);
            let pick = gate.and(ctx, keep[i], rank_eq);
            chosen = gate.select(ctx, a[i], chosen, pick);
        }
        out.push(chosen);
    }

    let len_ok = gate.is_equal(ctx, total, claimed_len);
    gate.assert_is_const(ctx, &len_ok, &F::ONE);
    for k in 0..N {
        let eq = gate.is_equal(ctx, out[k], claimed[k]);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }

    make_public.push(total);
    for v in &out { make_public.push(*v); }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(dedup_nonzero, args); }
