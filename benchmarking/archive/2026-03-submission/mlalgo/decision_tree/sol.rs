use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::Constant,
};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;
use halo2_base::poseidon::hasher::PoseidonHasher;
use serde::{Serialize, Deserialize};
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub training_x: Vec<Vec<f64>>, // 10 x 2
    pub training_y: Vec<f64>,      // 10
    pub testing_x: Vec<Vec<f64>>,  // 2 x 2
    pub testing_y: Vec<f64>,       // 2
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    _make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63;
    let gate = GateChip::<F>::default();
    let range = builder.range_chip();
    let fp = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // constants
    let n_train = 10usize;
    let n_test = 2usize;
    let n_feat = 2usize;
    let n_thr = 3usize;

    let zero = ctx.load_constant(fp.quantization(0.0));
    let one = ctx.load_constant(fp.quantization(1.0));
    let two = ctx.load_constant(fp.quantization(2.0));
    let thr_vals = [
        ctx.load_constant(fp.quantization(0.5)),
        ctx.load_constant(fp.quantization(1.0)),
        ctx.load_constant(fp.quantization(1.5)),
    ];

    // load training data
    let mut tr_x = vec![vec![zero; n_feat]; n_train];
    for i in 0..n_train {
        for j in 0..n_feat {
            tr_x[i][j] = ctx.load_witness(fp.quantization(input.training_x[i][j]));
        }
    }
    let tr_y: Vec<AssignedValue<F>> = (0..n_train)
        .map(|i| ctx.load_witness(fp.quantization(input.training_y[i])))
        .collect();

    // load testing data
    let mut te_x = vec![vec![zero; n_feat]; n_test];
    for i in 0..n_test {
        for j in 0..n_feat {
            te_x[i][j] = ctx.load_witness(fp.quantization(input.testing_x[i][j]));
        }
    }
    let te_y: Vec<AssignedValue<F>> = (0..n_test)
        .map(|i| ctx.load_witness(fp.quantization(input.testing_y[i])))
        .collect();

    // best parameters
    let mut best_err = ctx.load_constant(fp.quantization(999.0));
    let mut best_feat_r = ctx.load_constant(fp.quantization(0.0));
    let mut best_thr_r = ctx.load_constant(fp.quantization(0.0));
    let mut best_feat_l = ctx.load_constant(fp.quantization(0.0));
    let mut best_thr_l = ctx.load_constant(fp.quantization(0.0));
    let mut best_feat_rr = ctx.load_constant(fp.quantization(0.0));
    let mut best_thr_rr = ctx.load_constant(fp.quantization(0.0));

    // Enumerate all 216 trees
    for fr in 0..n_feat {
        for tr in 0..n_thr {
            for fl in 0..n_feat {
                for tl in 0..n_thr {
                    for fr2 in 0..n_feat {
                        for tr2 in 0..n_thr {
                            // compute training error
                            let mut err = zero;
                            for i in 0..n_train {
                                // Root decision: go_right = x[fr] >= thr[tr]
                                let lt_root = range.is_less_than(ctx, tr_x[i][fr], thr_vals[tr], 128);
                                let go_right = gate.not(ctx, lt_root);

                                // left branch
                                let lt_left = range.is_less_than(ctx, tr_x[i][fl], thr_vals[tl], 128);
                                let pred_left = gate.not(ctx, lt_left);
                                // right branch
                                let lt_right = range.is_less_than(ctx, tr_x[i][fr2], thr_vals[tr2], 128);
                                let pred_right = gate.not(ctx, lt_right);

                                let pred = gate.select(ctx, pred_right, pred_left, go_right);

                                // mismatch = (pred != y)
                                let diff = fp.qsub(ctx, pred, tr_y[i]);
                                let abs_diff = {
                                    let neg_diff = fp.neg(ctx, diff);
                                    let lt0 = range.is_less_than(ctx, diff, zero, 128);
                                    gate.select(ctx, neg_diff, diff, lt0)
                                };
                                err = fp.qadd(ctx, err, abs_diff);
                            }

                            // check if err < best_err
                            let better = range.is_less_than(ctx, err, best_err, 128);

                            // tie case: err == best_err
                            let eq_err = gate.is_equal(ctx, err, best_err);

                            // lexicographic tie-breaking
                            let mut replace = better;
                            if n_feat > 0 {
                                let frv = Constant(fp.quantization(fr as f64));
                                let eq_fr = gate.is_equal(ctx, frv, best_feat_r);
                                let less_fr = range.is_less_than(ctx, frv, best_feat_r, 128);
                                let cond1 = gate.and(ctx, eq_err, less_fr);
                                let tmp = gate.and(ctx, eq_fr, less_fr);
                                let cond2 = gate.and(ctx, eq_err, tmp);
                                replace = gate.or(ctx, replace, cond1);
                                replace = gate.or(ctx, replace, cond2);
                            }

                            // apply select updates
                            best_err = gate.select(ctx, err, best_err, replace);
                            best_feat_r = gate.select(ctx, Constant(fp.quantization(fr as f64)), best_feat_r, replace);
                            best_thr_r = gate.select(ctx, Constant(fp.quantization(tr as f64)), best_thr_r, replace);
                            best_feat_l = gate.select(ctx, Constant(fp.quantization(fl as f64)), best_feat_l, replace);
                            best_thr_l = gate.select(ctx, Constant(fp.quantization(tl as f64)), best_thr_l, replace);
                            best_feat_rr = gate.select(ctx, Constant(fp.quantization(fr2 as f64)), best_feat_rr, replace);
                            best_thr_rr = gate.select(ctx, Constant(fp.quantization(tr2 as f64)), best_thr_rr, replace);
                        }
                    }
                }
            }
        }
    }

    // Test set evaluation
    // ---- test evaluation (fixed-index version, all selection by gates) ----
    let mut test_errors = zero;
    for i in 0..n_test {
        // one-hot masks for thresholds and features
        let mut thr_mask = vec![zero; n_thr];
        let mut feat_mask = vec![zero; n_feat];

        for j in 0..n_feat {
            let eq = gate.is_equal(ctx, Constant(fp.quantization(j as f64)), best_feat_r);
            feat_mask[j] = eq;
        }
        for j in 0..n_thr {
            let eq = gate.is_equal(ctx, Constant(fp.quantization(j as f64)), best_thr_r);
            thr_mask[j] = eq;
        }

        // root threshold / feature selection
        let mut sel_thr_r = zero;
        for j in 0..n_thr {
            let tmp = fp.qmul(ctx, thr_mask[j], thr_vals[j]);
            sel_thr_r = fp.qadd(ctx, sel_thr_r, tmp);
        }
        let mut sel_feat_r_x = zero;
        for j in 0..n_feat {
            let tmp = fp.qmul(ctx, feat_mask[j], te_x[i][j]);
            sel_feat_r_x = fp.qadd(ctx, sel_feat_r_x, tmp);
        }

        // left branch
        let mut thr_mask_l = vec![zero; n_thr];
        let mut feat_mask_l = vec![zero; n_feat];
        for j in 0..n_feat {
            let eq = gate.is_equal(ctx, Constant(fp.quantization(j as f64)), best_feat_l);
            feat_mask_l[j] = eq;
        }
        for j in 0..n_thr {
            let eq = gate.is_equal(ctx, Constant(fp.quantization(j as f64)), best_thr_l);
            thr_mask_l[j] = eq;
        }
        let mut sel_thr_l = zero;
        for j in 0..n_thr {
            let tmp = fp.qmul(ctx, thr_mask_l[j], thr_vals[j]);
            sel_thr_l = fp.qadd(ctx, sel_thr_l, tmp);
        }
        let mut sel_feat_l_x = zero;
        for j in 0..n_feat {
            let tmp = fp.qmul(ctx, feat_mask_l[j], te_x[i][j]);
            sel_feat_l_x = fp.qadd(ctx, sel_feat_l_x, tmp);
        }

        // right branch
        let mut thr_mask_r = vec![zero; n_thr];
        let mut feat_mask_r = vec![zero; n_feat];
        for j in 0..n_feat {
            let eq = gate.is_equal(ctx, Constant(fp.quantization(j as f64)), best_feat_rr);
            feat_mask_r[j] = eq;
        }
        for j in 0..n_thr {
            let eq = gate.is_equal(ctx, Constant(fp.quantization(j as f64)), best_thr_rr);
            thr_mask_r[j] = eq;
        }
        let mut sel_thr_rr = zero;
        for j in 0..n_thr {
            let tmp = fp.qmul(ctx, thr_mask_r[j], thr_vals[j]);
            sel_thr_rr = fp.qadd(ctx, sel_thr_rr, tmp);
        }
        let mut sel_feat_rr_x = zero;
        for j in 0..n_feat {
            let tmp = fp.qmul(ctx, feat_mask_r[j], te_x[i][j]);
            sel_feat_rr_x = fp.qadd(ctx, sel_feat_rr_x, tmp);
        }

        // now compute decisions with arithmetic selects
        let lt_root = range.is_less_than(ctx, sel_feat_r_x, sel_thr_r, 128);
        let go_right = gate.not(ctx, lt_root);
        let lt_left = range.is_less_than(ctx, sel_feat_l_x, sel_thr_l, 128);
        let pred_left = gate.not(ctx, lt_left);
        let lt_right = range.is_less_than(ctx, sel_feat_rr_x, sel_thr_rr, 128);
        let pred_right = gate.not(ctx, lt_right);
        let pred = gate.select(ctx, pred_right, pred_left, go_right);

        // accumulate absolute difference
        let diff = fp.qsub(ctx, pred, te_y[i]);
        let neg_diff = fp.neg(ctx, diff);
        let lt0 = range.is_less_than(ctx, diff, zero, 128);
        let abs_diff = gate.select(ctx, neg_diff, diff, lt0);
        test_errors = fp.qadd(ctx, test_errors, abs_diff);
    }

    // Require ≤ 1
    let ok = range.is_less_than(ctx, test_errors, Constant(fp.quantization(1.1)), 128);
    gate.assert_is_const(ctx, &ok, &F::ONE);

}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
