// npbench_weather_stencils_vadv_vadv__vadv: literal translation of kernel.py (LITERAL_BASELINES.md).
// Every array operation materializes a full plane in source order; canonical nonzero division preserves the optimized divisor domain.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 15; // = utens_stage_0
const NJ: usize = 15; // = utens_stage_1
const NK: usize = 10; // = utens_stage_2

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let utens_stage = inputs.floats("utens_stage", &[NI, NJ, NK])?;
    let u_stage = inputs.floats("u_stage", &[NI, NJ, NK])?;
    let wcon = inputs.floats("wcon", &[NI + 1, NJ, NK])?;
    let u_pos = inputs.floats("u_pos", &[NI, NJ, NK])?;
    let utens = inputs.floats("utens", &[NI, NJ, NK])?;
    let dtr_stage = inputs.float("dtr_stage")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut stage = q.load_floats(ctx, &utens_stage);
    let u = q.load_floats(ctx, &u_stage);
    // wcon[:, :, 0] is never read.
    let wcon: Vec<Option<Cell>> = wcon
        .iter()
        .enumerate()
        .map(|(t, v)| (t % NK != 0).then(|| q.load_float(ctx, *v)))
        .collect();
    let pos = q.load_floats(ctx, &u_pos);
    let tens = q.load_floats(ctx, &utens);
    let dtr = q.load_float(ctx, dtr_stage);

    let mut ccol = vec![ctx.load_zero(); NI * NJ * NK];
    let mut dcol = vec![ctx.load_zero(); NI * NJ * NK];
    let mut data_col = vec![ctx.load_zero(); NI * NJ];
    for k in 0..1 {
        // gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        let t1: Vec<Cell> = (0..NI * NJ)
            .map(|p| {
                q.add(
                    ctx,
                    wcon[((p / NJ + 1) * NJ + p % NJ) * NK + (k + 1)].expect("read k >= 1"),
                    wcon[((p / NJ) * NJ + p % NJ) * NK + (k + 1)].expect("read k >= 1"),
                )
            })
            .collect();
        let t2: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, Constant(fe(1073741824)), t1[p]).cell)
            .collect();
        let gcv: Vec<Cell> = (0..NI * NJ).map(|p| t2[p]).collect();
        // cs = gcv * BET_M
        let t3: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, gcv[p], Constant(fe(2147483648))).cell)
            .collect();
        let cs: Vec<Cell> = (0..NI * NJ).map(|p| t3[p]).collect();
        // ccol[:, :, k] = gcv * BET_P
        let t4: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, gcv[p], Constant(fe(2147483648))).cell)
            .collect();
        for p in 0..NI * NJ {
            ccol[p * NK + (k)] = t4[p];
        }
        // bcol = dtr_stage - ccol[:, :, k]
        let t5: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, dtr, ccol[p * NK + (k)]))
            .collect();
        let bcol: Vec<Cell> = (0..NI * NJ).map(|p| t5[p]).collect();
        // correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        let t6: Vec<Cell> = (0..NI * NJ).map(|p| q.neg(ctx, cs[p])).collect();
        let t7: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, u[p * NK + (k + 1)], u[p * NK + (k)]))
            .collect();
        let t8: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, t6[p], t7[p]).cell)
            .collect();
        let correction_term: Vec<Cell> = (0..NI * NJ).map(|p| t8[p]).collect();
        // dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        let t9: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, dtr, pos[p * NK + (k)]).cell)
            .collect();
        let t10: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t9[p], tens[p * NK + (k)]))
            .collect();
        let t11: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t10[p], stage[p * NK + (k)]))
            .collect();
        let t12: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t11[p], correction_term[p]))
            .collect();
        for p in 0..NI * NJ {
            dcol[p * NK + (k)] = t12[p];
        }
        // divided = 1.0 / bcol
        let t13: Vec<Cell> = (0..NI * NJ)
            .map(|p| {
                q.canonical_div_nonzero(ctx, Constant(fe(4294967296)), bcol[p])
                    .cell
            })
            .collect();
        let divided: Vec<Cell> = (0..NI * NJ).map(|p| t13[p]).collect();
        // ccol[:, :, k] = ccol[:, :, k] * divided
        let t14: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, ccol[p * NK + (k)], divided[p]).cell)
            .collect();
        for p in 0..NI * NJ {
            ccol[p * NK + (k)] = t14[p];
        }
        // dcol[:, :, k] = dcol[:, :, k] * divided
        let t15: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, dcol[p * NK + (k)], divided[p]).cell)
            .collect();
        for p in 0..NI * NJ {
            dcol[p * NK + (k)] = t15[p];
        }
    }
    for k in 1..NK - 1 {
        // gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        let t16: Vec<Cell> = (0..NI * NJ)
            .map(|p| {
                q.add(
                    ctx,
                    wcon[((p / NJ + 1) * NJ + p % NJ) * NK + (k)].expect("read k >= 1"),
                    wcon[((p / NJ) * NJ + p % NJ) * NK + (k)].expect("read k >= 1"),
                )
            })
            .collect();
        let t17: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, Constant(fe(-1073741824)), t16[p]).cell)
            .collect();
        let gav: Vec<Cell> = (0..NI * NJ).map(|p| t17[p]).collect();
        // gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        let t18: Vec<Cell> = (0..NI * NJ)
            .map(|p| {
                q.add(
                    ctx,
                    wcon[((p / NJ + 1) * NJ + p % NJ) * NK + (k + 1)].expect("read k >= 1"),
                    wcon[((p / NJ) * NJ + p % NJ) * NK + (k + 1)].expect("read k >= 1"),
                )
            })
            .collect();
        let t19: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, Constant(fe(1073741824)), t18[p]).cell)
            .collect();
        let gcv: Vec<Cell> = (0..NI * NJ).map(|p| t19[p]).collect();
        // as_ = gav * BET_M
        let t20: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, gav[p], Constant(fe(2147483648))).cell)
            .collect();
        let as_: Vec<Cell> = (0..NI * NJ).map(|p| t20[p]).collect();
        // cs = gcv * BET_M
        let t21: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, gcv[p], Constant(fe(2147483648))).cell)
            .collect();
        let cs: Vec<Cell> = (0..NI * NJ).map(|p| t21[p]).collect();
        // acol = gav * BET_P
        let t22: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, gav[p], Constant(fe(2147483648))).cell)
            .collect();
        let acol: Vec<Cell> = (0..NI * NJ).map(|p| t22[p]).collect();
        // ccol[:, :, k] = gcv * BET_P
        let t23: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, gcv[p], Constant(fe(2147483648))).cell)
            .collect();
        for p in 0..NI * NJ {
            ccol[p * NK + (k)] = t23[p];
        }
        // bcol = dtr_stage - acol - ccol[:, :, k]
        let t24: Vec<Cell> = (0..NI * NJ).map(|p| q.sub(ctx, dtr, acol[p])).collect();
        let t25: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, t24[p], ccol[p * NK + (k)]))
            .collect();
        let bcol: Vec<Cell> = (0..NI * NJ).map(|p| t25[p]).collect();
        // correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        let t26: Vec<Cell> = (0..NI * NJ).map(|p| q.neg(ctx, as_[p])).collect();
        let t27: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, u[p * NK + (k - 1)], u[p * NK + (k)]))
            .collect();
        let t28: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, t26[p], t27[p]).cell)
            .collect();
        let t29: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, u[p * NK + (k + 1)], u[p * NK + (k)]))
            .collect();
        let t30: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, cs[p], t29[p]).cell)
            .collect();
        let t31: Vec<Cell> = (0..NI * NJ).map(|p| q.sub(ctx, t28[p], t30[p])).collect();
        let correction_term: Vec<Cell> = (0..NI * NJ).map(|p| t31[p]).collect();
        // dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        let t32: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, dtr, pos[p * NK + (k)]).cell)
            .collect();
        let t33: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t32[p], tens[p * NK + (k)]))
            .collect();
        let t34: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t33[p], stage[p * NK + (k)]))
            .collect();
        let t35: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t34[p], correction_term[p]))
            .collect();
        for p in 0..NI * NJ {
            dcol[p * NK + (k)] = t35[p];
        }
        // divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        let t36: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, ccol[p * NK + (k - 1)], acol[p]).cell)
            .collect();
        let t37: Vec<Cell> = (0..NI * NJ).map(|p| q.sub(ctx, bcol[p], t36[p])).collect();
        let t38: Vec<Cell> = (0..NI * NJ)
            .map(|p| {
                q.canonical_div_nonzero(ctx, Constant(fe(4294967296)), t37[p])
                    .cell
            })
            .collect();
        let divided: Vec<Cell> = (0..NI * NJ).map(|p| t38[p]).collect();
        // ccol[:, :, k] = ccol[:, :, k] * divided
        let t39: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, ccol[p * NK + (k)], divided[p]).cell)
            .collect();
        for p in 0..NI * NJ {
            ccol[p * NK + (k)] = t39[p];
        }
        // dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
        let t40: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, dcol[p * NK + (k - 1)], acol[p]).cell)
            .collect();
        let t41: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, dcol[p * NK + (k)], t40[p]))
            .collect();
        let t42: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, t41[p], divided[p]).cell)
            .collect();
        for p in 0..NI * NJ {
            dcol[p * NK + (k)] = t42[p];
        }
    }
    for k in NK - 1..NK {
        // gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        let t43: Vec<Cell> = (0..NI * NJ)
            .map(|p| {
                q.add(
                    ctx,
                    wcon[((p / NJ + 1) * NJ + p % NJ) * NK + (k)].expect("read k >= 1"),
                    wcon[((p / NJ) * NJ + p % NJ) * NK + (k)].expect("read k >= 1"),
                )
            })
            .collect();
        let t44: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, Constant(fe(-1073741824)), t43[p]).cell)
            .collect();
        let gav: Vec<Cell> = (0..NI * NJ).map(|p| t44[p]).collect();
        // as_ = gav * BET_M
        let t45: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, gav[p], Constant(fe(2147483648))).cell)
            .collect();
        let as_: Vec<Cell> = (0..NI * NJ).map(|p| t45[p]).collect();
        // acol = gav * BET_P
        let t46: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, gav[p], Constant(fe(2147483648))).cell)
            .collect();
        let acol: Vec<Cell> = (0..NI * NJ).map(|p| t46[p]).collect();
        // bcol = dtr_stage - acol
        let t47: Vec<Cell> = (0..NI * NJ).map(|p| q.sub(ctx, dtr, acol[p])).collect();
        let bcol: Vec<Cell> = (0..NI * NJ).map(|p| t47[p]).collect();
        // correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        let t48: Vec<Cell> = (0..NI * NJ).map(|p| q.neg(ctx, as_[p])).collect();
        let t49: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, u[p * NK + (k - 1)], u[p * NK + (k)]))
            .collect();
        let t50: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, t48[p], t49[p]).cell)
            .collect();
        let correction_term: Vec<Cell> = (0..NI * NJ).map(|p| t50[p]).collect();
        // dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
        let t51: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, dtr, pos[p * NK + (k)]).cell)
            .collect();
        let t52: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t51[p], tens[p * NK + (k)]))
            .collect();
        let t53: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t52[p], stage[p * NK + (k)]))
            .collect();
        let t54: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.add(ctx, t53[p], correction_term[p]))
            .collect();
        for p in 0..NI * NJ {
            dcol[p * NK + (k)] = t54[p];
        }
        // divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        let t55: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, ccol[p * NK + (k - 1)], acol[p]).cell)
            .collect();
        let t56: Vec<Cell> = (0..NI * NJ).map(|p| q.sub(ctx, bcol[p], t55[p])).collect();
        let t57: Vec<Cell> = (0..NI * NJ)
            .map(|p| {
                q.canonical_div_nonzero(ctx, Constant(fe(4294967296)), t56[p])
                    .cell
            })
            .collect();
        let divided: Vec<Cell> = (0..NI * NJ).map(|p| t57[p]).collect();
        // dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
        let t58: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, dcol[p * NK + (k - 1)], acol[p]).cell)
            .collect();
        let t59: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, dcol[p * NK + (k)], t58[p]))
            .collect();
        let t60: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, t59[p], divided[p]).cell)
            .collect();
        for p in 0..NI * NJ {
            dcol[p * NK + (k)] = t60[p];
        }
    }
    for t in 0..1 {
        let k = NK - 1 - t;
        // datacol = dcol[:, :, k]
        let datacol: Vec<Cell> = (0..NI * NJ).map(|p| dcol[p * NK + (k)]).collect();
        // data_col[:] = datacol
        for p in 0..NI * NJ {
            data_col[p] = datacol[p];
        }
        // utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
        let t61: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, datacol[p], pos[p * NK + (k)]))
            .collect();
        let t62: Vec<Cell> = (0..NI * NJ).map(|p| q.mul(ctx, dtr, t61[p]).cell).collect();
        for p in 0..NI * NJ {
            stage[p * NK + (k)] = t62[p];
        }
    }
    for t in 0..NK - 1 {
        let k = NK - 2 - t;
        // datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        let t63: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.mul(ctx, ccol[p * NK + (k)], data_col[p]).cell)
            .collect();
        let t64: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, dcol[p * NK + (k)], t63[p]))
            .collect();
        let datacol: Vec<Cell> = (0..NI * NJ).map(|p| t64[p]).collect();
        // data_col[:] = datacol
        for p in 0..NI * NJ {
            data_col[p] = datacol[p];
        }
        // utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
        let t65: Vec<Cell> = (0..NI * NJ)
            .map(|p| q.sub(ctx, datacol[p], pos[p * NK + (k)]))
            .collect();
        let t66: Vec<Cell> = (0..NI * NJ).map(|p| q.mul(ctx, dtr, t65[p]).cell).collect();
        for p in 0..NI * NJ {
            stage[p * NK + (k)] = t66[p];
        }
    }
    out.floats(&stage);
    Ok(())
}
