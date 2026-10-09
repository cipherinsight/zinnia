// npbench_weather_stencils_vadv_vadv__vadv: literal translation of kernel.py (LITERAL_BASELINES.md).
// Every array operation materializes a full plane in source order; canonical nonzero division preserves the optimized divisor domain.
// @outputs: float[NI * NJ * NK]

use noir_hw::{canonical_div_nonzero, check_input, check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global NI: u32 = 15; // = utens_stage_0
global NJ: u32 = 15; // = utens_stage_1
global NK: u32 = 10; // = utens_stage_2

fn main(
    utens_stage: [[[Field; NK]; NJ]; NI],
    u_stage: [[[Field; NK]; NJ]; NI],
    wcon: [[[Field; NK]; NJ]; NI + 1],
    u_pos: [[[Field; NK]; NJ]; NI],
    utens: [[[Field; NK]; NJ]; NI],
    dtr_stage: Field,
) -> pub [[[Field; NK]; NJ]; NI] {
    // Input domain: every element read is a Q32 code in the window.
    for i in 0..NI {
        for j in 0..NJ {
            check_inputs(utens_stage[i][j]);
            check_inputs(u_stage[i][j]);
            check_inputs(u_pos[i][j]);
            check_inputs(utens[i][j]);
        }
    }
    for i in 0..NI + 1 {
        for j in 0..NJ {
            for k in 1..NK {
                check_input(wcon[i][j][k]);
            }
        }
    }
    check_input(dtr_stage);
    let mut ccol = [0; NI*NJ*NK];
    let mut dcol = [0; NI*NJ*NK];
    let mut data_col = [0; NI*NJ];
    let mut utens_stage = utens_stage;
    for k in 0..1 {
    // gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
    let mut t1 = [0; NI*NJ]; for p in 0..NI*NJ { t1[p] = (wcon[p/NJ+1][p%NJ][k + 1]) + (wcon[p/NJ][p%NJ][k + 1]); }
    let mut t2 = [0; NI*NJ]; for p in 0..NI*NJ { t2[p] = mul(1073741824,t1[p]); }
    let mut gcv = [0; NI*NJ]; for p in 0..NI*NJ { gcv[p] = t2[p]; }
    // cs = gcv * BET_M
    let mut t3 = [0; NI*NJ]; for p in 0..NI*NJ { t3[p] = mul(gcv[p],2147483648); }
    let mut cs = [0; NI*NJ]; for p in 0..NI*NJ { cs[p] = t3[p]; }
    // ccol[:, :, k] = gcv * BET_P
    let mut t4 = [0; NI*NJ]; for p in 0..NI*NJ { t4[p] = mul(gcv[p],2147483648); }
    for p in 0..NI*NJ { ccol[p*NK+(k)] = t4[p]; }
    // bcol = dtr_stage - ccol[:, :, k]
    let mut t5 = [0; NI*NJ]; for p in 0..NI*NJ { t5[p] = (dtr_stage) - (ccol[p*NK+(k)]); }
    let mut bcol = [0; NI*NJ]; for p in 0..NI*NJ { bcol[p] = t5[p]; }
    // correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
    let mut t6 = [0; NI*NJ]; for p in 0..NI*NJ { t6[p] = -(cs[p]); }
    let mut t7 = [0; NI*NJ]; for p in 0..NI*NJ { t7[p] = (u_stage[p/NJ][p%NJ][k + 1]) - (u_stage[p/NJ][p%NJ][k]); }
    let mut t8 = [0; NI*NJ]; for p in 0..NI*NJ { t8[p] = mul(t6[p],t7[p]); }
    let mut correction_term = [0; NI*NJ]; for p in 0..NI*NJ { correction_term[p] = t8[p]; }
    // dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
    let mut t9 = [0; NI*NJ]; for p in 0..NI*NJ { t9[p] = mul(dtr_stage,u_pos[p/NJ][p%NJ][k]); }
    let mut t10 = [0; NI*NJ]; for p in 0..NI*NJ { t10[p] = (t9[p]) + (utens[p/NJ][p%NJ][k]); }
    let mut t11 = [0; NI*NJ]; for p in 0..NI*NJ { t11[p] = (t10[p]) + (utens_stage[p/NJ][p%NJ][k]); }
    let mut t12 = [0; NI*NJ]; for p in 0..NI*NJ { t12[p] = (t11[p]) + (correction_term[p]); }
    for p in 0..NI*NJ { dcol[p*NK+(k)] = t12[p]; }
    // divided = 1.0 / bcol
    let mut t13 = [0; NI*NJ]; for p in 0..NI*NJ { t13[p] = canonical_div_nonzero(4294967296,bcol[p]); }
    let mut divided = [0; NI*NJ]; for p in 0..NI*NJ { divided[p] = t13[p]; }
    // ccol[:, :, k] = ccol[:, :, k] * divided
    let mut t14 = [0; NI*NJ]; for p in 0..NI*NJ { t14[p] = mul(ccol[p*NK+(k)],divided[p]); }
    for p in 0..NI*NJ { ccol[p*NK+(k)] = t14[p]; }
    // dcol[:, :, k] = dcol[:, :, k] * divided
    let mut t15 = [0; NI*NJ]; for p in 0..NI*NJ { t15[p] = mul(dcol[p*NK+(k)],divided[p]); }
    for p in 0..NI*NJ { dcol[p*NK+(k)] = t15[p]; }
    }
    for k in 1..NK-1 {
    // gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
    let mut t16 = [0; NI*NJ]; for p in 0..NI*NJ { t16[p] = (wcon[p/NJ+1][p%NJ][k]) + (wcon[p/NJ][p%NJ][k]); }
    let mut t17 = [0; NI*NJ]; for p in 0..NI*NJ { t17[p] = mul(-1073741824,t16[p]); }
    let mut gav = [0; NI*NJ]; for p in 0..NI*NJ { gav[p] = t17[p]; }
    // gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
    let mut t18 = [0; NI*NJ]; for p in 0..NI*NJ { t18[p] = (wcon[p/NJ+1][p%NJ][k + 1]) + (wcon[p/NJ][p%NJ][k + 1]); }
    let mut t19 = [0; NI*NJ]; for p in 0..NI*NJ { t19[p] = mul(1073741824,t18[p]); }
    let mut gcv = [0; NI*NJ]; for p in 0..NI*NJ { gcv[p] = t19[p]; }
    // as_ = gav * BET_M
    let mut t20 = [0; NI*NJ]; for p in 0..NI*NJ { t20[p] = mul(gav[p],2147483648); }
    let mut as_ = [0; NI*NJ]; for p in 0..NI*NJ { as_[p] = t20[p]; }
    // cs = gcv * BET_M
    let mut t21 = [0; NI*NJ]; for p in 0..NI*NJ { t21[p] = mul(gcv[p],2147483648); }
    let mut cs = [0; NI*NJ]; for p in 0..NI*NJ { cs[p] = t21[p]; }
    // acol = gav * BET_P
    let mut t22 = [0; NI*NJ]; for p in 0..NI*NJ { t22[p] = mul(gav[p],2147483648); }
    let mut acol = [0; NI*NJ]; for p in 0..NI*NJ { acol[p] = t22[p]; }
    // ccol[:, :, k] = gcv * BET_P
    let mut t23 = [0; NI*NJ]; for p in 0..NI*NJ { t23[p] = mul(gcv[p],2147483648); }
    for p in 0..NI*NJ { ccol[p*NK+(k)] = t23[p]; }
    // bcol = dtr_stage - acol - ccol[:, :, k]
    let mut t24 = [0; NI*NJ]; for p in 0..NI*NJ { t24[p] = (dtr_stage) - (acol[p]); }
    let mut t25 = [0; NI*NJ]; for p in 0..NI*NJ { t25[p] = (t24[p]) - (ccol[p*NK+(k)]); }
    let mut bcol = [0; NI*NJ]; for p in 0..NI*NJ { bcol[p] = t25[p]; }
    // correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
    let mut t26 = [0; NI*NJ]; for p in 0..NI*NJ { t26[p] = -(as_[p]); }
    let mut t27 = [0; NI*NJ]; for p in 0..NI*NJ { t27[p] = (u_stage[p/NJ][p%NJ][k - 1]) - (u_stage[p/NJ][p%NJ][k]); }
    let mut t28 = [0; NI*NJ]; for p in 0..NI*NJ { t28[p] = mul(t26[p],t27[p]); }
    let mut t29 = [0; NI*NJ]; for p in 0..NI*NJ { t29[p] = (u_stage[p/NJ][p%NJ][k + 1]) - (u_stage[p/NJ][p%NJ][k]); }
    let mut t30 = [0; NI*NJ]; for p in 0..NI*NJ { t30[p] = mul(cs[p],t29[p]); }
    let mut t31 = [0; NI*NJ]; for p in 0..NI*NJ { t31[p] = (t28[p]) - (t30[p]); }
    let mut correction_term = [0; NI*NJ]; for p in 0..NI*NJ { correction_term[p] = t31[p]; }
    // dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
    let mut t32 = [0; NI*NJ]; for p in 0..NI*NJ { t32[p] = mul(dtr_stage,u_pos[p/NJ][p%NJ][k]); }
    let mut t33 = [0; NI*NJ]; for p in 0..NI*NJ { t33[p] = (t32[p]) + (utens[p/NJ][p%NJ][k]); }
    let mut t34 = [0; NI*NJ]; for p in 0..NI*NJ { t34[p] = (t33[p]) + (utens_stage[p/NJ][p%NJ][k]); }
    let mut t35 = [0; NI*NJ]; for p in 0..NI*NJ { t35[p] = (t34[p]) + (correction_term[p]); }
    for p in 0..NI*NJ { dcol[p*NK+(k)] = t35[p]; }
    // divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
    let mut t36 = [0; NI*NJ]; for p in 0..NI*NJ { t36[p] = mul(ccol[p*NK+(k - 1)],acol[p]); }
    let mut t37 = [0; NI*NJ]; for p in 0..NI*NJ { t37[p] = (bcol[p]) - (t36[p]); }
    let mut t38 = [0; NI*NJ]; for p in 0..NI*NJ { t38[p] = canonical_div_nonzero(4294967296,t37[p]); }
    let mut divided = [0; NI*NJ]; for p in 0..NI*NJ { divided[p] = t38[p]; }
    // ccol[:, :, k] = ccol[:, :, k] * divided
    let mut t39 = [0; NI*NJ]; for p in 0..NI*NJ { t39[p] = mul(ccol[p*NK+(k)],divided[p]); }
    for p in 0..NI*NJ { ccol[p*NK+(k)] = t39[p]; }
    // dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
    let mut t40 = [0; NI*NJ]; for p in 0..NI*NJ { t40[p] = mul(dcol[p*NK+(k - 1)],acol[p]); }
    let mut t41 = [0; NI*NJ]; for p in 0..NI*NJ { t41[p] = (dcol[p*NK+(k)]) - (t40[p]); }
    let mut t42 = [0; NI*NJ]; for p in 0..NI*NJ { t42[p] = mul(t41[p],divided[p]); }
    for p in 0..NI*NJ { dcol[p*NK+(k)] = t42[p]; }
    }
    for k in NK-1..NK {
    // gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
    let mut t43 = [0; NI*NJ]; for p in 0..NI*NJ { t43[p] = (wcon[p/NJ+1][p%NJ][k]) + (wcon[p/NJ][p%NJ][k]); }
    let mut t44 = [0; NI*NJ]; for p in 0..NI*NJ { t44[p] = mul(-1073741824,t43[p]); }
    let mut gav = [0; NI*NJ]; for p in 0..NI*NJ { gav[p] = t44[p]; }
    // as_ = gav * BET_M
    let mut t45 = [0; NI*NJ]; for p in 0..NI*NJ { t45[p] = mul(gav[p],2147483648); }
    let mut as_ = [0; NI*NJ]; for p in 0..NI*NJ { as_[p] = t45[p]; }
    // acol = gav * BET_P
    let mut t46 = [0; NI*NJ]; for p in 0..NI*NJ { t46[p] = mul(gav[p],2147483648); }
    let mut acol = [0; NI*NJ]; for p in 0..NI*NJ { acol[p] = t46[p]; }
    // bcol = dtr_stage - acol
    let mut t47 = [0; NI*NJ]; for p in 0..NI*NJ { t47[p] = (dtr_stage) - (acol[p]); }
    let mut bcol = [0; NI*NJ]; for p in 0..NI*NJ { bcol[p] = t47[p]; }
    // correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
    let mut t48 = [0; NI*NJ]; for p in 0..NI*NJ { t48[p] = -(as_[p]); }
    let mut t49 = [0; NI*NJ]; for p in 0..NI*NJ { t49[p] = (u_stage[p/NJ][p%NJ][k - 1]) - (u_stage[p/NJ][p%NJ][k]); }
    let mut t50 = [0; NI*NJ]; for p in 0..NI*NJ { t50[p] = mul(t48[p],t49[p]); }
    let mut correction_term = [0; NI*NJ]; for p in 0..NI*NJ { correction_term[p] = t50[p]; }
    // dcol[:, :, k] = dtr_stage * u_pos[:, :, k] + utens[:, :, k] + utens_stage[:, :, k] + correction_term
    let mut t51 = [0; NI*NJ]; for p in 0..NI*NJ { t51[p] = mul(dtr_stage,u_pos[p/NJ][p%NJ][k]); }
    let mut t52 = [0; NI*NJ]; for p in 0..NI*NJ { t52[p] = (t51[p]) + (utens[p/NJ][p%NJ][k]); }
    let mut t53 = [0; NI*NJ]; for p in 0..NI*NJ { t53[p] = (t52[p]) + (utens_stage[p/NJ][p%NJ][k]); }
    let mut t54 = [0; NI*NJ]; for p in 0..NI*NJ { t54[p] = (t53[p]) + (correction_term[p]); }
    for p in 0..NI*NJ { dcol[p*NK+(k)] = t54[p]; }
    // divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
    let mut t55 = [0; NI*NJ]; for p in 0..NI*NJ { t55[p] = mul(ccol[p*NK+(k - 1)],acol[p]); }
    let mut t56 = [0; NI*NJ]; for p in 0..NI*NJ { t56[p] = (bcol[p]) - (t55[p]); }
    let mut t57 = [0; NI*NJ]; for p in 0..NI*NJ { t57[p] = canonical_div_nonzero(4294967296,t56[p]); }
    let mut divided = [0; NI*NJ]; for p in 0..NI*NJ { divided[p] = t57[p]; }
    // dcol[:, :, k] = (dcol[:, :, k] - dcol[:, :, k - 1] * acol) * divided
    let mut t58 = [0; NI*NJ]; for p in 0..NI*NJ { t58[p] = mul(dcol[p*NK+(k - 1)],acol[p]); }
    let mut t59 = [0; NI*NJ]; for p in 0..NI*NJ { t59[p] = (dcol[p*NK+(k)]) - (t58[p]); }
    let mut t60 = [0; NI*NJ]; for p in 0..NI*NJ { t60[p] = mul(t59[p],divided[p]); }
    for p in 0..NI*NJ { dcol[p*NK+(k)] = t60[p]; }
    }
    for t in 0..1 {
    let k = NK-1-t;
    // datacol = dcol[:, :, k]
    let mut datacol = [0; NI*NJ]; for p in 0..NI*NJ { datacol[p] = dcol[p*NK+(k)]; }
    // data_col[:] = datacol
    for p in 0..NI*NJ { data_col[p] = datacol[p]; }
    // utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
    let mut t61 = [0; NI*NJ]; for p in 0..NI*NJ { t61[p] = (datacol[p]) - (u_pos[p/NJ][p%NJ][k]); }
    let mut t62 = [0; NI*NJ]; for p in 0..NI*NJ { t62[p] = mul(dtr_stage,t61[p]); }
    for p in 0..NI*NJ { utens_stage[p/NJ][p%NJ][k] = t62[p]; }
    }
    for t in 0..NK-1 {
    let k = NK-2-t;
    // datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
    let mut t63 = [0; NI*NJ]; for p in 0..NI*NJ { t63[p] = mul(ccol[p*NK+(k)],data_col[p]); }
    let mut t64 = [0; NI*NJ]; for p in 0..NI*NJ { t64[p] = (dcol[p*NK+(k)]) - (t63[p]); }
    let mut datacol = [0; NI*NJ]; for p in 0..NI*NJ { datacol[p] = t64[p]; }
    // data_col[:] = datacol
    for p in 0..NI*NJ { data_col[p] = datacol[p]; }
    // utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])
    let mut t65 = [0; NI*NJ]; for p in 0..NI*NJ { t65[p] = (datacol[p]) - (u_pos[p/NJ][p%NJ][k]); }
    let mut t66 = [0; NI*NJ]; for p in 0..NI*NJ { t66[p] = mul(dtr_stage,t65[p]); }
    for p in 0..NI*NJ { utens_stage[p/NJ][p%NJ][k] = t66[p]; }
    }
    utens_stage
}
