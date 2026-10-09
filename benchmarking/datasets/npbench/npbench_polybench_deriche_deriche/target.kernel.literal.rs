//! npbench_polybench_deriche_deriche__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Full slice arrays in source order; repeated exp calls remain repeated.
//! 2**(-alpha) uses shared canonical exp with the Q32 log(2) constant.
use crate::prelude::*;
const R: usize = 16; // = imgIn_0
const C: usize = 8; // = imgIn_1
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let img = inputs.floats("imgIn", &[R, C])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let alpha = q.load_float(ctx, alpha);
    let imgIn = q.load_floats(ctx, &img);
    // k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    let t1 = q.neg(ctx, alpha);
    let t3 = q.bound(ctx, t1);
    let t2 = q.canonical_exp(ctx, t3).cell;
    let t4 = q.sub(ctx, Constant(fe(quantize(1.0))), t2);
    let t5 = q.neg(ctx, alpha);
    let t7 = q.bound(ctx, t5);
    let t6 = q.canonical_exp(ctx, t7).cell;
    let t8 = q.sub(ctx, Constant(fe(quantize(1.0))), t6);
    let t9 = q.mul(ctx, t4, t8).cell;
    let t10 = q.neg(ctx, alpha);
    let t12 = q.bound(ctx, t10);
    let t11 = q.canonical_exp(ctx, t12).cell;
    let t13 = q.mul(ctx, alpha, t11).cell;
    let t14 = q.add(ctx, Constant(fe(quantize(1.0))), t13);
    let t15 = q.mul(ctx, Constant(fe(quantize(2.0))), alpha).cell;
    let t17 = q.bound(ctx, t15);
    let t16 = q.canonical_exp(ctx, t17).cell;
    let t18 = q.sub(ctx, t14, t16);
    let t20 = q.bound(ctx, t18);
    let t19 = q.div(ctx, t9, t20).cell;
    let k = t19;
    // a1 = a5 = k
    let a1 = k;
    let a5 = k;
    // a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    let t21 = q.neg(ctx, alpha);
    let t23 = q.bound(ctx, t21);
    let t22 = q.canonical_exp(ctx, t23).cell;
    let t24 = q.mul(ctx, k, t22).cell;
    let t25 = q.sub(ctx, alpha, Constant(fe(quantize(1.0))));
    let t26 = q.mul(ctx, t24, t25).cell;
    let a2 = t26;
    let a6 = t26;
    // a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    let t27 = q.neg(ctx, alpha);
    let t29 = q.bound(ctx, t27);
    let t28 = q.canonical_exp(ctx, t29).cell;
    let t30 = q.mul(ctx, k, t28).cell;
    let t31 = q.add(ctx, alpha, Constant(fe(quantize(1.0))));
    let t32 = q.mul(ctx, t30, t31).cell;
    let a3 = t32;
    let a7 = t32;
    // a4 = a8 = -k * np.exp(-2.0 * alpha)
    let t33 = q.neg(ctx, k);
    let t34 = q.neg(ctx, Constant(fe(quantize(2.0))));
    let t35 = q.mul(ctx, t34, alpha).cell;
    let t37 = q.bound(ctx, t35);
    let t36 = q.canonical_exp(ctx, t37).cell;
    let t38 = q.mul(ctx, t33, t36).cell;
    let a4 = t38;
    let a8 = t38;
    // b1 = 2.0 ** (-alpha)
    let t39 = q.neg(ctx, alpha);
    let t40 = q
        .mul(ctx, t39, Constant(fe(quantize(0.6931471805599453))))
        .cell;
    let t42 = q.bound(ctx, t40);
    let t41 = q.canonical_exp(ctx, t42).cell;
    let b1 = t41;
    // b2 = -np.exp(-2.0 * alpha)
    let t43 = q.neg(ctx, Constant(fe(quantize(2.0))));
    let t44 = q.mul(ctx, t43, alpha).cell;
    let t46 = q.bound(ctx, t44);
    let t45 = q.canonical_exp(ctx, t46).cell;
    let t47 = q.neg(ctx, t45);
    let b2 = t47;
    // c1 = c2 = 1
    let t48 = q.int_const(ctx, 1);
    let c1 = t48;
    let c2 = t48;
    // y1 = np.empty_like(imgIn)
    let mut y1 = vec![q.float_const(ctx, 0.0); R * C];
    // y1[:, 0] = a1 * imgIn[:, 0]
    let t49: Vec<Cell> = (0..R).map(|v| imgIn[v * C + (0)]).collect();
    let t50: Vec<Cell> = (0..R).map(|v| q.mul(ctx, a1, t49[v]).cell).collect();
    for v in 0..R {
        y1[v * C + (0)] = t50[v];
    }
    // y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    let t51: Vec<Cell> = (0..R).map(|v| imgIn[v * C + (1)]).collect();
    let t52: Vec<Cell> = (0..R).map(|v| q.mul(ctx, a1, t51[v]).cell).collect();
    let t53: Vec<Cell> = (0..R).map(|v| imgIn[v * C + (0)]).collect();
    let t54: Vec<Cell> = (0..R).map(|v| q.mul(ctx, a2, t53[v]).cell).collect();
    let t55: Vec<Cell> = (0..R).map(|v| q.add(ctx, t52[v], t54[v])).collect();
    let t56: Vec<Cell> = (0..R).map(|v| y1[v * C + (0)]).collect();
    let t57: Vec<Cell> = (0..R).map(|v| q.mul(ctx, b1, t56[v]).cell).collect();
    let t58: Vec<Cell> = (0..R).map(|v| q.add(ctx, t55[v], t57[v])).collect();
    for v in 0..R {
        y1[v * C + (1)] = t58[v];
    }
    // for j in range(2, imgIn.shape[1]):
    for j in 2..C {
        let t59: Vec<Cell> = (0..R).map(|v| imgIn[v * C + (j)]).collect();
        let t60: Vec<Cell> = (0..R).map(|v| q.mul(ctx, a1, t59[v]).cell).collect();
        let t61: Vec<Cell> = (0..R).map(|v| imgIn[v * C + (j - 1)]).collect();
        let t62: Vec<Cell> = (0..R).map(|v| q.mul(ctx, a2, t61[v]).cell).collect();
        let t63: Vec<Cell> = (0..R).map(|v| q.add(ctx, t60[v], t62[v])).collect();
        let t64: Vec<Cell> = (0..R).map(|v| y1[v * C + (j - 1)]).collect();
        let t65: Vec<Cell> = (0..R).map(|v| q.mul(ctx, b1, t64[v]).cell).collect();
        let t66: Vec<Cell> = (0..R).map(|v| q.add(ctx, t63[v], t65[v])).collect();
        let t67: Vec<Cell> = (0..R).map(|v| y1[v * C + (j - 2)]).collect();
        let t68: Vec<Cell> = (0..R).map(|v| q.mul(ctx, b2, t67[v]).cell).collect();
        let t69: Vec<Cell> = (0..R).map(|v| q.add(ctx, t66[v], t68[v])).collect();
        for v in 0..R {
            y1[v * C + (j)] = t69[v];
        }
    }
    // y2 = np.empty_like(imgIn)
    let mut y2 = vec![q.float_const(ctx, 0.0); R * C];
    // y2[:, -1] = 0.0
    let t70 = q.float_const(ctx, 0.0);
    for v in 0..R {
        y2[v * C + (C - 1)] = t70;
    }
    // y2[:, -2] = a3 * imgIn[:, -1]
    let t71: Vec<Cell> = (0..R).map(|v| imgIn[v * C + (C - 1)]).collect();
    let t72: Vec<Cell> = (0..R).map(|v| q.mul(ctx, a3, t71[v]).cell).collect();
    for v in 0..R {
        y2[v * C + (C - 2)] = t72[v];
    }
    // for j in range(imgIn.shape[1] - 3, -1, -1):
    for reverse in 0..C - 2 {
        let j = C - 3 - reverse;
        let t73: Vec<Cell> = (0..R).map(|v| imgIn[v * C + (j + 1)]).collect();
        let t74: Vec<Cell> = (0..R).map(|v| q.mul(ctx, a3, t73[v]).cell).collect();
        let t75: Vec<Cell> = (0..R).map(|v| imgIn[v * C + (j + 2)]).collect();
        let t76: Vec<Cell> = (0..R).map(|v| q.mul(ctx, a4, t75[v]).cell).collect();
        let t77: Vec<Cell> = (0..R).map(|v| q.add(ctx, t74[v], t76[v])).collect();
        let t78: Vec<Cell> = (0..R).map(|v| y2[v * C + (j + 1)]).collect();
        let t79: Vec<Cell> = (0..R).map(|v| q.mul(ctx, b1, t78[v]).cell).collect();
        let t80: Vec<Cell> = (0..R).map(|v| q.add(ctx, t77[v], t79[v])).collect();
        let t81: Vec<Cell> = (0..R).map(|v| y2[v * C + (j + 2)]).collect();
        let t82: Vec<Cell> = (0..R).map(|v| q.mul(ctx, b2, t81[v]).cell).collect();
        let t83: Vec<Cell> = (0..R).map(|v| q.add(ctx, t80[v], t82[v])).collect();
        for v in 0..R {
            y2[v * C + (j)] = t83[v];
        }
    }
    // imgOut = c1 * (y1 + y2)
    let sum: Vec<Cell> = (0..R * C).map(|v| q.add(ctx, y1[v], y2[v])).collect();
    let mut imgOut: Vec<Cell> = sum.iter().map(|v| q.mul_int(ctx, c1, *v)).collect();
    // y1[0, :] = a5 * imgOut[0, :]
    let t84: Vec<Cell> = (0..C).map(|v| imgOut[(0) * C + v]).collect();
    let t85: Vec<Cell> = (0..C).map(|v| q.mul(ctx, a5, t84[v]).cell).collect();
    for v in 0..C {
        y1[(0) * C + v] = t85[v];
    }
    // y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    let t86: Vec<Cell> = (0..C).map(|v| imgOut[(1) * C + v]).collect();
    let t87: Vec<Cell> = (0..C).map(|v| q.mul(ctx, a5, t86[v]).cell).collect();
    let t88: Vec<Cell> = (0..C).map(|v| imgOut[(0) * C + v]).collect();
    let t89: Vec<Cell> = (0..C).map(|v| q.mul(ctx, a6, t88[v]).cell).collect();
    let t90: Vec<Cell> = (0..C).map(|v| q.add(ctx, t87[v], t89[v])).collect();
    let t91: Vec<Cell> = (0..C).map(|v| y1[(0) * C + v]).collect();
    let t92: Vec<Cell> = (0..C).map(|v| q.mul(ctx, b1, t91[v]).cell).collect();
    let t93: Vec<Cell> = (0..C).map(|v| q.add(ctx, t90[v], t92[v])).collect();
    for v in 0..C {
        y1[(1) * C + v] = t93[v];
    }
    // for i in range(2, imgIn.shape[0]):
    for i in 2..R {
        let t94: Vec<Cell> = (0..C).map(|v| imgOut[(i) * C + v]).collect();
        let t95: Vec<Cell> = (0..C).map(|v| q.mul(ctx, a5, t94[v]).cell).collect();
        let t96: Vec<Cell> = (0..C).map(|v| imgOut[(i - 1) * C + v]).collect();
        let t97: Vec<Cell> = (0..C).map(|v| q.mul(ctx, a6, t96[v]).cell).collect();
        let t98: Vec<Cell> = (0..C).map(|v| q.add(ctx, t95[v], t97[v])).collect();
        let t99: Vec<Cell> = (0..C).map(|v| y1[(i - 1) * C + v]).collect();
        let t100: Vec<Cell> = (0..C).map(|v| q.mul(ctx, b1, t99[v]).cell).collect();
        let t101: Vec<Cell> = (0..C).map(|v| q.add(ctx, t98[v], t100[v])).collect();
        let t102: Vec<Cell> = (0..C).map(|v| y1[(i - 2) * C + v]).collect();
        let t103: Vec<Cell> = (0..C).map(|v| q.mul(ctx, b2, t102[v]).cell).collect();
        let t104: Vec<Cell> = (0..C).map(|v| q.add(ctx, t101[v], t103[v])).collect();
        for v in 0..C {
            y1[(i) * C + v] = t104[v];
        }
    }
    // y2[-1, :] = 0.0
    let t105 = q.float_const(ctx, 0.0);
    for v in 0..C {
        y2[(R - 1) * C + v] = t105;
    }
    // y2[-2, :] = a7 * imgOut[-1, :]
    let t106: Vec<Cell> = (0..C).map(|v| imgOut[(R - 1) * C + v]).collect();
    let t107: Vec<Cell> = (0..C).map(|v| q.mul(ctx, a7, t106[v]).cell).collect();
    for v in 0..C {
        y2[(R - 2) * C + v] = t107[v];
    }
    // for i in range(imgIn.shape[0] - 3, -1, -1):
    for reverse in 0..R - 2 {
        let i = R - 3 - reverse;
        let t108: Vec<Cell> = (0..C).map(|v| imgOut[(i + 1) * C + v]).collect();
        let t109: Vec<Cell> = (0..C).map(|v| q.mul(ctx, a7, t108[v]).cell).collect();
        let t110: Vec<Cell> = (0..C).map(|v| imgOut[(i + 2) * C + v]).collect();
        let t111: Vec<Cell> = (0..C).map(|v| q.mul(ctx, a8, t110[v]).cell).collect();
        let t112: Vec<Cell> = (0..C).map(|v| q.add(ctx, t109[v], t111[v])).collect();
        let t113: Vec<Cell> = (0..C).map(|v| y2[(i + 1) * C + v]).collect();
        let t114: Vec<Cell> = (0..C).map(|v| q.mul(ctx, b1, t113[v]).cell).collect();
        let t115: Vec<Cell> = (0..C).map(|v| q.add(ctx, t112[v], t114[v])).collect();
        let t116: Vec<Cell> = (0..C).map(|v| y2[(i + 2) * C + v]).collect();
        let t117: Vec<Cell> = (0..C).map(|v| q.mul(ctx, b2, t116[v]).cell).collect();
        let t118: Vec<Cell> = (0..C).map(|v| q.add(ctx, t115[v], t117[v])).collect();
        for v in 0..C {
            y2[(i) * C + v] = t118[v];
        }
    }
    // imgOut[:] = c2 * (y1 + y2)
    let sum: Vec<Cell> = (0..R * C).map(|v| q.add(ctx, y1[v], y2[v])).collect();
    imgOut = sum.iter().map(|v| q.mul_int(ctx, c2, *v)).collect();
    // return imgOut
    out.floats(&imgOut);
    Ok(())
}
