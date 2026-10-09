// npbench_polybench_deriche_deriche__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Full slice arrays in source order; repeated exp calls remain repeated.
// 2**(-alpha) uses shared canonical exp with the Q32 log(2) constant.
// @outputs: float[R * C]
use noir_hw::{canonical_exp, check_input, check_inputs, div, mul};
global R: u32 = 16; // = imgIn_0
global C: u32 = 8; // = imgIn_1
fn main(alpha: Field, imgIn: [[Field; C]; R]) -> pub [[Field; C]; R] {
    check_input(alpha);
    for i in 0..R {
        check_inputs(imgIn[i]);
    }
    // k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    let t1 = -alpha;
    let t2 = canonical_exp(t1);
    let t3 = 4294967296 - t2;
    let t4 = -alpha;
    let t5 = canonical_exp(t4);
    let t6 = 4294967296 - t5;
    let t7 = mul(t3, t6);
    let t8 = -alpha;
    let t9 = canonical_exp(t8);
    let t10 = mul(alpha, t9);
    let t11 = 4294967296 + t10;
    let t12 = mul(8589934592, alpha);
    let t13 = canonical_exp(t12);
    let t14 = t11 - t13;
    let t15 = div(t7, t14);
    let k = t15;
    // a1 = a5 = k
    let a1 = k;
    let a5 = k;
    // a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    let t16 = -alpha;
    let t17 = canonical_exp(t16);
    let t18 = mul(k, t17);
    let t19 = alpha - 4294967296;
    let t20 = mul(t18, t19);
    let a2 = t20;
    let a6 = t20;
    // a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    let t21 = -alpha;
    let t22 = canonical_exp(t21);
    let t23 = mul(k, t22);
    let t24 = alpha + 4294967296;
    let t25 = mul(t23, t24);
    let a3 = t25;
    let a7 = t25;
    // a4 = a8 = -k * np.exp(-2.0 * alpha)
    let t26 = -k;
    let t27 = -8589934592;
    let t28 = mul(t27, alpha);
    let t29 = canonical_exp(t28);
    let t30 = mul(t26, t29);
    let a4 = t30;
    let a8 = t30;
    // b1 = 2.0 ** (-alpha)
    let t31 = -alpha;
    let t32 = mul(t31, 2977044472);
    let t33 = canonical_exp(t32);
    let b1 = t33;
    // b2 = -np.exp(-2.0 * alpha)
    let t34 = -8589934592;
    let t35 = mul(t34, alpha);
    let t36 = canonical_exp(t35);
    let t37 = -t36;
    let b2 = t37;
    // c1 = c2 = 1
    let c1 = 1;
    let c2 = 1;
    // y1 = np.empty_like(imgIn)
    let mut y1 = [[0; C]; R];
    // y1[:, 0] = a1 * imgIn[:, 0]
    let mut t38 = [0; R];
    for v in 0..R {
        t38[v] = imgIn[v][0];
    }
    let mut t39 = [0; R];
    for v in 0..R {
        t39[v] = mul(a1, t38[v]);
    }
    for v in 0..R {
        y1[v][0] = t39[v];
    }
    // y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    let mut t40 = [0; R];
    for v in 0..R {
        t40[v] = imgIn[v][1];
    }
    let mut t41 = [0; R];
    for v in 0..R {
        t41[v] = mul(a1, t40[v]);
    }
    let mut t42 = [0; R];
    for v in 0..R {
        t42[v] = imgIn[v][0];
    }
    let mut t43 = [0; R];
    for v in 0..R {
        t43[v] = mul(a2, t42[v]);
    }
    let mut t44 = [0; R];
    for v in 0..R {
        t44[v] = t41[v] + t43[v];
    }
    let mut t45 = [0; R];
    for v in 0..R {
        t45[v] = y1[v][0];
    }
    let mut t46 = [0; R];
    for v in 0..R {
        t46[v] = mul(b1, t45[v]);
    }
    let mut t47 = [0; R];
    for v in 0..R {
        t47[v] = t44[v] + t46[v];
    }
    for v in 0..R {
        y1[v][1] = t47[v];
    }
    // for j in range(2, imgIn.shape[1]):
    for j in 2..C {
        let mut t48 = [0; R];
        for v in 0..R {
            t48[v] = imgIn[v][j];
        }
        let mut t49 = [0; R];
        for v in 0..R {
            t49[v] = mul(a1, t48[v]);
        }
        let mut t50 = [0; R];
        for v in 0..R {
            t50[v] = imgIn[v][j - 1];
        }
        let mut t51 = [0; R];
        for v in 0..R {
            t51[v] = mul(a2, t50[v]);
        }
        let mut t52 = [0; R];
        for v in 0..R {
            t52[v] = t49[v] + t51[v];
        }
        let mut t53 = [0; R];
        for v in 0..R {
            t53[v] = y1[v][j - 1];
        }
        let mut t54 = [0; R];
        for v in 0..R {
            t54[v] = mul(b1, t53[v]);
        }
        let mut t55 = [0; R];
        for v in 0..R {
            t55[v] = t52[v] + t54[v];
        }
        let mut t56 = [0; R];
        for v in 0..R {
            t56[v] = y1[v][j - 2];
        }
        let mut t57 = [0; R];
        for v in 0..R {
            t57[v] = mul(b2, t56[v]);
        }
        let mut t58 = [0; R];
        for v in 0..R {
            t58[v] = t55[v] + t57[v];
        }
        for v in 0..R {
            y1[v][j] = t58[v];
        }
    }
    // y2 = np.empty_like(imgIn)
    let mut y2 = [[0; C]; R];
    // y2[:, -1] = 0.0
    for v in 0..R {
        y2[v][C - 1] = 0;
    }
    // y2[:, -2] = a3 * imgIn[:, -1]
    let mut t59 = [0; R];
    for v in 0..R {
        t59[v] = imgIn[v][C - 1];
    }
    let mut t60 = [0; R];
    for v in 0..R {
        t60[v] = mul(a3, t59[v]);
    }
    for v in 0..R {
        y2[v][C - 2] = t60[v];
    }
    // for j in range(imgIn.shape[1] - 3, -1, -1):
    for reverse in 0..C - 2 {
        let j = C - 3 - reverse;
        let mut t61 = [0; R];
        for v in 0..R {
            t61[v] = imgIn[v][j + 1];
        }
        let mut t62 = [0; R];
        for v in 0..R {
            t62[v] = mul(a3, t61[v]);
        }
        let mut t63 = [0; R];
        for v in 0..R {
            t63[v] = imgIn[v][j + 2];
        }
        let mut t64 = [0; R];
        for v in 0..R {
            t64[v] = mul(a4, t63[v]);
        }
        let mut t65 = [0; R];
        for v in 0..R {
            t65[v] = t62[v] + t64[v];
        }
        let mut t66 = [0; R];
        for v in 0..R {
            t66[v] = y2[v][j + 1];
        }
        let mut t67 = [0; R];
        for v in 0..R {
            t67[v] = mul(b1, t66[v]);
        }
        let mut t68 = [0; R];
        for v in 0..R {
            t68[v] = t65[v] + t67[v];
        }
        let mut t69 = [0; R];
        for v in 0..R {
            t69[v] = y2[v][j + 2];
        }
        let mut t70 = [0; R];
        for v in 0..R {
            t70[v] = mul(b2, t69[v]);
        }
        let mut t71 = [0; R];
        for v in 0..R {
            t71[v] = t68[v] + t70[v];
        }
        for v in 0..R {
            y2[v][j] = t71[v];
        }
    }
    // imgOut = c1 * (y1 + y2)
    let mut sum = [[0; C]; R];
    for i in 0..R {
        for j in 0..C {
            sum[i][j] = y1[i][j] + y2[i][j];
        }
    }
    let mut imgOut = [[0; C]; R];
    for i in 0..R {
        for j in 0..C {
            imgOut[i][j] = c1 * sum[i][j];
        }
    }
    // y1[0, :] = a5 * imgOut[0, :]
    let mut t72 = [0; C];
    for v in 0..C {
        t72[v] = imgOut[0][v];
    }
    let mut t73 = [0; C];
    for v in 0..C {
        t73[v] = mul(a5, t72[v]);
    }
    for v in 0..C {
        y1[0][v] = t73[v];
    }
    // y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    let mut t74 = [0; C];
    for v in 0..C {
        t74[v] = imgOut[1][v];
    }
    let mut t75 = [0; C];
    for v in 0..C {
        t75[v] = mul(a5, t74[v]);
    }
    let mut t76 = [0; C];
    for v in 0..C {
        t76[v] = imgOut[0][v];
    }
    let mut t77 = [0; C];
    for v in 0..C {
        t77[v] = mul(a6, t76[v]);
    }
    let mut t78 = [0; C];
    for v in 0..C {
        t78[v] = t75[v] + t77[v];
    }
    let mut t79 = [0; C];
    for v in 0..C {
        t79[v] = y1[0][v];
    }
    let mut t80 = [0; C];
    for v in 0..C {
        t80[v] = mul(b1, t79[v]);
    }
    let mut t81 = [0; C];
    for v in 0..C {
        t81[v] = t78[v] + t80[v];
    }
    for v in 0..C {
        y1[1][v] = t81[v];
    }
    // for i in range(2, imgIn.shape[0]):
    for i in 2..R {
        let mut t82 = [0; C];
        for v in 0..C {
            t82[v] = imgOut[i][v];
        }
        let mut t83 = [0; C];
        for v in 0..C {
            t83[v] = mul(a5, t82[v]);
        }
        let mut t84 = [0; C];
        for v in 0..C {
            t84[v] = imgOut[i - 1][v];
        }
        let mut t85 = [0; C];
        for v in 0..C {
            t85[v] = mul(a6, t84[v]);
        }
        let mut t86 = [0; C];
        for v in 0..C {
            t86[v] = t83[v] + t85[v];
        }
        let mut t87 = [0; C];
        for v in 0..C {
            t87[v] = y1[i - 1][v];
        }
        let mut t88 = [0; C];
        for v in 0..C {
            t88[v] = mul(b1, t87[v]);
        }
        let mut t89 = [0; C];
        for v in 0..C {
            t89[v] = t86[v] + t88[v];
        }
        let mut t90 = [0; C];
        for v in 0..C {
            t90[v] = y1[i - 2][v];
        }
        let mut t91 = [0; C];
        for v in 0..C {
            t91[v] = mul(b2, t90[v]);
        }
        let mut t92 = [0; C];
        for v in 0..C {
            t92[v] = t89[v] + t91[v];
        }
        for v in 0..C {
            y1[i][v] = t92[v];
        }
    }
    // y2[-1, :] = 0.0
    for v in 0..C {
        y2[R - 1][v] = 0;
    }
    // y2[-2, :] = a7 * imgOut[-1, :]
    let mut t93 = [0; C];
    for v in 0..C {
        t93[v] = imgOut[R - 1][v];
    }
    let mut t94 = [0; C];
    for v in 0..C {
        t94[v] = mul(a7, t93[v]);
    }
    for v in 0..C {
        y2[R - 2][v] = t94[v];
    }
    // for i in range(imgIn.shape[0] - 3, -1, -1):
    for reverse in 0..R - 2 {
        let i = R - 3 - reverse;
        let mut t95 = [0; C];
        for v in 0..C {
            t95[v] = imgOut[i + 1][v];
        }
        let mut t96 = [0; C];
        for v in 0..C {
            t96[v] = mul(a7, t95[v]);
        }
        let mut t97 = [0; C];
        for v in 0..C {
            t97[v] = imgOut[i + 2][v];
        }
        let mut t98 = [0; C];
        for v in 0..C {
            t98[v] = mul(a8, t97[v]);
        }
        let mut t99 = [0; C];
        for v in 0..C {
            t99[v] = t96[v] + t98[v];
        }
        let mut t100 = [0; C];
        for v in 0..C {
            t100[v] = y2[i + 1][v];
        }
        let mut t101 = [0; C];
        for v in 0..C {
            t101[v] = mul(b1, t100[v]);
        }
        let mut t102 = [0; C];
        for v in 0..C {
            t102[v] = t99[v] + t101[v];
        }
        let mut t103 = [0; C];
        for v in 0..C {
            t103[v] = y2[i + 2][v];
        }
        let mut t104 = [0; C];
        for v in 0..C {
            t104[v] = mul(b2, t103[v]);
        }
        let mut t105 = [0; C];
        for v in 0..C {
            t105[v] = t102[v] + t104[v];
        }
        for v in 0..C {
            y2[i][v] = t105[v];
        }
    }
    // imgOut[:] = c2 * (y1 + y2)
    let mut sum = [[0; C]; R];
    for i in 0..R {
        for j in 0..C {
            sum[i][j] = y1[i][j] + y2[i][j];
        }
    }
    for i in 0..R {
        for j in 0..C {
            imgOut[i][j] = c2 * sum[i][j];
        }
    }
    // return imgOut
    imgOut
}
