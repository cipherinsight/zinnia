// pythran_d2q9_nxnyns__one_time_step: literal translation of kernel.py (LITERAL_BASELINES.md).
// Helpers and all arithmetic follow source order, without shared products or strength reduction.
// Nine populations are fixed by the source; periodic copies use the original loops.
// Overwritten input borders/interiors use placeholders until assigned (same L7 input plumbing).
use crate::prelude::*;
const NX: usize = 16; // = f1_0
const NY: usize = 16; // = f1_1
const MX: usize = 16; // = f2_0
const MY: usize = 16; // = f2_1
fn m2f_loc(q: &Q32, ctx: &mut Ctx, m: &[Cell; 9], f: &mut [Cell; 9]) {
    // c0 = 1.0 / 6
    let c0 = Constant(fe(quantize(0.16666666666666666)));
    // c1 = 1.0 / 9
    let c1 = Constant(fe(quantize(0.1111111111111111)));
    // c2 = 1.0 / 18
    let c2 = Constant(fe(quantize(0.05555555555555555)));
    // c3 = 1.0 / 36
    let c3 = Constant(fe(quantize(0.027777777777777776)));
    // c4 = 1.0 / 12
    let c4 = Constant(fe(quantize(0.08333333333333333)));
    // f[0] = c1 * m[0] - c1 * m[3] + c1 * m[4]
    let t1 = q.mul(ctx, c1, m[0]).cell;
    let t2 = q.mul(ctx, c1, m[3]).cell;
    let t3 = q.sub(ctx, t1, t2);
    let t4 = q.mul(ctx, c1, m[4]).cell;
    let t5 = q.add(ctx, t3, t4);
    f[0] = t5;
    // f[1] = c1 * m[0] + c0 * m[1] - c3 * m[3] - c2 * m[4] - c0 * m[5] + 0.25 * m[7]
    let t6 = q.mul(ctx, c1, m[0]).cell;
    let t7 = q.mul(ctx, c0, m[1]).cell;
    let t8 = q.add(ctx, t6, t7);
    let t9 = q.mul(ctx, c3, m[3]).cell;
    let t10 = q.sub(ctx, t8, t9);
    let t11 = q.mul(ctx, c2, m[4]).cell;
    let t12 = q.sub(ctx, t10, t11);
    let t13 = q.mul(ctx, c0, m[5]).cell;
    let t14 = q.sub(ctx, t12, t13);
    let t15 = q.mul(ctx, Constant(fe(quantize(0.25))), m[7]).cell;
    let t16 = q.add(ctx, t14, t15);
    f[1] = t16;
    // f[2] = c1 * m[0] + c0 * m[2] - c3 * m[3] - c2 * m[4] - c0 * m[6] - 0.25 * m[7]
    let t17 = q.mul(ctx, c1, m[0]).cell;
    let t18 = q.mul(ctx, c0, m[2]).cell;
    let t19 = q.add(ctx, t17, t18);
    let t20 = q.mul(ctx, c3, m[3]).cell;
    let t21 = q.sub(ctx, t19, t20);
    let t22 = q.mul(ctx, c2, m[4]).cell;
    let t23 = q.sub(ctx, t21, t22);
    let t24 = q.mul(ctx, c0, m[6]).cell;
    let t25 = q.sub(ctx, t23, t24);
    let t26 = q.mul(ctx, Constant(fe(quantize(0.25))), m[7]).cell;
    let t27 = q.sub(ctx, t25, t26);
    f[2] = t27;
    // f[3] = c1 * m[0] - c0 * m[1] - c3 * m[3] - c2 * m[4] + c0 * m[5] + 0.25 * m[7]
    let t28 = q.mul(ctx, c1, m[0]).cell;
    let t29 = q.mul(ctx, c0, m[1]).cell;
    let t30 = q.sub(ctx, t28, t29);
    let t31 = q.mul(ctx, c3, m[3]).cell;
    let t32 = q.sub(ctx, t30, t31);
    let t33 = q.mul(ctx, c2, m[4]).cell;
    let t34 = q.sub(ctx, t32, t33);
    let t35 = q.mul(ctx, c0, m[5]).cell;
    let t36 = q.add(ctx, t34, t35);
    let t37 = q.mul(ctx, Constant(fe(quantize(0.25))), m[7]).cell;
    let t38 = q.add(ctx, t36, t37);
    f[3] = t38;
    // f[4] = c1 * m[0] - c0 * m[2] - c3 * m[3] - c2 * m[4] + c0 * m[6] - 0.25 * m[7]
    let t39 = q.mul(ctx, c1, m[0]).cell;
    let t40 = q.mul(ctx, c0, m[2]).cell;
    let t41 = q.sub(ctx, t39, t40);
    let t42 = q.mul(ctx, c3, m[3]).cell;
    let t43 = q.sub(ctx, t41, t42);
    let t44 = q.mul(ctx, c2, m[4]).cell;
    let t45 = q.sub(ctx, t43, t44);
    let t46 = q.mul(ctx, c0, m[6]).cell;
    let t47 = q.add(ctx, t45, t46);
    let t48 = q.mul(ctx, Constant(fe(quantize(0.25))), m[7]).cell;
    let t49 = q.sub(ctx, t47, t48);
    f[4] = t49;
    // f[5] = c1 * m[0] + c0 * m[1] + c0 * m[2] + c2 * m[3] + c3 * m[4] + c4 * m[5] + c4 * m[6] + 0.25 * m[8]
    let t50 = q.mul(ctx, c1, m[0]).cell;
    let t51 = q.mul(ctx, c0, m[1]).cell;
    let t52 = q.add(ctx, t50, t51);
    let t53 = q.mul(ctx, c0, m[2]).cell;
    let t54 = q.add(ctx, t52, t53);
    let t55 = q.mul(ctx, c2, m[3]).cell;
    let t56 = q.add(ctx, t54, t55);
    let t57 = q.mul(ctx, c3, m[4]).cell;
    let t58 = q.add(ctx, t56, t57);
    let t59 = q.mul(ctx, c4, m[5]).cell;
    let t60 = q.add(ctx, t58, t59);
    let t61 = q.mul(ctx, c4, m[6]).cell;
    let t62 = q.add(ctx, t60, t61);
    let t63 = q.mul(ctx, Constant(fe(quantize(0.25))), m[8]).cell;
    let t64 = q.add(ctx, t62, t63);
    f[5] = t64;
    // f[6] = c1 * m[0] - c0 * m[1] + c0 * m[2] + c2 * m[3] + c3 * m[4] - c4 * m[5] + c4 * m[6] - 0.25 * m[8]
    let t65 = q.mul(ctx, c1, m[0]).cell;
    let t66 = q.mul(ctx, c0, m[1]).cell;
    let t67 = q.sub(ctx, t65, t66);
    let t68 = q.mul(ctx, c0, m[2]).cell;
    let t69 = q.add(ctx, t67, t68);
    let t70 = q.mul(ctx, c2, m[3]).cell;
    let t71 = q.add(ctx, t69, t70);
    let t72 = q.mul(ctx, c3, m[4]).cell;
    let t73 = q.add(ctx, t71, t72);
    let t74 = q.mul(ctx, c4, m[5]).cell;
    let t75 = q.sub(ctx, t73, t74);
    let t76 = q.mul(ctx, c4, m[6]).cell;
    let t77 = q.add(ctx, t75, t76);
    let t78 = q.mul(ctx, Constant(fe(quantize(0.25))), m[8]).cell;
    let t79 = q.sub(ctx, t77, t78);
    f[6] = t79;
    // f[7] = c1 * m[0] - c0 * m[1] - c0 * m[2] + c2 * m[3] + c3 * m[4] - c4 * m[5] - c4 * m[6] + 0.25 * m[8]
    let t80 = q.mul(ctx, c1, m[0]).cell;
    let t81 = q.mul(ctx, c0, m[1]).cell;
    let t82 = q.sub(ctx, t80, t81);
    let t83 = q.mul(ctx, c0, m[2]).cell;
    let t84 = q.sub(ctx, t82, t83);
    let t85 = q.mul(ctx, c2, m[3]).cell;
    let t86 = q.add(ctx, t84, t85);
    let t87 = q.mul(ctx, c3, m[4]).cell;
    let t88 = q.add(ctx, t86, t87);
    let t89 = q.mul(ctx, c4, m[5]).cell;
    let t90 = q.sub(ctx, t88, t89);
    let t91 = q.mul(ctx, c4, m[6]).cell;
    let t92 = q.sub(ctx, t90, t91);
    let t93 = q.mul(ctx, Constant(fe(quantize(0.25))), m[8]).cell;
    let t94 = q.add(ctx, t92, t93);
    f[7] = t94;
    // f[8] = c1 * m[0] + c0 * m[1] - c0 * m[2] + c2 * m[3] + c3 * m[4] + c4 * m[5] - c4 * m[6] - 0.25 * m[8]
    let t95 = q.mul(ctx, c1, m[0]).cell;
    let t96 = q.mul(ctx, c0, m[1]).cell;
    let t97 = q.add(ctx, t95, t96);
    let t98 = q.mul(ctx, c0, m[2]).cell;
    let t99 = q.sub(ctx, t97, t98);
    let t100 = q.mul(ctx, c2, m[3]).cell;
    let t101 = q.add(ctx, t99, t100);
    let t102 = q.mul(ctx, c3, m[4]).cell;
    let t103 = q.add(ctx, t101, t102);
    let t104 = q.mul(ctx, c4, m[5]).cell;
    let t105 = q.add(ctx, t103, t104);
    let t106 = q.mul(ctx, c4, m[6]).cell;
    let t107 = q.sub(ctx, t105, t106);
    let t108 = q.mul(ctx, Constant(fe(quantize(0.25))), m[8]).cell;
    let t109 = q.sub(ctx, t107, t108);
    f[8] = t109;
}

fn f2m_loc(q: &Q32, ctx: &mut Ctx, f: &[Cell; 9], m: &mut [Cell; 9]) {
    // m[0] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
    let t1 = q.add(ctx, f[0], f[1]);
    let t2 = q.add(ctx, t1, f[2]);
    let t3 = q.add(ctx, t2, f[3]);
    let t4 = q.add(ctx, t3, f[4]);
    let t5 = q.add(ctx, t4, f[5]);
    let t6 = q.add(ctx, t5, f[6]);
    let t7 = q.add(ctx, t6, f[7]);
    let t8 = q.add(ctx, t7, f[8]);
    m[0] = t8;
    // m[1] = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
    let t9 = q.sub(ctx, f[1], f[3]);
    let t10 = q.add(ctx, t9, f[5]);
    let t11 = q.sub(ctx, t10, f[6]);
    let t12 = q.sub(ctx, t11, f[7]);
    let t13 = q.add(ctx, t12, f[8]);
    m[1] = t13;
    // m[2] = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
    let t14 = q.sub(ctx, f[2], f[4]);
    let t15 = q.add(ctx, t14, f[5]);
    let t16 = q.add(ctx, t15, f[6]);
    let t17 = q.sub(ctx, t16, f[7]);
    let t18 = q.sub(ctx, t17, f[8]);
    m[2] = t18;
    // m[3] = -4.0 * f[0] - f[1] - f[2] - f[3] - f[4] + 2.0 * f[5] + 2.0 * f[6] + 2.0 * f[7] + 2.0 * f[8]
    let t19 = q.neg(ctx, Constant(fe(quantize(4.0))));
    let t20 = q.mul(ctx, t19, f[0]).cell;
    let t21 = q.sub(ctx, t20, f[1]);
    let t22 = q.sub(ctx, t21, f[2]);
    let t23 = q.sub(ctx, t22, f[3]);
    let t24 = q.sub(ctx, t23, f[4]);
    let t25 = q.mul(ctx, Constant(fe(quantize(2.0))), f[5]).cell;
    let t26 = q.add(ctx, t24, t25);
    let t27 = q.mul(ctx, Constant(fe(quantize(2.0))), f[6]).cell;
    let t28 = q.add(ctx, t26, t27);
    let t29 = q.mul(ctx, Constant(fe(quantize(2.0))), f[7]).cell;
    let t30 = q.add(ctx, t28, t29);
    let t31 = q.mul(ctx, Constant(fe(quantize(2.0))), f[8]).cell;
    let t32 = q.add(ctx, t30, t31);
    m[3] = t32;
    // m[4] = 4.0 * f[0] - 2.0 * f[1] - 2.0 * f[2] - 2.0 * f[3] - 2.0 * f[4] + f[5] + f[6] + f[7] + f[8]
    let t33 = q.mul(ctx, Constant(fe(quantize(4.0))), f[0]).cell;
    let t34 = q.mul(ctx, Constant(fe(quantize(2.0))), f[1]).cell;
    let t35 = q.sub(ctx, t33, t34);
    let t36 = q.mul(ctx, Constant(fe(quantize(2.0))), f[2]).cell;
    let t37 = q.sub(ctx, t35, t36);
    let t38 = q.mul(ctx, Constant(fe(quantize(2.0))), f[3]).cell;
    let t39 = q.sub(ctx, t37, t38);
    let t40 = q.mul(ctx, Constant(fe(quantize(2.0))), f[4]).cell;
    let t41 = q.sub(ctx, t39, t40);
    let t42 = q.add(ctx, t41, f[5]);
    let t43 = q.add(ctx, t42, f[6]);
    let t44 = q.add(ctx, t43, f[7]);
    let t45 = q.add(ctx, t44, f[8]);
    m[4] = t45;
    // m[5] = -2.0 * f[1] + 2.0 * f[3] + f[5] - f[6] - f[7] + f[8]
    let t46 = q.neg(ctx, Constant(fe(quantize(2.0))));
    let t47 = q.mul(ctx, t46, f[1]).cell;
    let t48 = q.mul(ctx, Constant(fe(quantize(2.0))), f[3]).cell;
    let t49 = q.add(ctx, t47, t48);
    let t50 = q.add(ctx, t49, f[5]);
    let t51 = q.sub(ctx, t50, f[6]);
    let t52 = q.sub(ctx, t51, f[7]);
    let t53 = q.add(ctx, t52, f[8]);
    m[5] = t53;
    // m[6] = -2.0 * f[2] + 2.0 * f[4] + f[5] + f[6] - f[7] - f[8]
    let t54 = q.neg(ctx, Constant(fe(quantize(2.0))));
    let t55 = q.mul(ctx, t54, f[2]).cell;
    let t56 = q.mul(ctx, Constant(fe(quantize(2.0))), f[4]).cell;
    let t57 = q.add(ctx, t55, t56);
    let t58 = q.add(ctx, t57, f[5]);
    let t59 = q.add(ctx, t58, f[6]);
    let t60 = q.sub(ctx, t59, f[7]);
    let t61 = q.sub(ctx, t60, f[8]);
    m[6] = t61;
    // m[7] = f[1] - f[2] + f[3] - f[4]
    let t62 = q.sub(ctx, f[1], f[2]);
    let t63 = q.add(ctx, t62, f[3]);
    let t64 = q.sub(ctx, t63, f[4]);
    m[7] = t64;
    // m[8] = f[5] - f[6] + f[7] - f[8]
    let t65 = q.sub(ctx, f[5], f[6]);
    let t66 = q.add(ctx, t65, f[7]);
    let t67 = q.sub(ctx, t66, f[8]);
    m[8] = t67;
}

fn relaxation_loc(q: &Q32, ctx: &mut Ctx, m: &mut [Cell; 9]) {
    // m[3] += 1.1312217194570136 * (-2 * m[0] + 3.0 * m[1] * m[1] + 3.0 * m[2] * m[2] - m[3])
    let t2 = q.mul_int(ctx, m[0], Constant(fe(-2)));
    let t3 = q.mul(ctx, Constant(fe(quantize(3.0))), m[1]).cell;
    let t4 = q.mul(ctx, t3, m[1]).cell;
    let t5 = q.add(ctx, t2, t4);
    let t6 = q.mul(ctx, Constant(fe(quantize(3.0))), m[2]).cell;
    let t7 = q.mul(ctx, t6, m[2]).cell;
    let t8 = q.add(ctx, t5, t7);
    let t9 = q.sub(ctx, t8, m[3]);
    let t10 = q
        .mul(ctx, Constant(fe(quantize(1.1312217194570136))), t9)
        .cell;
    m[3] = q.add(ctx, m[3], t10);
    // m[4] += 1.1312217194570136 * (m[0] + 1.5 * m[1] * m[1] + 1.5 * m[2] * m[2] - m[4])
    let t11 = q.mul(ctx, Constant(fe(quantize(1.5))), m[1]).cell;
    let t12 = q.mul(ctx, t11, m[1]).cell;
    let t13 = q.add(ctx, m[0], t12);
    let t14 = q.mul(ctx, Constant(fe(quantize(1.5))), m[2]).cell;
    let t15 = q.mul(ctx, t14, m[2]).cell;
    let t16 = q.add(ctx, t13, t15);
    let t17 = q.sub(ctx, t16, m[4]);
    let t18 = q
        .mul(ctx, Constant(fe(quantize(1.1312217194570136))), t17)
        .cell;
    m[4] = q.add(ctx, m[4], t18);
    // m[5] += 1.1312217194570136 * (-m[1] - m[5])
    let t19 = q.neg(ctx, m[1]);
    let t20 = q.sub(ctx, t19, m[5]);
    let t21 = q
        .mul(ctx, Constant(fe(quantize(1.1312217194570136))), t20)
        .cell;
    m[5] = q.add(ctx, m[5], t21);
    // m[6] += 1.1312217194570136 * (-m[2] - m[6])
    let t22 = q.neg(ctx, m[2]);
    let t23 = q.sub(ctx, t22, m[6]);
    let t24 = q
        .mul(ctx, Constant(fe(quantize(1.1312217194570136))), t23)
        .cell;
    m[6] = q.add(ctx, m[6], t24);
    // m[7] += 1.8573551263001487 * (m[1] * m[1] - m[2] * m[2] - m[7])
    let t25 = q.mul(ctx, m[1], m[1]).cell;
    let t26 = q.mul(ctx, m[2], m[2]).cell;
    let t27 = q.sub(ctx, t25, t26);
    let t28 = q.sub(ctx, t27, m[7]);
    let t29 = q
        .mul(ctx, Constant(fe(quantize(1.8573551263001487))), t28)
        .cell;
    m[7] = q.add(ctx, m[7], t29);
    // m[8] += 1.8573551263001487 * (m[1] * m[2] - m[8])
    let t30 = q.mul(ctx, m[1], m[2]).cell;
    let t31 = q.sub(ctx, t30, m[8]);
    let t32 = q
        .mul(ctx, Constant(fe(quantize(1.8573551263001487))), t31)
        .cell;
    m[8] = q.add(ctx, m[8], t32);
}
fn getf(f: &[Cell], floc: &mut [Cell; 9], i: usize, j: usize) {
    floc[0] = f[(((i) * NY + (j)) * 9) + 0];
    floc[1] = f[(((i - 1) * NY + (j)) * 9) + 1];
    floc[2] = f[(((i) * NY + (j - 1)) * 9) + 2];
    floc[3] = f[(((i + 1) * NY + (j)) * 9) + 3];
    floc[4] = f[(((i) * NY + (j + 1)) * 9) + 4];
    floc[5] = f[(((i - 1) * NY + (j - 1)) * 9) + 5];
    floc[6] = f[(((i + 1) * NY + (j - 1)) * 9) + 6];
    floc[7] = f[(((i + 1) * NY + (j + 1)) * 9) + 7];
    floc[8] = f[(((i - 1) * NY + (j + 1)) * 9) + 8];
}
fn setf(f: &mut [Cell], floc: &[Cell; 9], i: usize, j: usize) {
    f[(j * MY + i) * 9 + 0] = floc[0];
    f[(j * MY + i) * 9 + 1] = floc[1];
    f[(j * MY + i) * 9 + 2] = floc[2];
    f[(j * MY + i) * 9 + 3] = floc[3];
    f[(j * MY + i) * 9 + 4] = floc[4];
    f[(j * MY + i) * 9 + 5] = floc[5];
    f[(j * MY + i) * 9 + 6] = floc[6];
    f[(j * MY + i) * 9 + 7] = floc[7];
    f[(j * MY + i) * 9 + 8] = floc[8];
}
fn periodic_bc(f: &mut [Cell]) {
    for j in 0..NY {
        for k in 0..9 {
            f[j * 9 + k] = f[((NX - 2) * NY + j) * 9 + k];
            f[((NX - 1) * NY + j) * 9 + k] = f[(NY + j) * 9 + k];
        }
    }
    for i in 0..NX {
        for k in 0..9 {
            f[i * NY * 9 + k] = f[(i * NY + NY - 2) * 9 + k];
            f[(i * NY + NY - 1) * 9 + k] = f[(i * NY + 1) * 9 + k];
        }
    }
}
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input1 = inputs.floats("f1", &[NX, NY, 9])?;
    let input2 = inputs.floats("f2", &[MX, MY, 9])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let zero = q.float_const(ctx, 0.0);
    let mut f1 = vec![zero; NX * NY * 9];
    let mut f2 = vec![zero; MX * MY * 9];
    for i in 1..NX - 1 {
        for j in 1..NY - 1 {
            for k in 0..9 {
                let at = (i * NY + j) * 9 + k;
                f1[at] = q.load_float(ctx, input1[at]);
            }
        }
    }
    for j in 0..MX {
        for i in 0..MY {
            if !(j > 0 && j < NY - 1 && i > 0 && i < NX - 1) {
                for k in 0..9 {
                    let at = (j * MY + i) * 9 + k;
                    f2[at] = q.load_float(ctx, input2[at]);
                }
            }
        }
    }
    let mut floc = [zero; 9];
    let mut mloc = [zero; 9];
    periodic_bc(&mut f1);
    for i in 1..NX - 1 {
        for j in 1..NY - 1 {
            getf(&f1, &mut floc, i, j);
            f2m_loc(&q, ctx, &floc, &mut mloc);
            relaxation_loc(&q, ctx, &mut mloc);
            m2f_loc(&q, ctx, &mloc, &mut floc);
            setf(&mut f2, &floc, i, j);
        }
    }
    out.floats(&f2);
    Ok(())
}
