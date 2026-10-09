// pythran_d2q9_nxnyns__one_time_step: literal translation of kernel.py (LITERAL_BASELINES.md).
// Helpers and all arithmetic follow source order, without shared products or strength reduction.
// The first nine population assignments are fixed by the source; shape loops use NS/MS.
// Overwritten input borders/interiors use placeholders until assigned (same L7 input plumbing).
// @outputs: float[MX * MY * MS]
global NX: u32 = 16; // = f1_0
global NY: u32 = 16; // = f1_1
global MX: u32 = 16; // = f2_0
global MY: u32 = 16; // = f2_1
global NS: u32 = 9; // = f1_2
global MS: u32 = 9; // = f2_2
fn m2f_loc(m: [Field; NS], f: [Field; NS]) -> [Field; NS] {
    let mut f = f;
    // c0 = 1.0 / 6
    let c0 = 715827883;
    // c1 = 1.0 / 9
    let c1 = 477218588;
    // c2 = 1.0 / 18
    let c2 = 238609294;
    // c3 = 1.0 / 36
    let c3 = 119304647;
    // c4 = 1.0 / 12
    let c4 = 357913941;
    // f[0] = c1 * m[0] - c1 * m[3] + c1 * m[4]
    let t1 = noir_hw::mul(c1, m[0]);
    let t2 = noir_hw::mul(c1, m[3]);
    let t3 = (t1) - (t2);
    let t4 = noir_hw::mul(c1, m[4]);
    let t5 = (t3) + (t4);
    f[0] = t5;
    // f[1] = c1 * m[0] + c0 * m[1] - c3 * m[3] - c2 * m[4] - c0 * m[5] + 0.25 * m[7]
    let t6 = noir_hw::mul(c1, m[0]);
    let t7 = noir_hw::mul(c0, m[1]);
    let t8 = (t6) + (t7);
    let t9 = noir_hw::mul(c3, m[3]);
    let t10 = (t8) - (t9);
    let t11 = noir_hw::mul(c2, m[4]);
    let t12 = (t10) - (t11);
    let t13 = noir_hw::mul(c0, m[5]);
    let t14 = (t12) - (t13);
    let t15 = noir_hw::mul(1073741824, m[7]);
    let t16 = (t14) + (t15);
    f[1] = t16;
    // f[2] = c1 * m[0] + c0 * m[2] - c3 * m[3] - c2 * m[4] - c0 * m[6] - 0.25 * m[7]
    let t17 = noir_hw::mul(c1, m[0]);
    let t18 = noir_hw::mul(c0, m[2]);
    let t19 = (t17) + (t18);
    let t20 = noir_hw::mul(c3, m[3]);
    let t21 = (t19) - (t20);
    let t22 = noir_hw::mul(c2, m[4]);
    let t23 = (t21) - (t22);
    let t24 = noir_hw::mul(c0, m[6]);
    let t25 = (t23) - (t24);
    let t26 = noir_hw::mul(1073741824, m[7]);
    let t27 = (t25) - (t26);
    f[2] = t27;
    // f[3] = c1 * m[0] - c0 * m[1] - c3 * m[3] - c2 * m[4] + c0 * m[5] + 0.25 * m[7]
    let t28 = noir_hw::mul(c1, m[0]);
    let t29 = noir_hw::mul(c0, m[1]);
    let t30 = (t28) - (t29);
    let t31 = noir_hw::mul(c3, m[3]);
    let t32 = (t30) - (t31);
    let t33 = noir_hw::mul(c2, m[4]);
    let t34 = (t32) - (t33);
    let t35 = noir_hw::mul(c0, m[5]);
    let t36 = (t34) + (t35);
    let t37 = noir_hw::mul(1073741824, m[7]);
    let t38 = (t36) + (t37);
    f[3] = t38;
    // f[4] = c1 * m[0] - c0 * m[2] - c3 * m[3] - c2 * m[4] + c0 * m[6] - 0.25 * m[7]
    let t39 = noir_hw::mul(c1, m[0]);
    let t40 = noir_hw::mul(c0, m[2]);
    let t41 = (t39) - (t40);
    let t42 = noir_hw::mul(c3, m[3]);
    let t43 = (t41) - (t42);
    let t44 = noir_hw::mul(c2, m[4]);
    let t45 = (t43) - (t44);
    let t46 = noir_hw::mul(c0, m[6]);
    let t47 = (t45) + (t46);
    let t48 = noir_hw::mul(1073741824, m[7]);
    let t49 = (t47) - (t48);
    f[4] = t49;
    // f[5] = c1 * m[0] + c0 * m[1] + c0 * m[2] + c2 * m[3] + c3 * m[4] + c4 * m[5] + c4 * m[6] + 0.25 * m[8]
    let t50 = noir_hw::mul(c1, m[0]);
    let t51 = noir_hw::mul(c0, m[1]);
    let t52 = (t50) + (t51);
    let t53 = noir_hw::mul(c0, m[2]);
    let t54 = (t52) + (t53);
    let t55 = noir_hw::mul(c2, m[3]);
    let t56 = (t54) + (t55);
    let t57 = noir_hw::mul(c3, m[4]);
    let t58 = (t56) + (t57);
    let t59 = noir_hw::mul(c4, m[5]);
    let t60 = (t58) + (t59);
    let t61 = noir_hw::mul(c4, m[6]);
    let t62 = (t60) + (t61);
    let t63 = noir_hw::mul(1073741824, m[8]);
    let t64 = (t62) + (t63);
    f[5] = t64;
    // f[6] = c1 * m[0] - c0 * m[1] + c0 * m[2] + c2 * m[3] + c3 * m[4] - c4 * m[5] + c4 * m[6] - 0.25 * m[8]
    let t65 = noir_hw::mul(c1, m[0]);
    let t66 = noir_hw::mul(c0, m[1]);
    let t67 = (t65) - (t66);
    let t68 = noir_hw::mul(c0, m[2]);
    let t69 = (t67) + (t68);
    let t70 = noir_hw::mul(c2, m[3]);
    let t71 = (t69) + (t70);
    let t72 = noir_hw::mul(c3, m[4]);
    let t73 = (t71) + (t72);
    let t74 = noir_hw::mul(c4, m[5]);
    let t75 = (t73) - (t74);
    let t76 = noir_hw::mul(c4, m[6]);
    let t77 = (t75) + (t76);
    let t78 = noir_hw::mul(1073741824, m[8]);
    let t79 = (t77) - (t78);
    f[6] = t79;
    // f[7] = c1 * m[0] - c0 * m[1] - c0 * m[2] + c2 * m[3] + c3 * m[4] - c4 * m[5] - c4 * m[6] + 0.25 * m[8]
    let t80 = noir_hw::mul(c1, m[0]);
    let t81 = noir_hw::mul(c0, m[1]);
    let t82 = (t80) - (t81);
    let t83 = noir_hw::mul(c0, m[2]);
    let t84 = (t82) - (t83);
    let t85 = noir_hw::mul(c2, m[3]);
    let t86 = (t84) + (t85);
    let t87 = noir_hw::mul(c3, m[4]);
    let t88 = (t86) + (t87);
    let t89 = noir_hw::mul(c4, m[5]);
    let t90 = (t88) - (t89);
    let t91 = noir_hw::mul(c4, m[6]);
    let t92 = (t90) - (t91);
    let t93 = noir_hw::mul(1073741824, m[8]);
    let t94 = (t92) + (t93);
    f[7] = t94;
    // f[8] = c1 * m[0] + c0 * m[1] - c0 * m[2] + c2 * m[3] + c3 * m[4] + c4 * m[5] - c4 * m[6] - 0.25 * m[8]
    let t95 = noir_hw::mul(c1, m[0]);
    let t96 = noir_hw::mul(c0, m[1]);
    let t97 = (t95) + (t96);
    let t98 = noir_hw::mul(c0, m[2]);
    let t99 = (t97) - (t98);
    let t100 = noir_hw::mul(c2, m[3]);
    let t101 = (t99) + (t100);
    let t102 = noir_hw::mul(c3, m[4]);
    let t103 = (t101) + (t102);
    let t104 = noir_hw::mul(c4, m[5]);
    let t105 = (t103) + (t104);
    let t106 = noir_hw::mul(c4, m[6]);
    let t107 = (t105) - (t106);
    let t108 = noir_hw::mul(1073741824, m[8]);
    let t109 = (t107) - (t108);
    f[8] = t109;
    f
}

fn f2m_loc(f: [Field; NS], m: [Field; NS]) -> [Field; NS] {
    let mut m = m;
    // m[0] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
    let t1 = (f[0]) + (f[1]);
    let t2 = (t1) + (f[2]);
    let t3 = (t2) + (f[3]);
    let t4 = (t3) + (f[4]);
    let t5 = (t4) + (f[5]);
    let t6 = (t5) + (f[6]);
    let t7 = (t6) + (f[7]);
    let t8 = (t7) + (f[8]);
    m[0] = t8;
    // m[1] = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
    let t9 = (f[1]) - (f[3]);
    let t10 = (t9) + (f[5]);
    let t11 = (t10) - (f[6]);
    let t12 = (t11) - (f[7]);
    let t13 = (t12) + (f[8]);
    m[1] = t13;
    // m[2] = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
    let t14 = (f[2]) - (f[4]);
    let t15 = (t14) + (f[5]);
    let t16 = (t15) + (f[6]);
    let t17 = (t16) - (f[7]);
    let t18 = (t17) - (f[8]);
    m[2] = t18;
    // m[3] = -4.0 * f[0] - f[1] - f[2] - f[3] - f[4] + 2.0 * f[5] + 2.0 * f[6] + 2.0 * f[7] + 2.0 * f[8]
    let t19 = -(17179869184);
    let t20 = noir_hw::mul(t19, f[0]);
    let t21 = (t20) - (f[1]);
    let t22 = (t21) - (f[2]);
    let t23 = (t22) - (f[3]);
    let t24 = (t23) - (f[4]);
    let t25 = noir_hw::mul(8589934592, f[5]);
    let t26 = (t24) + (t25);
    let t27 = noir_hw::mul(8589934592, f[6]);
    let t28 = (t26) + (t27);
    let t29 = noir_hw::mul(8589934592, f[7]);
    let t30 = (t28) + (t29);
    let t31 = noir_hw::mul(8589934592, f[8]);
    let t32 = (t30) + (t31);
    m[3] = t32;
    // m[4] = 4.0 * f[0] - 2.0 * f[1] - 2.0 * f[2] - 2.0 * f[3] - 2.0 * f[4] + f[5] + f[6] + f[7] + f[8]
    let t33 = noir_hw::mul(17179869184, f[0]);
    let t34 = noir_hw::mul(8589934592, f[1]);
    let t35 = (t33) - (t34);
    let t36 = noir_hw::mul(8589934592, f[2]);
    let t37 = (t35) - (t36);
    let t38 = noir_hw::mul(8589934592, f[3]);
    let t39 = (t37) - (t38);
    let t40 = noir_hw::mul(8589934592, f[4]);
    let t41 = (t39) - (t40);
    let t42 = (t41) + (f[5]);
    let t43 = (t42) + (f[6]);
    let t44 = (t43) + (f[7]);
    let t45 = (t44) + (f[8]);
    m[4] = t45;
    // m[5] = -2.0 * f[1] + 2.0 * f[3] + f[5] - f[6] - f[7] + f[8]
    let t46 = -(8589934592);
    let t47 = noir_hw::mul(t46, f[1]);
    let t48 = noir_hw::mul(8589934592, f[3]);
    let t49 = (t47) + (t48);
    let t50 = (t49) + (f[5]);
    let t51 = (t50) - (f[6]);
    let t52 = (t51) - (f[7]);
    let t53 = (t52) + (f[8]);
    m[5] = t53;
    // m[6] = -2.0 * f[2] + 2.0 * f[4] + f[5] + f[6] - f[7] - f[8]
    let t54 = -(8589934592);
    let t55 = noir_hw::mul(t54, f[2]);
    let t56 = noir_hw::mul(8589934592, f[4]);
    let t57 = (t55) + (t56);
    let t58 = (t57) + (f[5]);
    let t59 = (t58) + (f[6]);
    let t60 = (t59) - (f[7]);
    let t61 = (t60) - (f[8]);
    m[6] = t61;
    // m[7] = f[1] - f[2] + f[3] - f[4]
    let t62 = (f[1]) - (f[2]);
    let t63 = (t62) + (f[3]);
    let t64 = (t63) - (f[4]);
    m[7] = t64;
    // m[8] = f[5] - f[6] + f[7] - f[8]
    let t65 = (f[5]) - (f[6]);
    let t66 = (t65) + (f[7]);
    let t67 = (t66) - (f[8]);
    m[8] = t67;
    m
}

fn relaxation_loc(m: [Field; NS]) -> [Field; NS] {
    let mut m = m;
    // m[3] += 1.1312217194570136 * (-2 * m[0] + 3.0 * m[1] * m[1] + 3.0 * m[2] * m[2] - m[3])
    let t2 = -2 * (m[0]);
    let t3 = noir_hw::mul(12884901888, m[1]);
    let t4 = noir_hw::mul(t3, m[1]);
    let t5 = (t2) + (t4);
    let t6 = noir_hw::mul(12884901888, m[2]);
    let t7 = noir_hw::mul(t6, m[2]);
    let t8 = (t5) + (t7);
    let t9 = (t8) - (m[3]);
    let t10 = noir_hw::mul(4858560290, t9);
    m[3] = m[3] + t10;
    // m[4] += 1.1312217194570136 * (m[0] + 1.5 * m[1] * m[1] + 1.5 * m[2] * m[2] - m[4])
    let t11 = noir_hw::mul(6442450944, m[1]);
    let t12 = noir_hw::mul(t11, m[1]);
    let t13 = (m[0]) + (t12);
    let t14 = noir_hw::mul(6442450944, m[2]);
    let t15 = noir_hw::mul(t14, m[2]);
    let t16 = (t13) + (t15);
    let t17 = (t16) - (m[4]);
    let t18 = noir_hw::mul(4858560290, t17);
    m[4] = m[4] + t18;
    // m[5] += 1.1312217194570136 * (-m[1] - m[5])
    let t19 = -(m[1]);
    let t20 = (t19) - (m[5]);
    let t21 = noir_hw::mul(4858560290, t20);
    m[5] = m[5] + t21;
    // m[6] += 1.1312217194570136 * (-m[2] - m[6])
    let t22 = -(m[2]);
    let t23 = (t22) - (m[6]);
    let t24 = noir_hw::mul(4858560290, t23);
    m[6] = m[6] + t24;
    // m[7] += 1.8573551263001487 * (m[1] * m[1] - m[2] * m[2] - m[7])
    let t25 = noir_hw::mul(m[1], m[1]);
    let t26 = noir_hw::mul(m[2], m[2]);
    let t27 = (t25) - (t26);
    let t28 = (t27) - (m[7]);
    let t29 = noir_hw::mul(7977279525, t28);
    m[7] = m[7] + t29;
    // m[8] += 1.8573551263001487 * (m[1] * m[2] - m[8])
    let t30 = noir_hw::mul(m[1], m[2]);
    let t31 = (t30) - (m[8]);
    let t32 = noir_hw::mul(7977279525, t31);
    m[8] = m[8] + t32;
    m
}
fn getf(f: [[[Field; NS]; NY]; NX], floc: [Field; NS], i: u32, j: u32) -> [Field; NS] {
    let mut floc = floc;
    floc[0] = f[i][j][0];
    floc[1] = f[i - 1][j][1];
    floc[2] = f[i][j - 1][2];
    floc[3] = f[i + 1][j][3];
    floc[4] = f[i][j + 1][4];
    floc[5] = f[i - 1][j - 1][5];
    floc[6] = f[i + 1][j - 1][6];
    floc[7] = f[i + 1][j + 1][7];
    floc[8] = f[i - 1][j + 1][8];
    floc
}
fn setf(f: [[[Field; MS]; MY]; MX], floc: [Field; NS], i: u32, j: u32) -> [[[Field; MS]; MY]; MX] {
    let mut f = f;
    f[j][i][0] = floc[0];
    f[j][i][1] = floc[1];
    f[j][i][2] = floc[2];
    f[j][i][3] = floc[3];
    f[j][i][4] = floc[4];
    f[j][i][5] = floc[5];
    f[j][i][6] = floc[6];
    f[j][i][7] = floc[7];
    f[j][i][8] = floc[8];
    f
}
fn periodic_bc(f: [[[Field; NS]; NY]; NX]) -> [[[Field; NS]; NY]; NX] {
    let mut f = f;
    for j in 0..NY {
        for k in 0..NS {
            f[0][j][k] = f[NX - 2][j][k];
            f[NX - 1][j][k] = f[1][j][k];
        }
    }
    for i in 0..NX {
        for k in 0..NS {
            f[i][0][k] = f[i][NY - 2][k];
            f[i][NY - 1][k] = f[i][1][k];
        }
    }
    f
}
fn main(f1: [[[Field; NS]; NY]; NX], f2: [[[Field; MS]; MY]; MX]) -> pub [[[Field; MS]; MY]; MX] {
    for i in 1..NX - 1 {
        for j in 1..NY - 1 {
            noir_hw::check_inputs(f1[i][j]);
        }
    }
    for j in 0..MX {
        for i in 0..MY {
            for k in 0..MS {
                if !((j > 0) & (j < NY - 1) & (i > 0) & (i < NX - 1) & (k < 9)) {
                    noir_hw::check_input(f2[j][i][k]);
                }
            }
        }
    }
    let mut f2 = f2;
    let mut floc = [0; NS];
    let mut mloc = [0; NS];
    let f1 = periodic_bc(f1);
    for i in 1..NX - 1 {
        for j in 1..NY - 1 {
            floc = getf(f1, floc, i, j);
            mloc = f2m_loc(floc, mloc);
            mloc = relaxation_loc(mloc);
            floc = m2f_loc(mloc, floc);
            f2 = setf(f2, floc, i, j);
        }
    }
    f2
}
