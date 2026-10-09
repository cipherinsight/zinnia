// pythran_d2q9_nxnyns__one_time_step (Pythran d2q9_nxnyns), Noir baseline.
// Kernel: one lattice-Boltzmann D2Q9 step on f1: Float[NX, NY, 9] into
// f2: Float[MX, MY, 9]; returns f2. periodic_bc(f1) copies rows NX - 2 and 1
// into rows 0 and NX - 1, then columns NY - 2 and 1 into columns 0 and
// NY - 1; for every interior site (i, j) the nine populations streamed in
// (getf) go to moments (f2m_loc), are relaxed (relaxation_loc), go back
// (m2f_loc) and are stored at f2[j, i] (setf, transposed). The 9 is D2Q9's,
// not a size.
// periodic_bc only copies, so f1 after it is the input read through `wrap`
// on both axes, and every interior site reads only interior elements of the
// input: those are the f1 elements checked. The written f2 elements are not
// checked; the others are output as they are.
// The moments are exact integer combinations. Every product of relaxation
// and m2f_loc is rounded as NumPy evaluates it (3.0 * m1 * m1 is (3 m1) m1,
// exact then rounded; 1.5 * m1 * m1 rounds both); products m2f_loc repeats
// across the nine outputs (c1 * m[0], ...) are computed once. Constants that
// are powers of two times an integer (0.25 * m, 1.5 * m) round by
// `floor_div_pow2` (floor(m / 4), floor(3 m / 2)): the same values as a
// rescale, with a remainder of 1 or 2 bits.
// @outputs: float[MX * MY * 9]

use noir_hw::{check_inputs, floor_div_pow2, mul, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global NX: u32 = 16; // = f1_0
global NY: u32 = 16; // = f1_1
global MX: u32 = 16; // = f2_0
global MY: u32 = 16; // = f2_1

// round(c * 2^32) of the kernel's float constants.
global C0_Q: Field = 715827883; // 1.0 / 6
global C1_Q: Field = 477218588; // 1.0 / 9
global C2_Q: Field = 238609294; // 1.0 / 18
global C3_Q: Field = 119304647; // 1.0 / 36
global C4_Q: Field = 357913941; // 1.0 / 12
global OMEGA_E_Q: Field = 4858560290; // 1.1312217194570136
global OMEGA_XY_Q: Field = 7977279525; // 1.8573551263001487

// Index into the post-periodic_bc f1 along an axis of length len.
fn wrap(v: u32, len: u32) -> u32 {
    if v == 0 {
        len - 2
    } else if v == len - 1 {
        1
    } else {
        v
    }
}

fn moments(f: [Field; 9]) -> [Field; 9] {
    [
        f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8],
        f[1] - f[3] + f[5] - f[6] - f[7] + f[8],
        f[2] - f[4] + f[5] + f[6] - f[7] - f[8],
        -4 * f[0] - f[1] - f[2] - f[3] - f[4] + 2 * (f[5] + f[6] + f[7] + f[8]),
        4 * f[0] - 2 * (f[1] + f[2] + f[3] + f[4]) + f[5] + f[6] + f[7] + f[8],
        -2 * f[1] + 2 * f[3] + f[5] - f[6] - f[7] + f[8],
        -2 * f[2] + 2 * f[4] + f[5] + f[6] - f[7] - f[8],
        f[1] - f[2] + f[3] - f[4],
        f[5] - f[6] + f[7] - f[8],
    ]
}

fn relaxation(m: [Field; 9]) -> [Field; 9] {
    let m1_sq = m[1] * m[1];
    let m2_sq = m[2] * m[2];
    // 3.0 * m1 * m1 = (3 m1) m1: one rounding of 3 m1^2.
    let three_m1_m1 = rescale(3 * m1_sq);
    let three_m2_m2 = rescale(3 * m2_sq);
    // 1.5 * m1 * m1 = floor(3 m1 / 2) * m1, rounded.
    let one_half_m1_m1 = mul(floor_div_pow2::<1>(3 * m[1]), m[1]);
    let one_half_m2_m2 = mul(floor_div_pow2::<1>(3 * m[2]), m[2]);
    let mut out = m;
    out[3] = m[3] + mul(OMEGA_E_Q, -2 * m[0] + three_m1_m1 + three_m2_m2 - m[3]);
    out[4] = m[4] + mul(OMEGA_E_Q, m[0] + one_half_m1_m1 + one_half_m2_m2 - m[4]);
    out[5] = m[5] + mul(OMEGA_E_Q, -m[1] - m[5]);
    out[6] = m[6] + mul(OMEGA_E_Q, -m[2] - m[6]);
    out[7] = m[7] + mul(OMEGA_XY_Q, rescale(m1_sq) - rescale(m2_sq) - m[7]);
    out[8] = m[8] + mul(OMEGA_XY_Q, mul(m[1], m[2]) - m[8]);
    out
}

fn populations(m: [Field; 9]) -> [Field; 9] {
    let c1_m0 = mul(C1_Q, m[0]);
    let c0_m1 = mul(C0_Q, m[1]);
    let c0_m2 = mul(C0_Q, m[2]);
    let c3_m3 = mul(C3_Q, m[3]);
    let c2_m4 = mul(C2_Q, m[4]);
    let c0_m5 = mul(C0_Q, m[5]);
    let c0_m6 = mul(C0_Q, m[6]);
    let q_m7 = floor_div_pow2::<2>(m[7]); // 0.25 * m[7]
    let c2_m3 = mul(C2_Q, m[3]);
    let c3_m4 = mul(C3_Q, m[4]);
    let c4_m5 = mul(C4_Q, m[5]);
    let c4_m6 = mul(C4_Q, m[6]);
    let q_m8 = floor_div_pow2::<2>(m[8]); // 0.25 * m[8]
    let c1_m3 = mul(C1_Q, m[3]);
    let c1_m4 = mul(C1_Q, m[4]);
    [
        c1_m0 - c1_m3 + c1_m4,
        c1_m0 + c0_m1 - c3_m3 - c2_m4 - c0_m5 + q_m7,
        c1_m0 + c0_m2 - c3_m3 - c2_m4 - c0_m6 - q_m7,
        c1_m0 - c0_m1 - c3_m3 - c2_m4 + c0_m5 + q_m7,
        c1_m0 - c0_m2 - c3_m3 - c2_m4 + c0_m6 - q_m7,
        c1_m0 + c0_m1 + c0_m2 + c2_m3 + c3_m4 + c4_m5 + c4_m6 + q_m8,
        c1_m0 - c0_m1 + c0_m2 + c2_m3 + c3_m4 - c4_m5 + c4_m6 - q_m8,
        c1_m0 - c0_m1 - c0_m2 + c2_m3 + c3_m4 - c4_m5 - c4_m6 + q_m8,
        c1_m0 + c0_m1 - c0_m2 + c2_m3 + c3_m4 + c4_m5 - c4_m6 - q_m8,
    ]
}

fn main(f1: [[[Field; 9]; NY]; NX], f2: [[[Field; 9]; MY]; MX]) -> pub [[[Field; 9]; MY]; MX] {
    // Input domain: the interior of f1, and every element of f2 no site
    // writes, are Q32 codes in the window.
    for r in 1..NX - 1 {
        for c in 1..NY - 1 {
            check_inputs(f1[r][c]);
        }
    }
    let mut result = f2;
    for r in 0..MX {
        for c in 0..MY {
            if (r < 1) | (r > NY - 2) | (c < 1) | (c > NX - 2) {
                check_inputs(f2[r][c]);
            }
        }
    }
    for i in 1..NX - 1 {
        for j in 1..NY - 1 {
            let streamed = [
                f1[wrap(i, NX)][wrap(j, NY)][0],
                f1[wrap(i - 1, NX)][wrap(j, NY)][1],
                f1[wrap(i, NX)][wrap(j - 1, NY)][2],
                f1[wrap(i + 1, NX)][wrap(j, NY)][3],
                f1[wrap(i, NX)][wrap(j + 1, NY)][4],
                f1[wrap(i - 1, NX)][wrap(j - 1, NY)][5],
                f1[wrap(i + 1, NX)][wrap(j - 1, NY)][6],
                f1[wrap(i + 1, NX)][wrap(j + 1, NY)][7],
                f1[wrap(i - 1, NX)][wrap(j + 1, NY)][8],
            ];
            result[j][i] = populations(relaxation(moments(streamed)));
        }
    }
    result
}
