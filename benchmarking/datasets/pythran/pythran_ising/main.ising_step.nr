// Source sequential Ising with exp(-dE*beta)>0.5, using canonical Q32.
// np.float32 shares that format; no threshold or binary32-rounding rewrite.
// @outputs: int[N * M]

use noir_hw::{check_input, check_inputs, window, exp, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 2; // = field_0
global M: u32 = 2; // = field_1

// The kernel's clamp(v, n) of a neighbour index v in [-1, n] (a periodic
// index), with v passed as v + 1 to stay unsigned.
fn clamp(v_plus_one: u32, n: u32) -> u32 {
    (v_plus_one + n - 1) % n
}

fn main(field: [[Field; M]; N], beta: Field) -> pub [[Field; M]; N] {
    // Input domain: field holds int64s, beta a Q32 code, in the window.
    for n in 0..N {
        check_inputs(field[n]);
    }
    check_input(beta);
    let mut f = field;
    for n_offset in 0..2 {
        for m_offset in 0..2 {
            for n in n_offset..N {
                for m in m_offset..M {
                    if ((n - n_offset) % 2 == 0) & ((m - m_offset) % 2 == 0) {
                        let mut total: Field = 0;
                        for i in 0..3 {
                            for j in 0..3 {
                                if (i != 1) | (j != 1) {
                                    total += f[clamp(n + i, N)][clamp(m + j, M)];
                                }
                            }
                        }
                        let de = 2 * f[n][m] * total;
                        let non_positive = window(de) <= window(0);
                        let argument = if non_positive { 0 } else { -de * beta };
                        let exponential = exp(argument);
                        let flip = non_positive | (window(exponential) > window(SCALE / 2));
                        if flip {
                            f[n][m] = -f[n][m];
                        }
                    }
                }
            }
        }
    }
    f
}
