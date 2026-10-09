// pythran_ising__ising_step (Pythran ising), Noir baseline.
// Kernel: one checkerboard sweep of the Ising model on field: Int[N, M]
// (returned): the sites (n, m) in the kernel's order (parity of n, then of m),
// each updated in turn from the current field. A site sums its eight periodic
// neighbours (`clamp`; the indices are loop constants),
// dE = 2 * field[n, m] * total, and flips (field[n, m] *= -1) when dE <= 0 or
// np.exp(-dE * beta) > 0.5.
// exp is increasing, so exp(-dE * beta) > 0.5 is decided exactly as
// beta * dE < -ln(0.5) on the Q32 code of beta * dE (an integer comparison
// with the constant -round(ln(0.5) * 2^32); no exp gadget, no approximation
// error). np.float32(beta) is computed in Q32 like any float (NPB_BASELINES.md
// section 5.4): rounding beta to float32 changes the outcome only when
// beta * dE is within 6e-8 relative of ln 2; the fixture's values are far
// from it (beta * dE is 0 or 3.2, against ln 2 = 0.69). dE is an int64 (a
// comparison puts it in the window); the product beta * dE is compared as a
// wide integer, so any int64 dE and Q32 beta have a proof, as NumPy's float64
// product has a value.
// @outputs: int[N * M]

use noir_hw::{check_input, check_inputs, window};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 2; // = field_0
global M: u32 = 2; // = field_1

// -round(ln(0.5) * 2^32): for an integer code c, c / 2^32 < -ln(0.5) exactly
// when c < NEG_LN_HALF (ln(0.5) * 2^32 is not an integer).
global NEG_LN_HALF: Field = 2977044472;

// The kernel's clamp(v, n) of a neighbour index v in [-1, n] (a periodic
// index), with v passed as v + 1 to stay unsigned.
fn clamp(v_plus_one: u32, n: u32) -> u32 {
    (v_plus_one + n - 1) % n
}

// c < NEG_LN_HALF for a c of magnitude below 2^126: c - NEG_LN_HALF is
// negative, i.e. its canonical representative is above 2^200.
unconstrained fn below_hint(c: Field) -> bool {
    !(c - NEG_LN_HALF).lt(2.pow_32(200))
}

// c < NEG_LN_HALF for |c| < 2^126 (beta * dE with both codes in the window):
// the hinted outcome selects NEG_LN_HALF - 1 - c or c - NEG_LN_HALF, which
// must be a 140-bit value (ten 14-bit limbs); the true outcome gives a value
// below 2^127, the false one a negative value, p - (something below 2^127).
fn below_ln_half(c: Field) -> bool {
    // Safety: pinned by the range check below.
    let below = unsafe { below_hint(c) };
    let gap = if below { NEG_LN_HALF - 1 - c } else { c - NEG_LN_HALF };
    gap.assert_max_bit_size::<140>();
    below
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
                        let flip = non_positive | below_ln_half(beta * de);
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
