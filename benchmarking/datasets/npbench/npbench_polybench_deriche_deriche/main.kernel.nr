// Source Deriche coefficients and recurrences with canonical Q32 arithmetic.
// canonical_exp(-2*alpha) is independent, not squared canonical_exp(-alpha). log(2) is the
// canonical constant 2977044472 in the canonical_exp((-alpha)*log(2)) power lowering.
// @outputs: float[R * C]

use noir_hw::{check_input, check_inputs, div, canonical_exp, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global R: u32 = 16; // = imgIn_0
global C: u32 = 8; // = imgIn_1

global LN2_Q: Field = 2977044472; // round(ln 2 * 2^32)

// The coefficients of one recurrence pair:
// y1[j] = a1 x[j] + a2 x[j - 1] + b1 y1[j - 1] + b2 y1[j - 2] and
// y2[j] = a3 x[j + 1] + a4 x[j + 2] + b1 y2[j + 1] + b2 y2[j + 2].
struct Coefficients {
    a1: Field,
    a2: Field,
    a3: Field,
    a4: Field,
    b1: Field,
    b2: Field,
}

// y1 + y2 along x (c1 * (y1 + y2) with c1 = 1).
fn filter<let K: u32>(x: [Field; K], c: Coefficients) -> [Field; K] {
    // y1[0] = a1 x[0]; y1[1] = a1 x[1] + a2 x[0] + b1 y1[0]; then four terms.
    let mut y1: [Field; K] = [0; K];
    for j in 0..K {
        let mut y = mul(c.a1, x[j]);
        if j >= 1 {
            y += mul(c.a2, x[j - 1]) + mul(c.b1, y1[j - 1]);
        }
        if j >= 2 {
            y += mul(c.b2, y1[j - 2]);
        }
        y1[j] = y;
    }
    // y2[K - 1] = 0; y2[K - 2] = a3 x[K - 1]; then the terms that are not
    // products with y2[K - 1].
    let mut y2: [Field; K] = [0; K];
    for t in 1..K {
        let j = K - 1 - t;
        let mut y = mul(c.a3, x[j + 1]);
        if j + 2 < K {
            y += mul(c.a4, x[j + 2]) + mul(c.b1, y2[j + 1]);
        }
        if j + 3 < K {
            y += mul(c.b2, y2[j + 2]);
        }
        y2[j] = y;
    }
    let mut sum: [Field; K] = [0; K];
    for j in 0..K {
        sum[j] = y1[j] + y2[j];
    }
    sum
}

fn main(alpha: Field, imgIn: [[Field; C]; R]) -> pub [[Field; C]; R] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    for i in 0..R {
        check_inputs(imgIn[i]);
    }
    let e1 = canonical_exp(-alpha); // np.exp(-alpha)
    let e_two = canonical_exp(2 * alpha); // np.exp(2.0 * alpha)
    let e2 = canonical_exp(-2 * alpha); // np.exp(-2.0 * alpha)
    let k = div(mul(SCALE - e1, SCALE - e1), SCALE + mul(alpha, e1) - e_two);
    let k_e1 = mul(k, e1);
    let c = Coefficients {
        a1: k,
        a2: mul(k_e1, alpha - SCALE),
        a3: mul(k_e1, alpha + SCALE),
        a4: mul(-k, e2),
        b1: canonical_exp(mul(-alpha, LN2_Q)), // 2.0 ** (-alpha)
        b2: -e2,
    };
    // Along the rows, then along the columns of the result.
    let mut horizontal: [[Field; C]; R] = [[0; C]; R];
    for i in 0..R {
        horizontal[i] = filter(imgIn[i], c);
    }
    let mut result: [[Field; C]; R] = [[0; C]; R];
    for j in 0..C {
        let mut column: [Field; R] = [0; R];
        for i in 0..R {
            column[i] = horizontal[i][j];
        }
        let filtered = filter(column, c);
        for i in 0..R {
            result[i][j] = filtered[i];
        }
    }
    result
}
