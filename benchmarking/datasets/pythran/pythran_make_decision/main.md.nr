// pythran_make_decision__md (Pythran make_decision), Noir baseline.
// Kernel: for each E[i] of E: Complex[L], the symbol of symbols: Complex[M]
// nearest to it, symbols[np.argmin(abs(E[i] - symbols) ** 2)] (the first one
// on ties); returns them as Complex[L].
// Preserve complex magnitude, then Q32 square, then sequential first argmin.
// Component squares are individually rounded before sqrt; strict comparisons
// preserve first-index ties. No argmin hint or raw-distance shortcut.
// @outputs: float[2 * L]

use noir_hw::{check_inputs, mul, sqrt, lt};

// Sizes (hw_check.sizes): from fixture.json.
global L: u32 = 16; // = E_0
global M: u32 = 8; // = symbols_0

fn main(E: [[Field; 2]; L], symbols: [[Field; 2]; M]) -> pub [[Field; 2]; L] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..L {
        check_inputs(E[i]);
    }
    for k in 0..M {
        check_inputs(symbols[k]);
    }
    let mut result: [[Field; 2]; L] = [[0; 2]; L];
    for i in 0..L {
        let mut r: [Field; M] = [0; M];
        for k in 0..M {
            let dr = E[i][0] - symbols[k][0];
            let di = E[i][1] - symbols[k][1];
            let magnitude = sqrt(mul(dr, dr) + mul(di, di));
            r[k] = mul(magnitude, magnitude);
        }
        let mut minimum = r[0];
        let mut selected = symbols[0];
        for k in 1..M {
            let better = lt(r[k], minimum);
            minimum = if better { r[k] } else { minimum };
            selected = if better { symbols[k] } else { selected };
        }
        result[i] = selected;
    }
    result
}
