// pythran_lu__factorMatrix0 (Pythran lu), Noir baseline.
// Kernel: LU elimination with partial pivoting of `M: Float[N, COLS]` in
// place; the output is M after the call (`mutated_outputs`). For each
// line < N - 1: the pivot row is the first row i >= line with the largest
// |M[i, line]| (np.argmax), it is swapped with row `line`, and every row below
// subtracts M[j, line] / pivot times row `line` from column `line` on.
// The pivot search compares the magnitudes (`abs_int64`, below 2^63, so the
// u64 casts are exact), keeping a one-hot of the row reached so far (a strict
// `<` keeps the first maximum); the swap selects whole rows through the
// one-hot. The largest magnitude is the pivot's, so the quotients divide by it
// directly (`div_by_abs`: no window check of the pivot).
// @outputs: float[N * COLS]

use noir_hw::{abs_int64, check_inputs, div_by_abs, mul};

// Sizes (hw_check.sizes): from fixture.factorMatrix0.json.
global N: u32 = 5; // = M_0
global COLS: u32 = 5; // = M_1

fn main(M: [[Field; COLS]; N]) -> pub [[Field; COLS]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(M[i]);
    }
    let mut m = M;
    for line in 0..N - 1 {
        // The first row with the largest magnitude in column `line`.
        let mut largest = abs_int64(m[line][line]);
        let mut hot: [Field; N] = [0; N];
        hot[line] = 1;
        for i in line + 1..N {
            let magnitude = abs_int64(m[i][line]);
            let larger = ((largest as u64) < (magnitude as u64)) as Field;
            largest += larger * (magnitude - largest);
            for t in line..i {
                hot[t] -= larger * hot[t];
            }
            hot[i] = larger;
        }
        // Swap rows `line` and the marked one.
        for k in 0..COLS {
            let top = m[line][k];
            let mut picked: Field = 0;
            for i in line..N {
                picked += hot[i] * m[i][k];
            }
            for i in line + 1..N {
                m[i][k] += hot[i] * (top - m[i][k]);
            }
            m[line][k] = picked;
        }
        // Eliminate below the pivot.
        let pivot = m[line][line];
        for j in line + 1..N {
            let v = div_by_abs(m[j][line], pivot, largest);
            for k in line..COLS {
                m[j][k] -= mul(v, m[line][k]);
            }
        }
    }
    m
}
