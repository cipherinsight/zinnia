// pythran_lu__factorMatrix1: literal translation of kernel.py (LITERAL_BASELINES.md).
// Repeat abs in the taken branch; private writes scan every row, then select the branch.
// @outputs: float[N * COLS]

use noir_hw::{abs_int64, check_inputs, div, index_pos, mul};

// Sizes (hw_check.sizes): from fixture.factorMatrix1.json.
global N: u32 = 5; // = M_0
global COLS: u32 = 5; // = M_1
// `index_pos` width: 8 bits up to 128 elements, 28 above (NOIR_BASELINES.md).
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (M_0 - 1) // 128)

fn main(M: [[Field; COLS]; N]) -> pub [[Field; COLS]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(M[i]);
    }
    let mut m = M;
    for line in 0..N - 1 {
        // cmax = line; vmax = abs(M[line,line]); scan with conditional updates.
        let mut cmax = line as Field;
        let mut best_val = abs_int64(m[line][line]);
        for i in line+1..N {
            let candidate = abs_int64(m[i][line]);
            let larger = noir_hw::gt(candidate,best_val);
            let assigned = abs_int64(m[i][line]);
            best_val = if larger { assigned } else { best_val };
            cmax = if larger { i as Field } else { cmax };
        }
        let swap = cmax != line as Field;
        let old = m;
        let mut branch = old;
        for k in line..COLS {
            let t = branch[line][k];
            branch[line][k] = branch[index_pos::<INDEX_BITS>(cmax,N)][k];
            for p in 0..N { branch[p][k] = if cmax == p as Field { t } else { branch[p][k] }; }
        }
        for p in 0..N { for k in 0..COLS { m[p][k] = if swap { branch[p][k] } else { old[p][k] }; } }

        // pivot = M[line, line]; for j in line+1..n: v = M[j, line] / pivot.
        let pivot = m[line][line];
        for j in line + 1..N {
            let v = div(m[j][line], pivot);
            for k in line..COLS {
                m[j][k] -= mul(v, m[line][k]);
            }
        }
    }
    m
}
