// pythran_lu__factorMatrix0: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize abs, then scan argmax; gather both RHS rows before scatter/select.
// @outputs: float[N * COLS]

use noir_hw::{abs_int64, check_inputs, div, index_pos, mul};

// Sizes (hw_check.sizes): from fixture.factorMatrix0.json.
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
        // cmax = np.argmax(abs(M[line:n, line])) + line.
        let mut magnitudes = [0; N];
        for i in line..N { magnitudes[i] = abs_int64(m[i][line]); }
        let mut best_val = magnitudes[line];
        let mut rel_idx: Field = 0;
        for i in line+1..N {
            let candidate = magnitudes[i];
            let larger = noir_hw::gt(candidate, best_val);
            best_val = if larger { candidate } else { best_val };
            rel_idx = if larger { (i-line) as Field } else { rel_idx };
        }
        let cmax = rel_idx + line as Field;
        let swap = cmax != line as Field;
        let old = m;
        let mut row = [0; COLS];
        for k in 0..COLS { row[k] = old[index_pos::<INDEX_BITS>(cmax,N)][k]; }
        let mut branch = old;
        for k in 0..COLS { branch[line][k] = row[k]; }
        for p in 0..N {
            for k in 0..COLS {
                branch[p][k] = if cmax == p as Field { old[line][k] } else { branch[p][k] };
            }
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
