// pythran_lu__factorMatrix1 (Pythran lu), Noir baseline: literal translation
// of kernel.py (LITERAL_BASELINES.md).
//
// Kernel (factorMatrix1): LU elimination with partial pivoting of
// M: Float[N, COLS] in place; the output is M after the call
// (`mutated_outputs`). For each line < N - 1:
//   cmax = line; vmax = abs(M[line, line])
//   for i in range(line + 1, n):
//       if abs(M[i, line]) > vmax: vmax = abs(M[i, line]); cmax = i
//   if cmax != line:
//       for j in range(line, m): t = M[line,j]; M[line,j] = M[cmax,j]; M[cmax,j] = t
//   pivot = M[line, line]
//   for j in range(line + 1, n):
//       v = M[j, line] / pivot
//       for k in range(line, m): M[j, k] -= v * M[line, k]
//
// Interpretations (L1-L6): same as factorMatrix0's literal baseline -- the
// `if ... > vmax: ...; cmax = i` loop body (L4) is the same running
// best-value/best-index update as an argmax scan; the explicit
// column-by-column swap with a temporary (t = M[line,j]; ...) is the same
// private-index row swap, restricted to columns line..m as the source loop
// says (unlike factorMatrix0's fancy-index swap, which moves the whole
// row); and v = M[j, line] / pivot uses the plain `div` rather than the
// optimized baseline's `div_by_abs` reuse of the argmax scan's magnitude
// (L6-forbidden cross-statement reuse).
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
        // cmax = line; vmax = abs(M[line, line]).
        // for i in line+1..n: if abs(M[i, line]) > vmax: vmax = ...; cmax = i.
        let mut best_val = abs_int64(m[line][line]);
        let mut rel_idx: Field = 0;
        for i in line + 1..N {
            let candidate = abs_int64(m[i][line]);
            let larger = ((best_val as u64) < (candidate as u64)) as Field;
            best_val += larger * (candidate - best_val);
            rel_idx += larger * (((i - line) as Field) - rel_idx);
        }
        let cmax = rel_idx + (line as Field);

        // if cmax != line: for j in line..m: swap M[line, j] and M[cmax, j].
        let swap = (cmax != (line as Field)) as Field;
        for k in line..COLS {
            let row_line_k = m[line][k];
            let mut candidates: [Field; N] = [0; N];
            for p in line..N {
                candidates[p] = m[p][k];
            }
            let pos = index_pos::<INDEX_BITS>(rel_idx, N - line);
            let row_cmax_k = candidates[line + pos];
            m[line][k] = row_line_k + swap * (row_cmax_k - row_line_k);
            for p in line + 1..N {
                let is_cmax = ((((p - line) as Field) - rel_idx) == 0) as Field;
                let hit = is_cmax * swap;
                m[p][k] += hit * (row_line_k - m[p][k]);
            }
        }

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
