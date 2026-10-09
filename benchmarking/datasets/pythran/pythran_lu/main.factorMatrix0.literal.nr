// pythran_lu__factorMatrix0 (Pythran lu), Noir baseline: literal translation
// of kernel.py (LITERAL_BASELINES.md).
//
// Kernel (factorMatrix0): LU elimination with partial pivoting of
// M: Float[N, COLS] in place; the output is M after the call
// (`mutated_outputs`). For each line < N - 1:
//   cmax = np.argmax(abs(M[line:n, line])) + line
//   if cmax != line: M[[line, cmax]] = M[[cmax, line]]
//   pivot = M[line, line]
//   for j in range(line + 1, n):
//       v = M[j, line] / pivot
//       for k in range(line, m): M[j, k] -= v * M[line, k]
//
// Interpretations (L1-L6):
// - np.argmax(abs(...)) (L5): a linear scan over rows line..N of column
//   line, tracking the best magnitude and its 0-based position in the
//   subarray as plain values (not a one-hot), strict `<` so the first
//   maximum wins; cmax = argmax + line is then one addition, as in the
//   source.
// - if cmax != line: M[[line, cmax]] = M[[cmax, line]] (L4/L5): the
//   comparison cmax != line is evaluated explicitly as its own step. The
//   swap reads row cmax by the private position with `index_pos` (L5:
//   "a[i] with private index: library's standard indexing gadget -- this IS
//   the direct form"; Noir's one-hot `pick` is L6-forbidden) and writes it
//   back with the private-index write formula a[k] = select(i == k, v,
//   a[k]), both gated by that comparison -- a separate step from the argmax
//   scan, not fused with it.
// - v = M[j, line] / pivot (L5): the plain `div`, which accepts a divisor of
//   either sign. The optimized baseline's `div_by_abs` reuses the argmax
//   scan's running magnitude as the pivot's magnitude to skip `div`'s own
//   sign handling; L6 forbids both `div_by_abs` and reusing a value from one
//   source operation (the argmax) in another (the division), so this
//   baseline calls plain `div`.
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
        let mut best_val = abs_int64(m[line][line]);
        let mut rel_idx: Field = 0;
        for i in line + 1..N {
            let candidate = abs_int64(m[i][line]);
            let larger = ((best_val as u64) < (candidate as u64)) as Field;
            best_val += larger * (candidate - best_val);
            rel_idx += larger * (((i - line) as Field) - rel_idx);
        }
        let cmax = rel_idx + (line as Field);

        // if cmax != line: M[[line, cmax]] = M[[cmax, line]]
        let swap = (cmax != (line as Field)) as Field;
        for k in 0..COLS {
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
