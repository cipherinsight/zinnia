// pythran_value_index__tri (Pythran value_index), Noir baseline.
// Kernel: score[np.argsort(score[:, 1])] for `score: Int[N, COLS]`, the rows
// reordered by their second column; `score_t` is written, then rebound, and
// never read (not loaded).
// NumPy's tie order is not specified (its default sort is not stable on every
// platform); like CONVENTIONS section 3, ties keep the row order (a stable
// argsort). The order is computed unconstrained and verified: row i has the
// key window(score[i, 1]) * 2^KEY_SHIFT + i, and the keys read through the
// claimed order must strictly increase (one range check of each step). The
// keys are distinct, so the order is a permutation, and it is the stable
// order. The rows are then read through it (native indexing). KEY_SHIFT makes
// the step check a multiple of 14 bits wide: no range table beyond the 64-bit
// checks' (NOIR_BASELINES.md).
// @outputs: int[N * COLS]

use noir_hw::{array_index, check_inputs, window};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = score_0
global COLS: u32 = 10; // = score_1
global KEY_SHIFT: u32 = 6; // = 14 * cdiv(64 + clog2(score_0), 14) - 64
global KEY_BITS: u32 = 64 + KEY_SHIFT;
// `array_index` width: 8 bits up to 128 elements, 28 above (NOIR_BASELINES.md).
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (score_0 - 1) // 128)

// The stable ascending order of the rows by their windowed second column
// (insertion sort).
unconstrained fn order_hint(score: [[Field; COLS]; N]) -> [Field; N] {
    let mut w: [u64; N] = [0; N];
    let mut order: [u32; N] = [0; N];
    for i in 0..N {
        w[i] = window(score[i][1]);
        order[i] = i;
    }
    for i in 1..N {
        let mut j = i;
        while j > 0 {
            if w[order[j]] >= w[order[j - 1]] {
                break;
            }
            let t = order[j];
            order[j] = order[j - 1];
            order[j - 1] = t;
            j -= 1;
        }
    }
    let mut out: [Field; N] = [0; N];
    for k in 0..N {
        out[k] = order[k] as Field;
    }
    out
}

fn main(score: [[Field; COLS]; N], score_t: [[Field; COLS]; N]) -> pub [[Field; COLS]; N] {
    let _ = score_t;
    // Input domain: every input element read is an int64.
    for i in 0..N {
        check_inputs(score[i]);
    }
    // Safety: pinned by the key checks below.
    let order = unsafe { order_hint(score) };
    let shift = 2.pow_32(KEY_SHIFT as Field);
    let mut sorted: [[Field; COLS]; N] = [[0; COLS]; N];
    let mut previous: Field = 0;
    for k in 0..N {
        // `order[k]` in [0, N - 1] (`array_index`), so the key's index part
        // fits below 2^KEY_SHIFT and the reads are in bounds.
        let i = order[k];
        let row = score[array_index::<INDEX_BITS>(i, (N - 1) as Field)];
        let key = (window(row[1]) as Field) * shift + i;
        if k > 0 {
            (key - previous - 1).assert_max_bit_size::<KEY_BITS>();
        }
        previous = key;
        sorted[k] = row;
    }
    sorted
}
