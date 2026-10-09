// pythran_value_index__tri: literal translation of kernel.py (LITERAL_BASELINES.md).
// Stable adjacent bubble sort: N*(N-1)/2 comparisons carrying indices, then standard indexing.
// score_t is written before it is rebound; none of its private elements is read.
// @outputs: int[N * COLS]
use noir_hw::{check_inputs, gt, index_pos};
global N: u32 = 10; // = score_0
global COLS: u32 = 10; // = score_1
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (score_0 - 1) // 128)
fn main(score: [[Field; COLS]; N], score_t: [[Field; COLS]; N]) -> pub [[Field; COLS]; N] {
    for i in 0..N {
        check_inputs(score[i]);
    }
    let mut score_t = score_t;
    score_t[0][0] = score[0][0];
    let mut keys = [0; N];
    let mut order = [0; N];
    for i in 0..N {
        keys[i] = score[i][1];
        order[i] = i as Field;
    }
    for pass in 0..N - 1 {
        for j in 0..N - 1 - pass {
            let swap = gt(keys[j], keys[j + 1]);
            let a = keys[j];
            let b = keys[j + 1];
            keys[j] = if swap { b } else { a };
            keys[j + 1] = if swap { a } else { b };
            let a = order[j];
            let b = order[j + 1];
            order[j] = if swap { b } else { a };
            order[j + 1] = if swap { a } else { b };
        }
    }
    let mut gathered = [[0; COLS]; N];
    for i in 0..N {
        gathered[i] = score[index_pos::<INDEX_BITS>(order[i], N)];
    }
    score_t = gathered;
    score_t
}
