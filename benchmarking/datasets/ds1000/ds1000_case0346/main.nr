// DS-1000 case 0346 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[row-1, :] *= multiply_number; result = cumsum(a[row-1, :]) on Float[8,6].
// Mirrors Zinnia's lowering ALGORITHM (row variant of 0345): select-chain
// mux gather over rows, surviving max-stride bound assert, per-column
// qmul, dual-equality scatter-back, second post-write gather, cumsum.

use noir_fpc::{qadd, qmul, Q};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], row: Field, multiply_number: Field) -> pub [Field; 6] {
    let idx = row - 1;
    let mut sel: [Field; 8] = [0; 8];
    for i in 1..ROWS {
        sel[i] = if idx == (i as Field) { 1 } else { 0 };
    }
    // First gather: select-chain mux per column.
    let mut g: [Field; 6] = [0; 6];
    for j in 0..COLS {
        let mut cur: Field = a[j];
        for i in 1..ROWS {
            cur = sel[i] * a[i * COLS + j] + (1 - sel[i]) * cur;
        }
        g[j] = cur;
    }
    // Max-stride flat-address bound assert.
    let flat: Field = idx * (COLS as Field) + 5;
    assert((flat as u128) < (48 as u128));
    // In-place row multiply.
    let mn_fp = multiply_number * Q;
    let mut scaled: [Field; 6] = [0; 6];
    for j in 0..COLS {
        scaled[j] = qmul(g[j], mn_fp);
    }
    // Scatter-back with dual-equality indicators.
    let mut ind: [Field; 8] = [0; 8];
    for i in 0..ROWS {
        let hit = (idx == i as Field) | (idx + ((ROWS - i) as Field) == 0);
        ind[i] = if hit { 1 } else { 0 };
    }
    let mut b: [Field; 48] = a;
    for i in 0..ROWS {
        for j in 0..COLS {
            b[i * COLS + j] = ind[i] * scaled[j] + (1 - ind[i]) * a[i * COLS + j];
        }
    }
    // Second gather (post-write) + cumsum chain.
    let mut out: [Field; 6] = [0; 6];
    let mut acc: Field = 0;
    for j in 0..COLS {
        let mut cur: Field = b[j];
        for i in 1..ROWS {
            cur = sel[i] * b[i * COLS + j] + (1 - sel[i]) * cur;
        }
        if j == 0 {
            acc = cur;
        } else {
            acc = qadd(acc, cur);
        }
        out[j] = acc;
    }
    out
}
