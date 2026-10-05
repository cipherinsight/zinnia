// DS-1000 case 0345 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[:, col-1] *= multiply_number; result = cumsum(a[:, col-1]) on Float[8,6].
// Mirrors Zinnia's lowering ALGORITHM: select-chain mux gather (single-eq
// indicators, column 0 as base), surviving max-stride flat-address bound
// assert, per-row qmul, dual-equality scatter-back of the full array
// (in-place `*=` semantics), a second post-write gather, then the cumsum
// chain.

use noir_fpc::{qadd, qmul, Q};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], col: Field, multiply_number: Field) -> pub [Field; 8] {
    let idx = col - 1;
    // Read-path mux indicators (single equality; base = column 0).
    let mut sel: [Field; 6] = [0; 6];
    for j in 1..COLS {
        sel[j] = if idx == (j as Field) { 1 } else { 0 };
    }
    // First gather: select-chain mux per row.
    let mut g: [Field; 8] = [0; 8];
    for i in 0..ROWS {
        let mut cur: Field = a[i * COLS];
        for j in 1..COLS {
            cur = sel[j] * a[i * COLS + j] + (1 - sel[j]) * cur;
        }
        g[i] = cur;
    }
    // Max-stride flat-address bound (the one discharge the subsumption
    // pass keeps; pins idx to [0, COLS) over the u128 view).
    let flat: Field = 42 + idx;
    assert((flat as u128) < (48 as u128));
    // In-place column multiply.
    let mn_fp = multiply_number * Q;
    let mut scaled: [Field; 8] = [0; 8];
    for i in 0..ROWS {
        scaled[i] = qmul(g[i], mn_fp);
    }
    // Scatter-back with dual-equality indicators (numpy negative-index
    // write semantics).
    let mut ind: [Field; 6] = [0; 6];
    for j in 0..COLS {
        let hit = (idx == j as Field) | (idx + ((COLS - j) as Field) == 0);
        ind[j] = if hit { 1 } else { 0 };
    }
    let mut b: [Field; 48] = a;
    for i in 0..ROWS {
        for j in 0..COLS {
            b[i * COLS + j] = ind[j] * scaled[i] + (1 - ind[j]) * a[i * COLS + j];
        }
    }
    // Second gather (post-write) + cumsum chain.
    let mut out: [Field; 8] = [0; 8];
    let mut acc: Field = 0;
    for i in 0..ROWS {
        let mut cur: Field = b[i * COLS];
        for j in 1..COLS {
            cur = sel[j] * b[i * COLS + j] + (1 - sel[j]) * cur;
        }
        if i == 0 {
            acc = cur;
        } else {
            acc = qadd(acc, cur);
        }
        out[i] = acc;
    }
    out
}
