// DS-1000 case 0354 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const NA: usize = 2;
    const NB: usize = 7;
    const K: usize = 3;
    let mut a = [[0i64; K]; NA];
    for i in 0..NA { for j in 0..K { a[i][j] = env::read(); } }
    let mut b = [[0i64; K]; NB];
    for i in 0..NB { for j in 0..K { b[i][j] = env::read(); } }
    let mut dims = [0i64; K];
    for j in 0..K {
        let mut m = a[0][j];
        for i in 1..NA { if a[i][j] > m { m = a[i][j]; } }
        for i in 0..NB { if b[i][j] > m { m = b[i][j]; } }
        dims[j] = m + 1;
    }
    let ravel = |row: [i64; K]| -> i64 {
        let mut idx: i64 = 0;
        for j in 0..K { idx = idx * dims[j] + row[j]; }
        idx
    };
    let flat_b: [i64; NB] = std::array::from_fn(|i| ravel(b[i]));
    let mut keep = [false; NA];
    let mut count: u32 = 0;
    for i in 0..NA {
        let f = ravel(a[i]);
        let in_b = flat_b.iter().any(|&x| x == f);
        keep[i] = !in_b;
        if keep[i] { count += 1; }
    }
    let claim_count: u32 = env::read();
    assert_eq!(count, claim_count);
    for i in 0..NA {
        if keep[i] {
            for j in 0..K {
                let claim: i64 = env::read();
                assert_eq!(a[i][j], claim);
            }
        }
    }
}
