// DS-1000 case 468 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [[i64; 4]; 4] = [[0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = env::read();
        }
    }
    let size0: i64 = env::read();
    let size1: i64 = env::read();

    let r_win: i64 = size0 / 2;
    let c_win: i64 = size1 / 2;
    let x: i64 = 4;
    let y: i64 = 4;

    let mut flat: Vec<i64> = Vec::new();
    let mut counts: Vec<u64> = Vec::new();
    for j in 0..y {
        let ymin = if j - c_win > 0 { j - c_win } else { 0 };
        let ymax = if j + c_win + 1 < y { j + c_win + 1 } else { y };
        for i in 0..x {
            let xmin = if i - r_win > 0 { i - r_win } else { 0 };
            let xmax = if i + r_win + 1 < x { i + r_win + 1 } else { x };
            let mut cnt: u64 = 0;
            for ii in xmin..xmax {
                for jj in ymin..ymax {
                    flat.push(a[ii as usize][jj as usize]);
                    cnt += 1;
                }
            }
            counts.push(cnt);
        }
    }

    let mut claim_counts: [u64; 16] = [0; 16];
    for k in 0..16 {
        claim_counts[k] = env::read();
    }
    let claim_total: u64 = env::read();
    let mut claim_flat: Vec<i64> = Vec::with_capacity(claim_total as usize);
    for _ in 0..claim_total {
        claim_flat.push(env::read());
    }

    assert_eq!(counts.len() as u64, 16);
    for k in 0..16 {
        assert_eq!(counts[k], claim_counts[k]);
    }
    assert_eq!(flat.len() as u64, claim_total);
    for k in 0..(claim_total as usize) {
        assert_eq!(flat[k], claim_flat[k]);
    }
}
