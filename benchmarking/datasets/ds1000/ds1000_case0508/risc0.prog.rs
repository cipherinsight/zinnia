// DS-1000 case 508 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 10;

    let mut a = [[0i64; N]; N];
    for i in 0..N {
        for j in 0..N {
            a[i][j] = env::read();
        }
    }
    let claimed_ystart: u64 = env::read();
    let claimed_ystop: u64 = env::read();
    let claimed_xstart: u64 = env::read();
    let claimed_xstop: u64 = env::read();
    let rh: usize = (claimed_ystop - claimed_ystart) as usize;
    let ch: usize = (claimed_xstop - claimed_xstart) as usize;
    let mut claimed_result: Vec<i64> = Vec::with_capacity(rh * ch);
    for _ in 0..(rh * ch) {
        claimed_result.push(env::read());
    }

    let mut b: Vec<(u64, u64)> = Vec::new();
    for i in 0..N {
        for j in 0..N {
            if a[i][j] != 0 {
                b.push((i as u64, j as u64));
            }
        }
    }
    let mut ystart: u64 = u64::MAX;
    let mut xstart: u64 = u64::MAX;
    let mut ystop_m1: u64 = 0;
    let mut xstop_m1: u64 = 0;
    for &(y, x) in b.iter() {
        if y < ystart { ystart = y; }
        if x < xstart { xstart = x; }
        if y > ystop_m1 { ystop_m1 = y; }
        if x > xstop_m1 { xstop_m1 = x; }
    }
    let ystop: u64 = ystop_m1 + 1;
    let xstop: u64 = xstop_m1 + 1;

    assert_eq!(ystart, claimed_ystart);
    assert_eq!(ystop, claimed_ystop);
    assert_eq!(xstart, claimed_xstart);
    assert_eq!(xstop, claimed_xstop);

    let mut idx = 0;
    for i in (claimed_ystart as usize)..(claimed_ystop as usize) {
        for j in (claimed_xstart as usize)..(claimed_xstop as usize) {
            assert_eq!(a[i][j], claimed_result[idx]);
            idx += 1;
        }
    }
}
