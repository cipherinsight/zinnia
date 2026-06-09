// DS-1000 case 508 — SP1 guest program.
// Kernel:
//     B = np.argwhere(A)
//     (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
//     result = A[ystart:ystop, xstart:xstop]
//
// A: Integer[10,10]. Output shape is data-dependent; the driver picks an
// input that makes the bbox a fixed 4×4 (rows 2..5, cols 3..6).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 10;

    let mut a = [[0i64; N]; N];
    for i in 0..N {
        for j in 0..N {
            a[i][j] = sp1_zkvm::io::read::<i64>();
        }
    }
    let claimed_ystart: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_ystop: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_xstart: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_xstop: u64 = sp1_zkvm::io::read::<u64>();
    let rh: usize = (claimed_ystop - claimed_ystart) as usize;
    let ch: usize = (claimed_xstop - claimed_xstart) as usize;
    let mut claimed_result: Vec<i64> = Vec::with_capacity(rh * ch);
    for _ in 0..(rh * ch) {
        claimed_result.push(sp1_zkvm::io::read::<i64>());
    }

    // Build B = argwhere(A): list of (i, j) where A[i][j] != 0.
    let mut b: Vec<(u64, u64)> = Vec::new();
    for i in 0..N {
        for j in 0..N {
            if a[i][j] != 0 {
                b.push((i as u64, j as u64));
            }
        }
    }
    // B.min(0) and B.max(0) — componentwise.
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
