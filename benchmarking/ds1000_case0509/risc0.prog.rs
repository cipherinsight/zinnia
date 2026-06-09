// DS-1000 case 509 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 10;

    let mut im = [[0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            im[i][j] = env::read();
        }
    }
    let claimed_rmin: u64 = env::read();
    let claimed_rmax: u64 = env::read();
    let claimed_cmin: u64 = env::read();
    let claimed_cmax: u64 = env::read();
    let rh: usize = (claimed_rmax - claimed_rmin + 1) as usize;
    let ch: usize = (claimed_cmax - claimed_cmin + 1) as usize;
    let mut claimed_result: Vec<f64> = Vec::with_capacity(rh * ch);
    for _ in 0..(rh * ch) {
        claimed_result.push(env::read());
    }

    let mut mask = [[false; N]; N];
    for i in 0..N {
        for j in 0..N {
            mask[i][j] = im[i][j] == 0.0;
        }
    }
    let mut row_sum = [0u64; N];
    for i in 0..N {
        let mut s: u64 = 0;
        for j in 0..N {
            if mask[i][j] { s += 1; }
        }
        row_sum[i] = s;
    }
    let mut col_sum = [0u64; N];
    for j in 0..N {
        let mut s: u64 = 0;
        for i in 0..N {
            if mask[i][j] { s += 1; }
        }
        col_sum[j] = s;
    }

    let mut rmin: u64 = u64::MAX;
    let mut rmax: u64 = 0;
    for i in 0..N {
        if row_sum[i] != 0 {
            if (i as u64) < rmin { rmin = i as u64; }
            if (i as u64) > rmax { rmax = i as u64; }
        }
    }
    let mut cmin: u64 = u64::MAX;
    let mut cmax: u64 = 0;
    for j in 0..N {
        if col_sum[j] != 0 {
            if (j as u64) < cmin { cmin = j as u64; }
            if (j as u64) > cmax { cmax = j as u64; }
        }
    }

    assert_eq!(rmin, claimed_rmin);
    assert_eq!(rmax, claimed_rmax);
    assert_eq!(cmin, claimed_cmin);
    assert_eq!(cmax, claimed_cmax);

    let mut idx = 0;
    for i in (claimed_rmin as usize)..=(claimed_rmax as usize) {
        for j in (claimed_cmin as usize)..=(claimed_cmax as usize) {
            assert!((im[i][j] - claimed_result[idx]).abs() < 1e-6);
            idx += 1;
        }
    }
}
