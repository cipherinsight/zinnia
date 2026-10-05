// DS-1000 case 507 — SP1 guest program.
// Kernel:
//     mask = im == 0
//     rows = np.flatnonzero((~mask).sum(axis=1))
//     cols = np.flatnonzero((~mask).sum(axis=0))
//     result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//
// im: Integer[5,6]. The bounding-box shape is data-dependent in NumPy; the
// driver picks a sample input that makes the bbox a fixed 3×3 (rows 1..3,
// cols 2..4). The guest still recomputes the bounds and asserts they match.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const R: usize = 5;
    const C: usize = 6;

    let mut im = [[0i64; C]; R];
    for i in 0..R {
        for j in 0..C {
            im[i][j] = sp1_zkvm::io::read::<i64>();
        }
    }
    let claimed_rmin: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_rmax: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_cmin: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_cmax: u64 = sp1_zkvm::io::read::<u64>();
    // Claimed result is a (claimed_rmax - claimed_rmin + 1) × (claimed_cmax - claimed_cmin + 1)
    // matrix, flattened row-major.
    let rh: usize = (claimed_rmax - claimed_rmin + 1) as usize;
    let ch: usize = (claimed_cmax - claimed_cmin + 1) as usize;
    let mut claimed_result: Vec<i64> = Vec::with_capacity(rh * ch);
    for _ in 0..(rh * ch) {
        claimed_result.push(sp1_zkvm::io::read::<i64>());
    }

    // Compute mask, row/col sums, then bounding box.
    let mut mask = [[false; C]; R];
    for i in 0..R {
        for j in 0..C {
            mask[i][j] = im[i][j] == 0;
        }
    }
    let mut row_sum = [0u64; R];
    for i in 0..R {
        let mut s: u64 = 0;
        for j in 0..C {
            if !mask[i][j] { s += 1; }
        }
        row_sum[i] = s;
    }
    let mut col_sum = [0u64; C];
    for j in 0..C {
        let mut s: u64 = 0;
        for i in 0..R {
            if !mask[i][j] { s += 1; }
        }
        col_sum[j] = s;
    }

    let mut rmin: u64 = u64::MAX;
    let mut rmax: u64 = 0;
    for i in 0..R {
        if row_sum[i] != 0 {
            if (i as u64) < rmin { rmin = i as u64; }
            if (i as u64) > rmax { rmax = i as u64; }
        }
    }
    let mut cmin: u64 = u64::MAX;
    let mut cmax: u64 = 0;
    for j in 0..C {
        if col_sum[j] != 0 {
            if (j as u64) < cmin { cmin = j as u64; }
            if (j as u64) > cmax { cmax = j as u64; }
        }
    }

    assert_eq!(rmin, claimed_rmin);
    assert_eq!(rmax, claimed_rmax);
    assert_eq!(cmin, claimed_cmin);
    assert_eq!(cmax, claimed_cmax);

    // Slice and compare with the claimed result.
    let mut idx = 0;
    for i in (claimed_rmin as usize)..=(claimed_rmax as usize) {
        for j in (claimed_cmin as usize)..=(claimed_cmax as usize) {
            assert_eq!(im[i][j], claimed_result[idx]);
            idx += 1;
        }
    }
}
