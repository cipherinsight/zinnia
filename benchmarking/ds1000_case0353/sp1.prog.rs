// DS-1000 case 353 — SP1 guest program.
// Kernel: rows of A not contained as a row of B.
//   dims = np.maximum(B.max(0), A.max(0))+1
//   output = A[~np.in1d(ravel_multi_index(A.T, dims), ravel_multi_index(B.T, dims))]
// A: Integer[2,3], B: Integer[7,3]. Output: 0..2 rows of length 3.
// We follow the algorithm exactly: compute dims, ravel-mul indices, in1d, filter.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // Read A (2x3), B (7x3) row-major.
    let mut a: [[i64; 3]; 2] = [[0; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            a[i][j] = sp1_zkvm::io::read::<i64>();
        }
    }
    let mut b: [[i64; 3]; 7] = [[0; 3]; 7];
    for i in 0..7 {
        for j in 0..3 {
            b[i][j] = sp1_zkvm::io::read::<i64>();
        }
    }

    // Read claimed result: prefixed by length, then length*3 ints.
    let n_claimed: u64 = sp1_zkvm::io::read::<u64>();
    let mut claimed: Vec<[i64; 3]> = Vec::with_capacity(n_claimed as usize);
    for _ in 0..n_claimed {
        let mut row = [0i64; 3];
        for j in 0..3 {
            row[j] = sp1_zkvm::io::read::<i64>();
        }
        claimed.push(row);
    }

    // dims[k] = max(A[:,k].max(), B[:,k].max()) + 1
    let mut dims = [0i64; 3];
    for k in 0..3 {
        let mut m = a[0][k];
        for i in 1..2 { if a[i][k] > m { m = a[i][k]; } }
        for i in 0..7 { if b[i][k] > m { m = b[i][k]; } }
        dims[k] = m + 1;
    }

    // ravel_multi_index of A.T (cols-as-rows view): for each row r of A,
    // index = ((a[r][0])*dims[1] + a[r][1])*dims[2] + a[r][2].
    let mut a_idx = [0i64; 2];
    for r in 0..2 {
        a_idx[r] = ((a[r][0] * dims[1]) + a[r][1]) * dims[2] + a[r][2];
    }
    let mut b_idx = [0i64; 7];
    for r in 0..7 {
        b_idx[r] = ((b[r][0] * dims[1]) + b[r][1]) * dims[2] + b[r][2];
    }

    // in1d(a_idx, b_idx)
    let mut in_b = [false; 2];
    for r in 0..2 {
        for s in 0..7 {
            if a_idx[r] == b_idx[s] { in_b[r] = true; }
        }
    }

    // output = A[~in_b]
    let mut out: Vec<[i64; 3]> = Vec::new();
    for r in 0..2 {
        if !in_b[r] {
            out.push([a[r][0], a[r][1], a[r][2]]);
        }
    }

    assert_eq!(out.len() as u64, n_claimed);
    for r in 0..out.len() {
        for j in 0..3 {
            assert_eq!(out[r][j], claimed[r][j]);
        }
    }
}
