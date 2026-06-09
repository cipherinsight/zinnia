// DS-1000 case 0354 — SP1 guest program.
// Kernel: dims = max(B.max(0), A.max(0)) + 1
//         flatA = ravel_multi_index(A.T, dims); same for flatB.
//         result = A[~in1d(flatA, flatB)].
// A: Int[2,3], B: Int[7,3].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const NA: usize = 2;
    const NB: usize = 7;
    const K: usize = 3;
    let mut a = [[0i64; K]; NA];
    for i in 0..NA { for j in 0..K { a[i][j] = sp1_zkvm::io::read::<i64>(); } }
    let mut b = [[0i64; K]; NB];
    for i in 0..NB { for j in 0..K { b[i][j] = sp1_zkvm::io::read::<i64>(); } }
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
    let claim_count: u32 = sp1_zkvm::io::read::<u32>();
    assert_eq!(count, claim_count);
    for i in 0..NA {
        if keep[i] {
            for j in 0..K {
                let claim: i64 = sp1_zkvm::io::read::<i64>();
                assert_eq!(a[i][j], claim);
            }
        }
    }
}
