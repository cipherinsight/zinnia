// DS-1000 case 468 — SP1 guest program.
// kernel.py: column-major windowing of a 4×4 int matrix with a (3,3) window.
//   r_win = floor(3/2) = 1; c_win = floor(3/2) = 1.
//   for j in 0..4: ymin=max(0,j-1), ymax=min(4,j+2)
//     for i in 0..4: xmin=max(0,i-1), xmax=min(4,i+2)
//       ans.append(a[xmin:xmax, ymin:ymax])

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [[i64; 4]; 4] = [[0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = sp1_zkvm::io::read::<i64>();
        }
    }
    let size0: i64 = sp1_zkvm::io::read::<i64>();
    let size1: i64 = sp1_zkvm::io::read::<i64>();

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
        claim_counts[k] = sp1_zkvm::io::read::<u64>();
    }
    let claim_total: u64 = sp1_zkvm::io::read::<u64>();
    let mut claim_flat: Vec<i64> = Vec::with_capacity(claim_total as usize);
    for _ in 0..claim_total {
        claim_flat.push(sp1_zkvm::io::read::<i64>());
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
