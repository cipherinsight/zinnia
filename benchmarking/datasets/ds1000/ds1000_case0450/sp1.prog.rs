// DS-1000 case 450 — SP1 guest.
// Kernel: `a[:, second.reshape(-1,1), third]` with a: Float[7,8,9],
// second: Integer[2], third: Integer[2]. Result: Float[7, 2, 2] where
// result[i][p][q] = a[i][second[p]][third[q]].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [[[f64; 9]; 8]; 7] = [[[0.0; 9]; 8]; 7];
    for i in 0..7 {
        for j in 0..8 {
            for k in 0..9 {
                a[i][j][k] = sp1_zkvm::io::read::<f64>();
            }
        }
    }
    let mut second: [i64; 2] = [0; 2];
    for p in 0..2 { second[p] = sp1_zkvm::io::read::<i64>(); }
    let mut third: [i64; 2] = [0; 2];
    for q in 0..2 { third[q] = sp1_zkvm::io::read::<i64>(); }
    let mut claimed: [[[f64; 2]; 2]; 7] = [[[0.0; 2]; 2]; 7];
    for i in 0..7 {
        for p in 0..2 {
            for q in 0..2 {
                claimed[i][p][q] = sp1_zkvm::io::read::<f64>();
            }
        }
    }

    let mut result: [[[f64; 2]; 2]; 7] = [[[0.0; 2]; 2]; 7];
    for i in 0..7 {
        for p in 0..2 {
            for q in 0..2 {
                let j = second[p] as usize;
                let k = third[q] as usize;
                result[i][p][q] = a[i][j][k];
            }
        }
    }

    for i in 0..7 {
        for p in 0..2 {
            for q in 0..2 {
                assert!((result[i][p][q] - claimed[i][p][q]).abs() < 0.001);
            }
        }
    }
}
