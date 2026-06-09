// DS-1000 case 490 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 5;
    const M: usize = 4;
    const K: usize = 4;
    let mut arr: [[[f64; K]; M]; N] = [[[0.0; K]; M]; N];
    for i in 0..N {
        for j in 0..M {
            for k in 0..K {
                arr[i][j][k] = env::read();
            }
        }
    }
    let mut n1: [i64; 5] = [0; 5];
    for k in 0..5 { n1[k] = env::read(); }
    let mut n2: [i64; 5] = [0; 5];
    for k in 0..5 { n2[k] = env::read(); }
    let mut claimed: [[[f64; K]; M]; N] = [[[0.0; K]; M]; N];
    for i in 0..N {
        for j in 0..M {
            for k in 0..K {
                claimed[i][j][k] = env::read();
            }
        }
    }

    for i in 0..5 {
        let t1 = n1[i] as f64;
        let t2 = n2[i] as f64;
        let mut temp: [[f64; K]; M] = [[0.0; K]; M];
        for j in 0..M {
            for k in 0..K {
                temp[j][k] = arr[i][j][k];
            }
        }
        for j in 0..M {
            for k in 0..K {
                if temp[j][k] < t1 {
                    arr[i][j][k] = 0.0;
                }
            }
        }
        for j in 0..M {
            for k in 0..K {
                if temp[j][k] >= t2 {
                    arr[i][j][k] = 30.0;
                }
            }
        }
        for j in 0..M {
            for k in 0..K {
                if temp[j][k] >= t1 && temp[j][k] < t2 {
                    arr[i][j][k] += 5.0;
                }
            }
        }
    }

    for i in 0..N {
        for j in 0..M {
            for k in 0..K {
                assert!((arr[i][j][k] - claimed[i][j][k]).abs() <= 0.001);
            }
        }
    }
}
