// DS-1000 case 490 — SP1 guest program.
// kernel.py:
//     for a, t1, t2 in zip(arr, n1, n2):
//         temp = a.copy()
//         a[np.where(temp < t1)] = 0
//         a[np.where(temp >= t2)] = 30
//         a[np.logical_and(temp >= t1, temp < t2)] += 5
//     return arr
// Sample: N=5, M=4, K=4 (within N<=64, M<=64, K<=64).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 5;
    const M: usize = 4;
    const K: usize = 4;
    let mut arr: [[[f64; K]; M]; N] = [[[0.0; K]; M]; N];
    for i in 0..N {
        for j in 0..M {
            for k in 0..K {
                arr[i][j][k] = sp1_zkvm::io::read::<f64>();
            }
        }
    }
    let mut n1: [i64; 5] = [0; 5];
    for k in 0..5 { n1[k] = sp1_zkvm::io::read::<i64>(); }
    let mut n2: [i64; 5] = [0; 5];
    for k in 0..5 { n2[k] = sp1_zkvm::io::read::<i64>(); }
    let mut claimed: [[[f64; K]; M]; N] = [[[0.0; K]; M]; N];
    for i in 0..N {
        for j in 0..M {
            for k in 0..K {
                claimed[i][j][k] = sp1_zkvm::io::read::<f64>();
            }
        }
    }

    // 5 iterations (length of n1/n2).
    for i in 0..5 {
        let t1 = n1[i] as f64;
        let t2 = n2[i] as f64;
        // temp = a.copy()
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
