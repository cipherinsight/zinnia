// DS-1000 case 505 — SP1 guest program.
// Kernel:
//     arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//     result = np.sum(a) - np.sum(arr)
// a: Float[13, 13, 5], b: Integer[13, 13]; output: scalar Float.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<Vec<Vec<f64>>> = Vec::with_capacity(13);
    for _ in 0..13 {
        let mut plane: Vec<Vec<f64>> = Vec::with_capacity(13);
        for _ in 0..13 {
            let mut row: Vec<f64> = Vec::with_capacity(5);
            for _ in 0..5 {
                row.push(sp1_zkvm::io::read::<f64>());
            }
            plane.push(row);
        }
        a.push(plane);
    }
    let mut b: Vec<Vec<u64>> = Vec::with_capacity(13);
    for _ in 0..13 {
        let mut row: Vec<u64> = Vec::with_capacity(13);
        for _ in 0..13 {
            row.push(sp1_zkvm::io::read::<u64>());
        }
        b.push(row);
    }
    let claimed: f64 = sp1_zkvm::io::read::<f64>();

    let mut arr: Vec<Vec<f64>> = Vec::with_capacity(13);
    for i in 0..13 {
        let mut row: Vec<f64> = Vec::with_capacity(13);
        for j in 0..13 {
            let k = b[i][j] as usize;
            row.push(a[i][j][k]);
        }
        arr.push(row);
    }

    let mut sum_a: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            for k in 0..5 {
                sum_a += a[i][j][k];
            }
        }
    }
    let mut sum_arr: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            sum_arr += arr[i][j];
        }
    }
    let result: f64 = sum_a - sum_arr;

    assert!((result - claimed).abs() < 1e-6);
}
