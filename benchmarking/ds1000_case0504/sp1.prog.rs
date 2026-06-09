// DS-1000 case 504 — SP1 guest program.
// Kernel:
//     arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//     result = np.sum(arr)
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

    let mut result: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            result += arr[i][j];
        }
    }

    assert!((result - claimed).abs() < 1e-6);
}
