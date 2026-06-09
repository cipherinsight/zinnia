// DS-1000 case 364 — SP1 guest program.
// Kernel: np.insert(a, pos, element, axis=0) on Float[100, 10] with
//   element: Float[10]. Output: Float[101, 10].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a = [[0.0f64; 10]; 100];
    for i in 0..100 { for j in 0..10 { a[i][j] = sp1_zkvm::io::read::<f64>(); } }
    let pos: i64 = sp1_zkvm::io::read::<i64>();
    let mut element = [0.0f64; 10];
    for j in 0..10 { element[j] = sp1_zkvm::io::read::<f64>(); }
    let mut claimed = [[0.0f64; 10]; 101];
    for i in 0..101 { for j in 0..10 { claimed[i][j] = sp1_zkvm::io::read::<f64>(); } }

    let pos_u: usize = pos as usize;
    let mut result = [[0.0f64; 10]; 101];
    for i in 0..101 {
        if i < pos_u {
            for j in 0..10 { result[i][j] = a[i][j]; }
        } else if i == pos_u {
            for j in 0..10 { result[i][j] = element[j]; }
        } else {
            for j in 0..10 { result[i][j] = a[i - 1][j]; }
        }
    }
    for i in 0..101 { for j in 0..10 {
        assert!((result[i][j] - claimed[i][j]).abs() < 1e-3);
    } }
}
