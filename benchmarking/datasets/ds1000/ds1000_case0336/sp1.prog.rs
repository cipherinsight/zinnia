// DS-1000 case 336 — SP1 guest. np.diag(np.fliplr(a)) on Float[8, 8].
// result[i] == a[i][7 - i].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            a[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    let mut result = [0.0f64; 8];
    for i in 0..8 { result[i] = sp1_zkvm::io::read::<f64>(); }

    for i in 0..8 {
        let computed = a[i][7 - i];
        assert!((result[i] - computed).abs() < 1e-12);
    }
}
