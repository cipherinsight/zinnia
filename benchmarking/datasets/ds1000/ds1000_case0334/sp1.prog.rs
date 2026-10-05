// DS-1000 case 334 — SP1 guest. np.mean([a, b, c], axis=0) over Float[50].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a = [0.0f64; 50];
    let mut b = [0.0f64; 50];
    let mut c = [0.0f64; 50];
    for i in 0..50 { a[i] = sp1_zkvm::io::read::<f64>(); }
    for i in 0..50 { b[i] = sp1_zkvm::io::read::<f64>(); }
    for i in 0..50 { c[i] = sp1_zkvm::io::read::<f64>(); }
    let mut result = [0.0f64; 50];
    for i in 0..50 { result[i] = sp1_zkvm::io::read::<f64>(); }

    for i in 0..50 {
        let computed = (a[i] + b[i] + c[i]) / 3.0;
        assert!((computed - result[i]).abs() < 1e-9);
    }
}
