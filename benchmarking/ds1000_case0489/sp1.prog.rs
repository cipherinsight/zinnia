// DS-1000 case 0489 — SP1 guest program.
// Kernel: snapshot result = arr; arr[result<-10]=0; arr[result>=15]=30;
// arr[(-10 <= result) & (result < 15)] += 5. We use concrete dims 2×2×2 (8 cells).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 8; // 2 * 2 * 2
    let mut arr: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N { arr.push(sp1_zkvm::io::read::<f64>()); }
    let result: Vec<f64> = arr.clone();
    for i in 0..N {
        if result[i] < -10.0 {
            arr[i] = 0.0;
        } else if result[i] >= 15.0 {
            arr[i] = 30.0;
        } else {
            arr[i] += 5.0;
        }
    }
    for i in 0..N {
        let claim: f64 = sp1_zkvm::io::read::<f64>();
        assert!((arr[i] - claim).abs() < 1e-9);
    }
}
