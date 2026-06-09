// DS-1000 case 437 — SP1 guest.
// Kernel: mask = (a.min(axis=1, keepdims=1) == a) on Float[10, 5].
// Per row: row_min via running min; then 5 element==row_min bools.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { a.push(sp1_zkvm::io::read::<f64>()); }
    let mut claimed: Vec<u32> = Vec::with_capacity(50);
    for _ in 0..50 { claimed.push(sp1_zkvm::io::read::<u32>()); }

    for i in 0..10 {
        let mut row_min = a[i * 5];
        for j in 1..5 {
            let cur = a[i * 5 + j];
            if cur < row_min { row_min = cur; }
        }
        for j in 0..5 {
            let diff = a[i * 5 + j] - row_min;
            let eq: u32 = if diff.abs() < 0.001 { 1 } else { 0 };
            assert_eq!(eq, claimed[i * 5 + j]);
        }
    }
}
