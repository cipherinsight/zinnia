// DS-1000 case 435 — SP1 guest.
// Kernel: a[1, :] = 0; a[:, 0] = 0 on Integer[4, 4]. Static indices.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<i64> = Vec::with_capacity(16);
    for _ in 0..16 { a.push(sp1_zkvm::io::read::<i64>()); }
    let mut claimed: Vec<i64> = Vec::with_capacity(16);
    for _ in 0..16 { claimed.push(sp1_zkvm::io::read::<i64>()); }

    for i in 0..4 {
        for j in 0..4 {
            let out: i64 = if i == 1 || j == 0 { 0 } else { a[i * 4 + j] };
            assert_eq!(out, claimed[i * 4 + j]);
        }
    }
}
