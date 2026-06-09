// DS-1000 case 335 — SP1 guest.
// `np.max([a, b, c], axis=0)` over three Float[50] arrays.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { a.push(sp1_zkvm::io::read::<f64>()); }
    let mut b: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { b.push(sp1_zkvm::io::read::<f64>()); }
    let mut c: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { c.push(sp1_zkvm::io::read::<f64>()); }
    let mut claimed: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { claimed.push(sp1_zkvm::io::read::<f64>()); }

    for k in 0..50usize {
        let ab = if a[k] > b[k] { a[k] } else { b[k] };
        let m = if ab > c[k] { ab } else { c[k] };
        assert!((m - claimed[k]).abs() < 1e-9);
    }
}
