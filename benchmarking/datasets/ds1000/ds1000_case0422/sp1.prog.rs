// DS-1000 case 422 — SP1 guest.
// Kernel: c = np.hstack((b[1:], b));  result = np.correlate(a, c, mode='valid')
// With len(a)=50, len(b)=50, len(c)=99 → len(result)=50.
// Formula (verified against numpy): out[k] = sum_{n=0..49} a[n] * c[n + 49 - k].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { a.push(sp1_zkvm::io::read::<f64>()); }
    let mut b: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { b.push(sp1_zkvm::io::read::<f64>()); }
    let mut claimed: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { claimed.push(sp1_zkvm::io::read::<f64>()); }

    // c = hstack((b[1:], b)) → length 99.
    let mut c: Vec<f64> = Vec::with_capacity(99);
    for i in 1..50 { c.push(b[i]); }
    for i in 0..50 { c.push(b[i]); }

    // np.correlate(a, c, 'valid'): out[k] = sum_n a[n] * c[n + 49 - k].
    for k in 0..50 {
        let mut s: f64 = 0.0;
        for n in 0..50 {
            s += a[n] * c[n + 49 - k];
        }
        assert!((s - claimed[k]).abs() < 0.001);
    }
}
