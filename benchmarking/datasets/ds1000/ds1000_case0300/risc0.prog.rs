// DS-1000 case 300 — Risc0 guest program.
// np.percentile(a, p) on Float[20].

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [f64; 20] = [0.0; 20];
    for i in 0..20 {
        a[i] = env::read();
    }
    let p: i64 = env::read();
    let claim: f64 = env::read();

    let mut s: [f64; 20] = a;
    for i in 0..20 {
        let mut mi = i;
        for j in (i + 1)..20 {
            if s[j] < s[mi] {
                mi = j;
            }
        }
        let t = s[i];
        s[i] = s[mi];
        s[mi] = t;
    }

    let n: f64 = 20.0;
    let rank: f64 = (n - 1.0) * (p as f64) / 100.0;
    let lo: usize = rank as usize;
    let hi: usize = if lo + 1 < 20 { lo + 1 } else { lo };
    let frac: f64 = rank - (lo as f64);
    let result: f64 = s[lo] + frac * (s[hi] - s[lo]);

    assert!((result - claim).abs() < 1e-9);
}
