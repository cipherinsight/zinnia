// DS-1000 case 300 — SP1 guest program.
// Kernel: np.percentile(a, p) on Float[20] with integer p.
// Linear interpolation between order statistics: rank = (n-1)*p/100,
// result = a_sorted[floor(rank)] + frac * (a_sorted[ceil(rank)] - a_sorted[floor(rank)]).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [f64; 20] = [0.0; 20];
    for i in 0..20 {
        a[i] = sp1_zkvm::io::read::<f64>();
    }
    let p: i64 = sp1_zkvm::io::read::<i64>();
    let claim: f64 = sp1_zkvm::io::read::<f64>();

    // Sort ascending (selection sort, no shortcuts).
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
    let lo: usize = rank as usize;          // floor for non-negative rank
    let hi: usize = if lo + 1 < 20 { lo + 1 } else { lo };
    let frac: f64 = rank - (lo as f64);
    let result: f64 = s[lo] + frac * (s[hi] - s[lo]);

    assert!((result - claim).abs() < 1e-9);
}
