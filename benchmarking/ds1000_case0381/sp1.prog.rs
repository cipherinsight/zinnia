// DS-1000 case 381 — SP1 guest program.
// Kernel: np.argsort(a)[::-1][:len(a)] on Float[100]. Result is argsort indices reversed.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let n: usize = 100;
    let mut a: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        a.push(sp1_zkvm::io::read::<f64>());
    }
    let mut claimed: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        claimed.push(sp1_zkvm::io::read::<u64>());
    }

    // argsort: indices that sort a ascending (stable).
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 1..n {
        let key = idx[i];
        let mut j: usize = i;
        while j > 0 && a[idx[j - 1]] > a[key] {
            idx[j] = idx[j - 1];
            j -= 1;
        }
        idx[j] = key;
    }
    // Reverse.
    let mut rev: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        rev.push(idx[n - 1 - i] as u64);
    }
    // [:len(a)] is a no-op for len = n.

    for i in 0..n {
        assert_eq!(rev[i], claimed[i]);
    }
}
