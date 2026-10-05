// DS-1000 case 365 — SP1 guest program.
// Kernel: np.insert(a, pos, element) on Float[100], with Integer pos and element.
// Result is Float[101]. Element is cast to float on insertion.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(100);
    for _ in 0..100 {
        a.push(sp1_zkvm::io::read::<f64>());
    }
    let pos: u64 = sp1_zkvm::io::read::<u64>();
    let element: i64 = sp1_zkvm::io::read::<i64>();
    let mut claimed: Vec<f64> = Vec::with_capacity(101);
    for _ in 0..101 {
        claimed.push(sp1_zkvm::io::read::<f64>());
    }

    let mut result: Vec<f64> = Vec::with_capacity(101);
    for i in 0..101usize {
        let v: f64 = if (i as u64) < pos {
            a[i]
        } else if (i as u64) == pos {
            element as f64
        } else {
            a[i - 1]
        };
        result.push(v);
    }

    for i in 0..101 {
        assert!((result[i] - claimed[i]).abs() < 1e-9);
    }
}
