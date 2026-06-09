// DS-1000 case 363 — SP1 guest program.
// Kernel: np.insert(a, pos, element) on Float[100] with scalar Integer pos & element.
// Output: Float[101]. Inserts `element` before index `pos`.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a = [0.0f64; 100];
    for i in 0..100 { a[i] = sp1_zkvm::io::read::<f64>(); }
    let pos: i64 = sp1_zkvm::io::read::<i64>();
    let element: i64 = sp1_zkvm::io::read::<i64>();
    let mut claimed = [0.0f64; 101];
    for i in 0..101 { claimed[i] = sp1_zkvm::io::read::<f64>(); }

    let pos_u: usize = pos as usize;
    let elem_f: f64 = element as f64;
    let mut result = [0.0f64; 101];
    for i in 0..101 {
        if i < pos_u {
            result[i] = a[i];
        } else if i == pos_u {
            result[i] = elem_f;
        } else {
            result[i] = a[i - 1];
        }
    }
    for i in 0..101 {
        assert!((result[i] - claimed[i]).abs() < 1e-3);
    }
}
