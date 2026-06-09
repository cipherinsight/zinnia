// DS-1000 case 0366 — SP1 guest program.
// Kernel: pos = pos - arange(5); a = np.insert(a, pos, element, axis=0).
// a Float[100,10], pos Int[5], element Float[5,10]. We use sorted pos so insertions are sequential.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const R: usize = 100;
    const C: usize = 10;
    const E: usize = 5;
    let mut a = vec![vec![0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = sp1_zkvm::io::read::<f64>(); } }
    let mut pos = [0i64; E];
    for k in 0..E { pos[k] = sp1_zkvm::io::read::<i64>(); }
    let mut element = [[0.0f64; C]; E];
    for k in 0..E { for j in 0..C { element[k][j] = sp1_zkvm::io::read::<f64>(); } }
    let adjusted: [i64; E] = std::array::from_fn(|k| pos[k] - (k as i64));
    let mut out: Vec<[f64; C]> = Vec::with_capacity(R + E);
    let mut k: usize = 0;
    for src in 0..R {
        while k < E && adjusted[k] as usize == src { out.push(element[k]); k += 1; }
        let mut row = [0.0f64; C];
        for j in 0..C { row[j] = a[src][j]; }
        out.push(row);
    }
    while k < E { out.push(element[k]); k += 1; }
    for row in &out {
        for v in row {
            let claim: f64 = sp1_zkvm::io::read::<f64>();
            assert!((v - claim).abs() < 1e-9);
        }
    }
}
