// DS-1000 case 349 — SP1 guest program.
// Kernel: `a.shape[1]` on Float[8, 6] → scalar integer 6.
// Mirrors Zinnia IR: reads 48 floats (witnessed but unused) and emits 6.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        a.push(row);
    }
    let claimed: u64 = sp1_zkvm::io::read::<u64>();

    // Touch all 48 cells to mirror Zinnia's ReadFloatIR witness reads.
    let mut sink: f64 = 0.0;
    for i in 0..8 {
        for j in 0..6 {
            sink += a[i][j] * 0.0;
        }
    }
    assert_eq!(sink, 0.0);

    let result: u64 = 6;
    assert_eq!(result, claimed);
}
