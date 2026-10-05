// DS-1000 case 451 — SP1 guest.
// Kernel: `arr = np.zeros((20,10,10,2))`. Zinnia represents this as a single
// zero-constant exposure rather than 4000 individual values; we mirror that.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let claimed: f64 = sp1_zkvm::io::read::<f64>();
    let zero: f64 = 0.0;
    assert!((zero - claimed).abs() < 0.001);
}
