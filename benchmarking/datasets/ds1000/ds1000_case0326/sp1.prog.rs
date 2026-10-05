// DS-1000 case 326 — SP1 guest.
// `np.degrees(np.arcsin(value))`. Asserts -1 ≤ value ≤ 1 first.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let value: f64 = sp1_zkvm::io::read::<f64>();
    let claimed: f64 = sp1_zkvm::io::read::<f64>();

    assert!(value >= -1.0);
    assert!(value <= 1.0);

    let asin_v: f64 = value.asin();
    let computed: f64 = asin_v * 180.0 / std::f64::consts::PI;

    assert!((computed - claimed).abs() < 1e-9);
}
