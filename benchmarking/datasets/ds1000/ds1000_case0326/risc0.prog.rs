// DS-1000 case 326 — Risc0 guest.
// `np.degrees(np.arcsin(value))`.

use risc0_zkvm::guest::env;

fn main() {
    let value: f64 = env::read();
    let claimed: f64 = env::read();

    assert!(value >= -1.0);
    assert!(value <= 1.0);

    let asin_v: f64 = value.asin();
    let computed: f64 = asin_v * 180.0 / std::f64::consts::PI;

    assert!((computed - claimed).abs() < 1e-9);
}
