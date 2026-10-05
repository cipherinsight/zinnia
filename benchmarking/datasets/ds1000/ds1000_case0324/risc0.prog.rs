// DS-1000 case 324 — Risc0 guest. cos(deg2rad(degree)).

use risc0_zkvm::guest::env;

fn main() {
    let degree: i64 = env::read();
    let claimed: f64 = env::read();

    let rad: f64 = (degree as f64) * std::f64::consts::PI / 180.0;
    let computed: f64 = rad.cos();

    assert!((computed - claimed).abs() < 1e-9);
}
