// DS-1000 case 324 — SP1 guest. cos(deg2rad(degree)).
// Idiomatic Rust: native f64::cos.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let degree: i64 = sp1_zkvm::io::read::<i64>();
    let claimed: f64 = sp1_zkvm::io::read::<f64>();

    let rad: f64 = (degree as f64) * std::f64::consts::PI / 180.0;
    let computed: f64 = rad.cos();

    assert!((computed - claimed).abs() < 1e-9);
}
