// DS-1000 case 325 — SP1 guest.
// `int(sin(n) > sin(deg2rad(n)))` for integer n.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let number: i64 = sp1_zkvm::io::read::<i64>();
    let claimed: u64 = sp1_zkvm::io::read::<u64>();

    let n_f: f64 = number as f64;
    let rad: f64 = n_f * std::f64::consts::PI / 180.0;
    let sin_deg: f64 = rad.sin();      // sin(deg2rad(n))
    let sin_n: f64 = n_f.sin();        // sin(n) — n interpreted as radians
    let result: u64 = if sin_n > sin_deg { 1 } else { 0 };

    assert_eq!(result, claimed);
}
