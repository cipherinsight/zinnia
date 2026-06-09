// DS-1000 case 325 — Risc0 guest.
// `int(sin(n) > sin(deg2rad(n)))`.

use risc0_zkvm::guest::env;

fn main() {
    let number: i64 = env::read();
    let claimed: u64 = env::read();

    let n_f: f64 = number as f64;
    let rad: f64 = n_f * std::f64::consts::PI / 180.0;
    let sin_deg: f64 = rad.sin();
    let sin_n: f64 = n_f.sin();
    let result: u64 = if sin_n > sin_deg { 1 } else { 0 };

    assert_eq!(result, claimed);
}
