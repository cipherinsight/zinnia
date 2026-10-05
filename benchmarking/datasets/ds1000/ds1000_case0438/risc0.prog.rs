// DS-1000 case 438 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut post_int: Vec<i64> = Vec::with_capacity(4);
    for _ in 0..4 { post_int.push(env::read()); }
    let mut distance_int: Vec<i64> = Vec::with_capacity(4);
    for _ in 0..4 { distance_int.push(env::read()); }
    let claimed: f64 = env::read();

    let xs: Vec<f64> = post_int.iter().map(|v| *v as f64).collect();
    let ys: Vec<f64> = distance_int.iter().map(|v| *v as f64).collect();

    let mut xsum: f64 = 0.0;
    let mut ysum: f64 = 0.0;
    for k in 0..4 { xsum += xs[k]; ysum += ys[k]; }
    let xbar = xsum / 4.0;
    let ybar = ysum / 4.0;

    let mut cov: f64 = 0.0;
    let mut vx: f64 = 0.0;
    let mut vy: f64 = 0.0;
    for k in 0..4 {
        let dx = xs[k] - xbar;
        let dy = ys[k] - ybar;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    let r = cov / denom;

    assert!((r - claimed).abs() < 0.005);
}
