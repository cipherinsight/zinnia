// DS-1000 case 438 — SP1 guest.
// Kernel: result = np.corrcoef(post, distance)[0][1] on Integer[4] × Integer[4].
// Pearson r = sum((x-xbar)(y-ybar)) / sqrt(sum((x-xbar)^2) * sum((y-ybar)^2)).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut post_int: Vec<i64> = Vec::with_capacity(4);
    for _ in 0..4 { post_int.push(sp1_zkvm::io::read::<i64>()); }
    let mut distance_int: Vec<i64> = Vec::with_capacity(4);
    for _ in 0..4 { distance_int.push(sp1_zkvm::io::read::<i64>()); }
    let claimed: f64 = sp1_zkvm::io::read::<f64>();

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
