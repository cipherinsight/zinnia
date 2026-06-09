// DS-1000 case 420 — chained np.where on a scalar integer.
// kernel.py: np.where(x<x_min, x_min, np.where(x>x_max, x_max, 3*x*x - 2*x*x*x))
// x_min, x_max are undefined globals in the kernel; we fix them at host as
// (x_min, x_max) = (0, 10). With x = 5: 3*25 - 2*125 = -175.

#[executable]
pub fn main() {
    let x: i128 = 5;
    let x_min: i128 = 0;
    let x_max: i128 = 10;
    let claim: i128 = -175;
    let inner: i128 = 3 * x * x - 2 * x * x * x;
    let mid: i128 = if x > x_max { x_max } else { inner };
    let out: i128 = if x < x_min { x_min } else { mid };
    assert!(out == claim);
}
