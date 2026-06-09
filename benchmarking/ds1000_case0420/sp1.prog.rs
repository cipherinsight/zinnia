// DS-1000 case 420 — SP1 guest program.
// Kernel: np.where(x<x_min, x_min, np.where(x>x_max, x_max, 3*x*x - 2*x*x*x))
// x_min and x_max are undefined globals in the kernel; we fix them at host as
// x_min=0, x_max=10. Choose x=5: 3*25 - 2*125 = -175 ∈ [x_min, x_max] range, so
// the inner branch is taken. Expected = -175.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let x: i64 = sp1_zkvm::io::read::<i64>();
    let x_min: i64 = sp1_zkvm::io::read::<i64>();
    let x_max: i64 = sp1_zkvm::io::read::<i64>();
    let claim: i64 = sp1_zkvm::io::read::<i64>();
    let inner = 3 * x * x - 2 * x * x * x;
    let mid = if x > x_max { x_max } else { inner };
    let out = if x < x_min { x_min } else { mid };
    assert_eq!(out, claim);
}
