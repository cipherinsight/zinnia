// DS-1000 case 492 — SP1 guest program.
// kernel.py: (~np.isclose(s1, s2, equal_nan=True)).sum() for Float[6,1] arrays.
// np.isclose default tolerances: atol = 1e-8, rtol = 1e-5.
// Zinnia has no NaN support, so equal_nan changes nothing for finite floats.
// Count indices where |s1 - s2| > atol + rtol * |s2|.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 6;
    const ATOL: f64 = 1.0e-8;
    const RTOL: f64 = 1.0e-5;
    let mut s1: [f64; N] = [0.0; N];
    for k in 0..N { s1[k] = sp1_zkvm::io::read::<f64>(); }
    let mut s2: [f64; N] = [0.0; N];
    for k in 0..N { s2[k] = sp1_zkvm::io::read::<f64>(); }
    let claimed: i64 = sp1_zkvm::io::read::<i64>();

    let mut count: i64 = 0;
    for k in 0..N {
        let diff = (s1[k] - s2[k]).abs();
        let bound = ATOL + RTOL * s2[k].abs();
        let close: bool = diff <= bound;
        let not_close: i64 = if close { 0 } else { 1 };
        count += not_close;
    }

    assert_eq!(count, claimed);
}
