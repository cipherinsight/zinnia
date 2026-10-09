// DS-1000 case 0437: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compute all row extrema by pairwise scan, then broadcast equality; fresh bounds at each comparison.
// @outputs: bool[50]
fn main(a: [Field; 50]) -> pub [bool; 50] {
    noir_hw::check_inputs(a);
    let mut extrema = [0; 10];
    for row in 0..10 {
        let mut best = a[row * 5];
        for col in 1..5 { best = noir_hw::min(best, a[row * 5 + col]); }
        extrema[row] = best;
    }
    let mut mask = [false; 50];
    for row in 0..10 {
        for col in 0..5 { mask[row * 5 + col] = extrema[row] == a[row * 5 + col]; }
    }
    mask
}
