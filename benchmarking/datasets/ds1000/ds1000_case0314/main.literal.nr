// DS-1000 case 0314: literal translation of kernel.py (LITERAL_BASELINES.md).
// Separate argmax, unravel, minimum, scatter, argmax and unravel; strict scans preserve first ties.
// @outputs: int[2]
fn argmax(a: [Field; 48]) -> u32 {
    let mut best = a[0];
    let mut idx: u32 = 0;
    for i in 1..48 {
        let better = noir_hw::gt(a[i], best);
        best = if better { a[i] } else { best };
        idx = if better { i } else { idx };
    }
    idx
}
fn main(a: [Field; 48]) -> pub [Field; 2] {
    noir_hw::check_inputs(a);
    let mut a = a;
    let flat = argmax(a);
    // flat is a select of loop constants, so native integer arithmetic is safe.
    let row = flat / 6;
    let col = flat % 6;
    let mut low = a[0];
    for i in 1..48 { low = noir_hw::min(low, a[i]); }
    for i in 0..8 {
        for j in 0..6 {
            a[i * 6 + j] = if (row == i) & (col == j) { low } else { a[i * 6 + j] };
        }
    }
    let flat = argmax(a);
    [(flat / 6) as Field, (flat % 6) as Field]
}
