// DS-1000 case 0321: literal translation of kernel.py (LITERAL_BASELINES.md).
// Linear scan, strict comparison keeps the first occurrence; fresh bounds per comparison.
// Unravel uses integer quotient/remainder; idx is a select of loop constants.
// @outputs: int[2]
fn main(a: [Field; 30]) -> pub [Field; 2] {
    noir_hw::check_inputs(a);
    let mut best = a[0];
    let mut idx: u32 = 0;
    for i in 1..30 {
        let better = noir_hw::gt(a[i], best);
        best = if better { a[i] } else { best };
        idx = if better { i } else { idx };
    }
    [(idx / 6) as Field, (idx % 6) as Field]
}
