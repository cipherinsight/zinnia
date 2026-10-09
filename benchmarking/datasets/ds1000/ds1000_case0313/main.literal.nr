// DS-1000 case 0313: literal translation of kernel.py (LITERAL_BASELINES.md).
// Linear scan, strict comparison keeps the first occurrence; fresh bounds per comparison.
// @outputs: int[1]
fn main(a: [Field; 48]) -> pub Field {
    noir_hw::check_inputs(a);
    let mut best = a[0];
    let mut idx: u32 = 0;
    for i in 1..48 {
        let better = noir_hw::gt(a[i], best);
        best = if better { a[i] } else { best };
        idx = if better { i } else { idx };
    }
    idx as Field
}
