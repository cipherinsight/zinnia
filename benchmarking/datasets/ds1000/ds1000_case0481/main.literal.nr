// DS-1000 case 0481: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compaction uses prefix counts and an equality test for every input/output pair, with zero padding.
// @outputs: dynamic int[9x1]
fn main(x: [Field; 9], y: [Field; 9], a: Field, b: Field) -> pub [Field; 10] {
    noir_hw::check_inputs(x); noir_hw::check_inputs(y);
    noir_hw::check_input(a); noir_hw::check_input(b);
    let mut xm = [false; 9]; let mut ym = [false; 9];
    for i in 0..9 { xm[i] = x[i] == a; }
    for i in 0..9 { ym[i] = y[i] == b; }
    let mut keep = [false; 9];
    for i in 0..9 { keep[i] = xm[i] & ym[i]; }
    let mut before = [0; 9];
    let mut count: Field = 0;
    for i in 0..9 { before[i] = count; count += keep[i] as Field; }
    let mut result = [0; 10]; result[0] = count;
    for j in 0..9 {
        let mut value: Field = 0;
        for i in 0..9 { value += ((keep[i] & (before[i] == j as Field)) as Field) * (i as Field); }
        result[j+1] = value;
    }
    result
}
