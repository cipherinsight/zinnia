// Input is a 20x1 column; Boolean indexing then reshape(-1, 1) retains
// the same row-major values. The public length counts rows of one value.
// DS-1000 case 0463: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compaction uses prefix counts and an equality test for every input/output pair, with zero padding.
// @outputs: dynamic int[20x1]
fn main(a: [Field; 20]) -> pub [Field; 21] {
    noir_hw::check_inputs(a);
    let mut keep = [true; 20];
    for i in 1..20 { keep[i] = a[i] != a[i-1]; }
    let mut nz = [false; 20];
    for i in 0..20 { nz[i] = a[i] != 0; }
    for i in 0..20 { keep[i] = keep[i] & nz[i]; }
    let mut before = [0; 20];
    let mut count: Field = 0;
    for i in 0..20 { before[i] = count; count += keep[i] as Field; }
    let mut result = [0; 21]; result[0] = count;
    for j in 0..20 {
        let mut value: Field = 0;
        for i in 0..20 { value += ((keep[i] & (before[i] == j as Field)) as Field) * a[i]; }
        result[j+1] = value;
    }
    result
}
