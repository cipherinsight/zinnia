// DS-1000 case 0442: literal translation of kernel.py (LITERAL_BASELINES.md).
// Membership is an equality/OR scan. Compaction uses prefix counts and an equality test for every input/output pair.
// @outputs: dynamic int[20x1]
fn main(A: [Field; 20], B: [Field; 3]) -> pub [Field; 21] {
    noir_hw::check_inputs(A);
    noir_hw::check_inputs(B);
    let mut membership = [false; 20];
    for i in 0..20 {
        let mut found = false;
        for j in 0..3 { found = found | (A[i] == B[j]); }
        membership[i] = found;
    }
    let mut keep = [false; 20];
    for i in 0..20 { keep[i] = !membership[i]; }
    let mut prefix: [Field; 21] = [0; 21];
    for i in 0..20 { prefix[i + 1] = prefix[i] + keep[i] as Field; }
    let mut result = [0; 21];
    result[0] = prefix[20];
    for j in 0..20 {
        let mut sum: Field = 0;
        for i in 0..20 {
            let active = keep[i] & (prefix[i] == j as Field);
            sum += (active as Field) * A[i];
        }
        result[j + 1] = sum;
    }
    result
}
