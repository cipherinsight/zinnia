// DS-1000 case 0444: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compute each comparison array independently in expression order. Compaction uses prefix counts and an equality test for every input/output pair.
// @outputs: dynamic int[20x1]
fn main(A: [Field; 20], B: [Field; 3]) -> pub [Field; 21] {
    noir_hw::check_inputs(A);
    noir_hw::check_inputs(B);
    let mut above0 = [false; 20];
    for i in 0..20 { above0[i] = noir_hw::gt(A[i], B[0]); }
    let mut below1 = [false; 20];
    for i in 0..20 { below1[i] = noir_hw::lt(A[i], B[1]); }
    let mut first = [false; 20];
    for i in 0..20 { first[i] = above0[i] & below1[i]; }
    let mut above1 = [false; 20];
    for i in 0..20 { above1[i] = noir_hw::gt(A[i], B[1]); }
    let mut below2 = [false; 20];
    for i in 0..20 { below2[i] = noir_hw::lt(A[i], B[2]); }
    let mut second = [false; 20];
    for i in 0..20 { second[i] = above1[i] & below2[i]; }
    let mut keep = [false; 20];
    for i in 0..20 { keep[i] = first[i] | second[i]; }
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
