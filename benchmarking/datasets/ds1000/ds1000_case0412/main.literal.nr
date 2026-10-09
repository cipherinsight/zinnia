// DS-1000 case 0412, Noir LITERAL baseline (LITERAL_BASELINES.md).
// Literal translation of kernel.py: `result = x[x >= 0]` on Float[10]
// (data-dependent length). x[mask] has no source to copy (L5): the textbook
// algorithm is a running count of kept elements, then for each output slot j
// the sum over i of [kept_i and count_i == j] * x_i (an equality test per
// pair) -- NOT noir_hw::compact, which L6 forbids. count_i (prefix[i]) is the
// number of kept elements before position i; an element kept at i lands in
// output slot prefix[i].
// @outputs: dynamic float[10x1]

fn main(x: [Field; 10]) -> pub [Field; 11] {
    noir_hw::check_inputs(x);

    let mut keep: [Field; 10] = [0; 10];
    for i in 0..10 {
        keep[i] = (noir_hw::ge(x[i], 0)) as Field;
    }

    let mut prefix: [Field; 11] = [0; 11];
    for i in 0..10 {
        prefix[i + 1] = prefix[i] + keep[i];
    }
    let count = prefix[10];

    let mut out: [Field; 11] = [0; 11];
    out[0] = count;
    for j in 0..10 {
        let mut acc: Field = 0;
        for i in 0..10 {
            let is_slot = (prefix[i] == j as Field) as Field;
            acc += keep[i] * is_slot * x[i];
        }
        out[j + 1] = acc;
    }
    out
}
