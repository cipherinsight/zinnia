// DS-1000 case 0481, Noir baseline.
// Kernel: `((x == a) & (y == b)).nonzero()[0]` on x, y: Integer[9] (data-dependent length).
// Output: the number of matching indices, then the indices in increasing
// order, zero-padded to 9 (compaction: noir_hw::compact; the payload of item i
// is the constant i).
// @outputs: dynamic int[9x1]

use noir_hw::compact;

fn main(x: [Field; 9], y: [Field; 9], a: Field, b: Field) -> pub [Field; 10] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(x);
    noir_hw::check_inputs(y);
    noir_hw::check_input(a);
    noir_hw::check_input(b);
    let mut hit: [Field; 9] = [0; 9];
    let mut w: [Field; 9] = [0; 9];
    let mut count: Field = 0;
    for i in 0..9 {
        hit[i] = ((x[i] == a) & (y[i] == b)) as Field;
        w[i] = hit[i] * (i as Field);
        count += hit[i];
    }
    let indices = compact(hit, w);
    let mut out: [Field; 10] = [0; 10];
    out[0] = count;
    for j in 0..9 {
        out[j + 1] = indices[j];
    }
    out
}

