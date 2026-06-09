// DS-1000 case 0442 -- Noir baseline. A[~np.in1d(A, B)] on Integer[20]x[3].
// Output 21 fields: 20 masked cells (kept value or 0) + length.

global NA: u32 = 20;
global NB: u32 = 3;

fn main(a: [Field; 20], b: [Field; 3]) -> pub [Field; 21] {
    let mut out: [Field; 21] = [0; 21];
    let mut count: Field = 0;
    for i in 0..NA {
        let mut any_eq: Field = 0;
        for j in 0..NB {
            let eq: Field = if a[i] == b[j] { 1 } else { 0 };
            any_eq = any_eq + eq - any_eq * eq;
        }
        let keep = 1 - any_eq;
        out[i] = a[i] * keep;
        count = count + keep;
    }
    out[NA] = count;
    out
}
