// DS-1000 case 0422: literal translation of kernel.py (LITERAL_BASELINES.md).
// Valid correlation when the second input is longer uses lag 49-k; each defining sum rescales once.
// @outputs: float[50]
fn main(a: [Field; 50], b: [Field; 50]) -> pub [Field; 50] {
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    let mut stacked = [0; 99];
    for i in 0..49 { stacked[i] = b[i + 1]; }
    for i in 0..50 { stacked[49 + i] = b[i]; }
    let mut result = [0; 50];
    for k in 0..50 {
        let mut acc: Field = 0;
        for n in 0..50 { acc += a[n] * stacked[n + 49 - k]; }
        result[k] = noir_hw::rescale(acc);
    }
    result
}
