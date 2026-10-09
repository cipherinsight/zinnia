// DS-1000 case 0478: literal translation of kernel.py (LITERAL_BASELINES.md).
// einsum ii->i is a writable diagonal view: the last assignment writes through to a.
// @outputs: int[1]; const int[4] = [0, 0, 0, 0]; int[1]; const int[4] = [0, 0, 0, 0]; int[1]; const int[4] = [0, 0, 0, 0]; int[1]
fn main(a: [Field; 16]) -> pub [Field; 4] {
    let result = [a[0],a[5],a[10],a[15]];
    noir_hw::check_inputs(result);
    let save = result;
    let mut a = [0; 16];
    for i in 0..4 { a[i*5] = save[i]; }
    // L7: constants are represented by the output layout.
    [a[0],a[5],a[10],a[15]]
}
