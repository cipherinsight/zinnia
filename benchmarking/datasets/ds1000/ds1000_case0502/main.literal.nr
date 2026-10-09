// DS-1000 case 0502: literal translation of kernel.py (LITERAL_BASELINES.md).
// One standard indexed read per row; newaxis and the final slice are wiring.
// @outputs: float[169]
fn main(a: [Field; 338], b: [Field; 169]) -> pub [Field; 169] {
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    let mut arr = [0; 169];
    for i in 0..169 {
        let row = [a[i*2+0], a[i*2+1]];
        arr[i] = row[noir_hw::index_pos::<8>(b[i], 2)];
    }
    arr
}
