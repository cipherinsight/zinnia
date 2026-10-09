// DS-1000 case 0435: literal translation of kernel.py (LITERAL_BASELINES.md).
// Constant writes are wiring; overwritten inputs are unread and constant output segments are omitted per L7.
// @outputs: const int[1] = [0]; int[3]; const int[5] = [0, 0, 0, 0, 0]; int[3]; const int[1] = [0]; int[3]
fn main(a: [Field; 16]) -> pub [Field; 9] {
    let read = [a[1], a[2], a[3], a[9], a[10], a[11], a[13], a[14], a[15]];
    noir_hw::check_inputs(read);
    let mut result = a;
    for col in 0..4 { result[4 + col] = 0; }
    for row in 0..4 { result[row * 4] = 0; }
    [result[1], result[2], result[3], result[9], result[10], result[11], result[13], result[14], result[15]]
}
