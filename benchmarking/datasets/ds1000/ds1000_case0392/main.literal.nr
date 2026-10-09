// DS-1000 case 0392, literal translation of kernel.py (LITERAL_BASELINES.md).
// Constant row slice a[4:7, :] reads only 30 float elements; check each once.
// @outputs: float[30]

fn main(a: [Field; 200]) -> pub [Field; 30] {
    let low = 4;
    let high = 7;
    let mut result = [0; 30];
    for i in low..high {
        for j in 0..10 {
            result[(i - low) * 10 + j] = noir_hw::checked(a[i * 10 + j]);
        }
    }
    result
}
