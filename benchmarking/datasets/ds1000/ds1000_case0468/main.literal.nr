// DS-1000 case 0468: literal translation of kernel.py (LITERAL_BASELINES.md).
// Clipped 3x3 windows on a 4x4 integer input, column-first window order.
// The helper preserves source statement and loop order; all slice bounds are shape constants.
// @outputs: int[100]
fn window(arr: [Field; 16], shape: [u32; 2]) -> [Field; 100] {
    let mut ans = [0; 100];
    let mut next: u32 = 0;
    // floor(shape / 2).astype(int): positive compile-time shape arithmetic.
    let r_win = shape[0] / 2;
    let c_win = shape[1] / 2;
    let x: u32 = 4;
    let y: u32 = 4;
    for j in 0..y {
        let ymin = if j > c_win { j - c_win } else { 0 };
        let ymax = if j + c_win + 1 < y { j + c_win + 1 } else { y };
        for i in 0..x {
            let xmin = if i > r_win { i - r_win } else { 0 };
            let xmax = if i + r_win + 1 < x { i + r_win + 1 } else { x };
            // Append the complete slice in row-major order.
            // All indices depend only on loop constants, never on private data.
            for r in xmin..xmax {
                for c in ymin..ymax {
                    ans[next] = arr[r * y + c];
                    next += 1;
                }
            }
        }
    }
    ans
}

fn main(a: [Field; 16]) -> pub [Field; 100] {
    noir_hw::check_inputs(a);
    let size: [u32; 2] = [3, 3];
    let result = window(a, size);
    result
}
