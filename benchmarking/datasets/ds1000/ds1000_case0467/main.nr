// DS-1000 case 0467: optimized baseline.
// Clipped 3x3 windows on a 4x4 integer input, row-first window order.
// Static slices alias input wires; only the 16 input-domain checks need constraints.
// @outputs: int[100]

fn main(a: [Field; 16]) -> pub [Field; 100] {
    noir_hw::check_inputs(a);
    // Compile-time flattened slice map: windows row-first, each slice row-major.
    [
        a[0], a[1], a[4], a[5], a[0], a[1], a[2], a[4], a[5], a[6],
        a[1], a[2], a[3], a[5], a[6], a[7], a[2], a[3], a[6], a[7],
        a[0], a[1], a[4], a[5], a[8], a[9], a[0], a[1], a[2], a[4],
        a[5], a[6], a[8], a[9], a[10], a[1], a[2], a[3], a[5], a[6],
        a[7], a[9], a[10], a[11], a[2], a[3], a[6], a[7], a[10], a[11],
        a[4], a[5], a[8], a[9], a[12], a[13], a[4], a[5], a[6], a[8],
        a[9], a[10], a[12], a[13], a[14], a[5], a[6], a[7], a[9], a[10],
        a[11], a[13], a[14], a[15], a[6], a[7], a[10], a[11], a[14], a[15],
        a[8], a[9], a[12], a[13], a[8], a[9], a[10], a[12], a[13], a[14],
        a[9], a[10], a[11], a[13], a[14], a[15], a[10], a[11], a[14], a[15],
    ]
}
