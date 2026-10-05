// DS-1000 case 0389, Noir baseline.
// Kernel: `a.reshape(6 // 3, -1, 3, 4).swapaxes(1, 2).reshape(6, 12)` on
// Float[6, 3, 4] (`h = 6`, `w = 12` are constants): the six 3x4 blocks laid out
// as a 2 x 3 grid of blocks, so output (R, C) is a[(R / 3) * 3 + C / 4, R % 3, C % 4].
// @outputs: float[72]

fn main(a: [Field; 72]) -> pub [Field; 72] {
    let mut grid = [0; 72];
    for row in 0..6 {
        for col in 0..12 {
            let block = (row / 3) * 3 + col / 4;
            grid[row * 12 + col] = a[block * 12 + (row % 3) * 4 + col % 4];
        }
    }
    grid
}
