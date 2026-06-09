// DS-1000 case 0385 -- Noir baseline.
// a.reshape(H/2, 2, W/2, 2).swapaxes(1,2).transpose(1,0,2,3).reshape(-1,2,2)
// on Float[100,200]. Output (5000, 2, 2) = 20000 cells.
//
// After transpose(1,0,2,3): outer = block_col (W/2=100), then block_row (H/2=50).
// Order of patches: for each block_col, for each block_row -> emit patch.

global H: u32 = 100;
global W: u32 = 200;

fn main(a: [Field; 20000]) -> pub [Field; 20000] {
    let mut out: [Field; 20000] = [0; 20000];
    let mut p: u32 = 0;
    for bc in 0..(W / 2) {
        for br in 0..(H / 2) {
            for pi in 0..2 {
                for qi in 0..2 {
                    let row = br * 2 + pi;
                    let col = bc * 2 + qi;
                    out[p] = a[row * W + col];
                    p = p + 1;
                }
            }
        }
    }
    out
}
