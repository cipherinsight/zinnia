// DS-1000 case 0387 -- Noir baseline. a.reshape(50,2,100,2).swapaxes(1,2)
// .reshape(-1,2,2) on Float[100,200]. Output = (5000,2,2) = 20000 cells.
// Q32 fixed-point but pure reshape so values are unmodified.

global H: u32 = 100;
global W: u32 = 200;

fn main(a: [Field; 20000]) -> pub [Field; 20000] {
    let mut out: [Field; 20000] = [0; 20000];
    let mut p: u32 = 0;
    for r_block in 0..(H / 2) {
        for c_block in 0..(W / 2) {
            for pi in 0..2 {
                for qi in 0..2 {
                    let row = r_block * 2 + pi;
                    let col = c_block * 2 + qi;
                    out[p] = a[row * W + col];
                    p = p + 1;
                }
            }
        }
    }
    out
}
