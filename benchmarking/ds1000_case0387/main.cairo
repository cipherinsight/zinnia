// DS-1000 case 387 — a.reshape(50, 2, 100, 2).swapaxes(1, 2).reshape(-1, 2, 2).
// Output[k][p][q] = a[r_block*2+p][c_block*2+q] with k = r_block*100 + c_block.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    const H: u32 = 100;
    const W: u32 = 200;

    let mut a: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i != H {
        let mut j: u32 = 0;
        let i_u128: u128 = i.into();
        while j != W {
            let j_u128: u128 = j.into();
            let i_q: u128 = (i_u128 * Q) / 1000;
            let j_q: u128 = (j_u128 * Q) / 10000;
            a.append(i_q + j_q);
            j = j + 1;
        };
        i = i + 1;
    };

    let mut expected: Array<u128> = ArrayTrait::new();
    let mut rb: u32 = 0;
    while rb != 50 {
        let mut cb: u32 = 0;
        while cb != 100 {
            let mut p: u32 = 0;
            while p != 2 {
                let mut q: u32 = 0;
                while q != 2 {
                    let row: u32 = rb * 2 + p;
                    let col: u32 = cb * 2 + q;
                    expected.append(*a.at(row * W + col));
                    q = q + 1;
                };
                p = p + 1;
            };
            cb = cb + 1;
        };
        rb = rb + 1;
    };

    let mut k: u32 = 0;
    let mut rb2: u32 = 0;
    while rb2 != 50 {
        let mut cb2: u32 = 0;
        while cb2 != 100 {
            let mut p2: u32 = 0;
            while p2 != 2 {
                let mut q2: u32 = 0;
                while q2 != 2 {
                    let row: u32 = rb2 * 2 + p2;
                    let col: u32 = cb2 * 2 + q2;
                    assert!(*a.at(row * W + col) == *expected.at(k));
                    k = k + 1;
                    q2 = q2 + 1;
                };
                p2 = p2 + 1;
            };
            cb2 = cb2 + 1;
        };
        rb2 = rb2 + 1;
    };
}
