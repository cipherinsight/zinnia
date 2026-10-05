// DS-1000 case 388 — parametric patchify: x = a[:H/p*p, :W/p*p]; reshape & swap.
// H=100, W=200, p=4 → 25 r_blocks, 50 c_blocks, 1250 patches of 4×4.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    const H: u32 = 100;
    const W: u32 = 200;
    let p: u32 = 4;

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

    let h_eff: u32 = (H / p) * p;
    let w_eff: u32 = (W / p) * p;
    let r_blocks: u32 = h_eff / p;
    let c_blocks: u32 = w_eff / p;

    let mut expected: Array<u128> = ArrayTrait::new();
    let mut rb: u32 = 0;
    while rb != r_blocks {
        let mut cb: u32 = 0;
        while cb != c_blocks {
            let mut pp: u32 = 0;
            while pp != p {
                let mut qq: u32 = 0;
                while qq != p {
                    let row: u32 = rb * p + pp;
                    let col: u32 = cb * p + qq;
                    expected.append(*a.at(row * W + col));
                    qq = qq + 1;
                };
                pp = pp + 1;
            };
            cb = cb + 1;
        };
        rb = rb + 1;
    };

    let mut k: u32 = 0;
    let mut rb2: u32 = 0;
    while rb2 != r_blocks {
        let mut cb2: u32 = 0;
        while cb2 != c_blocks {
            let mut pp2: u32 = 0;
            while pp2 != p {
                let mut qq2: u32 = 0;
                while qq2 != p {
                    let row: u32 = rb2 * p + pp2;
                    let col: u32 = cb2 * p + qq2;
                    assert!(*a.at(row * W + col) == *expected.at(k));
                    k = k + 1;
                    qq2 = qq2 + 1;
                };
                pp2 = pp2 + 1;
            };
            cb2 = cb2 + 1;
        };
        rb2 = rb2 + 1;
    };
}
