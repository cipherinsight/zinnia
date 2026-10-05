// DS-1000 case 409 — z = x + y over Float[8, 6].
// Q-scaled addition matches halo2 qadd; output exposed via assertion.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    // x[i][j] = ((i+1)*10 + (j+1)) / 100; y[i][j] = (i+1).
    let mut x: Array<u128> = ArrayTrait::new();
    let mut y: Array<u128> = ArrayTrait::new();
    let mut claimed: Array<u128> = ArrayTrait::new();
    for i in 0_u128..8 {
        for j in 0_u128..6 {
            let num: u128 = (i + 1) * 10 + (j + 1);
            let xv: u128 = (num * Q) / 100;
            let yv: u128 = (i + 1) * Q;
            x.append(xv);
            y.append(yv);
            claimed.append(xv + yv);
        };
    };

    let mut idx: u32 = 0;
    while idx < 48 {
        let xi: u128 = *x.at(idx);
        let yi: u128 = *y.at(idx);
        let zi: u128 = xi + yi;
        let ci: u128 = *claimed.at(idx);
        assert!(zi == ci);
        idx += 1;
    };
}
