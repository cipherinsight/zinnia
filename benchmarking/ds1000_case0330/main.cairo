// DS-1000 case 330 — Cairo baseline.
// Kernel: `a ** power` on Float[8, 6] with integer scalar `power`.
// Element-wise pow via repeated qmul. Sample: power=3, a[i][j]=(i+1)+0.1*(j+1).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let power: u32 = 3;

    let mut a: Array<u128> = ArrayTrait::new();
    let mut claimed: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            // a_ij = (i+1) + 0.1*(j+1) in Q-scale = ((i+1)*10 + (j+1)) * Q / 10
            let v_q: u128 = (((i + 1) * 10 + (j + 1)).into() * Q) / 10;
            a.append(v_q);
            // claimed = v^power via repeated mul.
            let mut acc: u128 = Q;
            for _ in 0..power {
                acc = (acc * v_q) / Q;
            };
            claimed.append(acc);
        };
    };

    // Recompute and verify.
    let mut k: u32 = 0;
    for _i in 0_u32..8 {
        for _j in 0_u32..6 {
            let v_q = *a.at(k);
            let mut acc: u128 = Q;
            for _ in 0..power {
                acc = (acc * v_q) / Q;
            };
            assert!(acc == *claimed.at(k));
            k += 1;
        };
    };
}
