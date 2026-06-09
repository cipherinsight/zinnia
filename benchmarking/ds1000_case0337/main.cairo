// DS-1000 case 337 — Cairo baseline.
// Kernel: `np.diag(np.fliplr(a))` on Float[8, 10] — pure static-index gather of
// the anti-diagonal a[i][9 - i].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..10 {
            let v_int: u128 = ((i + 1) * 10 + (j + 1)).into();
            a.append(v_int * Q);
        };
    };
    let mut claimed: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        claimed.append(*a.at(i * 10 + (9 - i)));
    };

    for i in 0_u32..8 {
        let v = *a.at(i * 10 + (9 - i));
        assert!(v == *claimed.at(i));
    };
}
