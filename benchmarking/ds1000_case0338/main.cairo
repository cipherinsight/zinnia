// DS-1000 case 338 — Cairo baseline.
// Kernel: `np.vstack((np.diag(a), np.diag(np.fliplr(a))))` on Float[8, 8].
// Output Float[2, 8]: row 0 = diag (a[i][i]), row 1 = anti-diag (a[i][7-i]).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..8 {
            let v_int: u128 = ((i + 1) * 10 + (j + 1)).into();
            a.append(v_int * Q);
        };
    };
    let mut claimed: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        claimed.append(*a.at(i * 8 + i));
    };
    for i in 0_u32..8 {
        claimed.append(*a.at(i * 8 + (7 - i)));
    };

    for i in 0_u32..8 {
        let v = *a.at(i * 8 + i);
        assert!(v == *claimed.at(i));
    };
    for i in 0_u32..8 {
        let v = *a.at(i * 8 + (7 - i));
        assert!(v == *claimed.at(8 + i));
    };
}
