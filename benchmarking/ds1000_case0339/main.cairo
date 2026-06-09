// DS-1000 case 339 — Cairo baseline.
// Kernel: `np.vstack((np.diag(b), np.diag(np.fliplr(b))))` where b = a[:8, :8]
// of a Float[8, 10] input. Static-index gather at row-major offset i*10+(i or 7-i).

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
        claimed.append(*a.at(i * 10 + i));
    };
    for i in 0_u32..8 {
        claimed.append(*a.at(i * 10 + (7 - i)));
    };

    for i in 0_u32..8 {
        assert!(*a.at(i * 10 + i) == *claimed.at(i));
    };
    for i in 0_u32..8 {
        assert!(*a.at(i * 10 + (7 - i)) == *claimed.at(8 + i));
    };
}
