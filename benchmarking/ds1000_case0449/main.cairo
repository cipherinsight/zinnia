// DS-1000 case 449 — `np.vstack(([x_dists.T], [y_dists.T])).T`.
// Float[3,4] × Float[3,4] → Float[3,4,2].
// dists[i][j][0] = x_dists[i][j]; dists[i][j][1] = y_dists[i][j].
// Quantisation ×2^32 (Q).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // x_dists[i][j] = ((i+1)*10 + (j+1)) / 100
    let mut x_dists: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..4 {
            let num: u128 = ((i + 1) * 10 + (j + 1)).into();
            x_dists.append((num * Q) / 100);
        };
    };
    // y_dists[i][j] = 1 + x_dists[i][j]
    let mut y_dists: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..4 {
            let num: u128 = ((i + 1) * 10 + (j + 1)).into();
            y_dists.append(Q + (num * Q) / 100);
        };
    };

    let mut dists: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..4 {
            dists.append(*x_dists.at(i * 4 + j));
            dists.append(*y_dists.at(i * 4 + j));
        };
    };

    // Claimed: equals dists itself by construction.
    for i in 0_u32..3 {
        for j in 0_u32..4 {
            let off: u32 = (i * 4 + j) * 2;
            assert!(*dists.at(off) == *x_dists.at(i * 4 + j));
            assert!(*dists.at(off + 1) == *y_dists.at(i * 4 + j));
        };
    };
}
