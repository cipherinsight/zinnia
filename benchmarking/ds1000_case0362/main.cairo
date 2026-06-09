// DS-1000 case 0362 — np.delete columns on Float[8,6] with runtime del_col.
// del_col = [1, 3, 5, 7]; mask drops 7; keep cols (0-based) {1, 3, 5}. Output (8, 3).
// a[i][j] = (i+1)*10 + (j+1). Quantisation ×2^32 to match Zinnia.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let r: u32 = 8;
    let c: u32 = 6;
    let mut a: Array<i128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < r {
        let mut j: u32 = 0;
        while j < c {
            let raw: i128 = ((i + 1) * 10 + (j + 1)).into();
            a.append(raw * Q);
            j += 1;
        };
        i += 1;
    };
    let mut del_col: Array<i64> = ArrayTrait::new();
    del_col.append(1); del_col.append(3); del_col.append(5); del_col.append(7);
    let mut delete_mask: Array<bool> = ArrayTrait::new();
    let mut jj: u32 = 0;
    while jj < c {
        let mut drop: bool = false;
        for k in 0_usize..4 {
            let v: i64 = *del_col.at(k);
            if v >= 1 && v <= c.into() && (v - 1) == jj.into() { drop = true; };
        };
        delete_mask.append(drop);
        jj += 1;
    };
    let mut out: Array<i128> = ArrayTrait::new();
    let mut ii: u32 = 0;
    while ii < r {
        let mut j2: u32 = 0;
        while j2 < c {
            if !*delete_mask.at(j2.try_into().unwrap()) {
                let idx: u32 = ii * c + j2;
                out.append(*a.at(idx.try_into().unwrap()));
            };
            j2 += 1;
        };
        ii += 1;
    };
    // Row 0 keep cols (1,3,5) → a[0][1]=12, a[0][3]=14, a[0][5]=16 (each × Q).
    assert!(*out.at(0) == 12 * Q);
    assert!(*out.at(1) == 14 * Q);
    assert!(*out.at(2) == 16 * Q);
    // Row 7 keep cols → a[7][1]=82, a[7][3]=84, a[7][5]=86 (each × Q).
    assert!(*out.at(21) == 82 * Q);
    assert!(*out.at(22) == 84 * Q);
    assert!(*out.at(23) == 86 * Q);
}
