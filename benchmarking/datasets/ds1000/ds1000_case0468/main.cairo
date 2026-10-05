// DS-1000 case 468 — Cairo baseline. Column-major 3×3 windowing of a 4×4 int matrix.

#[executable]
pub fn main() {
    let mut a: Array<i128> = ArrayTrait::new();
    for v in 1_i128..17 {
        a.append(v);
    };
    let size0: i128 = 3;
    let size1: i128 = 3;

    let r_win: i128 = size0 / 2;
    let c_win: i128 = size1 / 2;
    let x: i128 = 4;
    let y: i128 = 4;

    let claim_counts: Array<u32> = array![4, 6, 6, 4, 6, 9, 9, 6, 6, 9, 9, 6, 4, 6, 6, 4];
    let claim_total: u32 = 100;
    let claim_flat: Array<i128> = array![
        1, 2, 5, 6,
        1, 2, 5, 6, 9, 10,
        5, 6, 9, 10, 13, 14,
        9, 10, 13, 14,
        1, 2, 3, 5, 6, 7,
        1, 2, 3, 5, 6, 7, 9, 10, 11,
        5, 6, 7, 9, 10, 11, 13, 14, 15,
        9, 10, 11, 13, 14, 15,
        2, 3, 4, 6, 7, 8,
        2, 3, 4, 6, 7, 8, 10, 11, 12,
        6, 7, 8, 10, 11, 12, 14, 15, 16,
        10, 11, 12, 14, 15, 16,
        3, 4, 7, 8,
        3, 4, 7, 8, 11, 12,
        7, 8, 11, 12, 15, 16,
        11, 12, 15, 16,
    ];

    let mut flat: Array<i128> = ArrayTrait::new();
    let mut counts: Array<u32> = ArrayTrait::new();
    for j in 0_i128..y {
        let ymin: i128 = if j - c_win > 0 { j - c_win } else { 0 };
        let ymax: i128 = if j + c_win + 1 < y { j + c_win + 1 } else { y };
        for i in 0_i128..x {
            let xmin: i128 = if i - r_win > 0 { i - r_win } else { 0 };
            let xmax: i128 = if i + r_win + 1 < x { i + r_win + 1 } else { x };
            let mut cnt: u32 = 0;
            for ii in xmin..xmax {
                for jj in ymin..ymax {
                    let idx_i: u32 = ii.try_into().unwrap();
                    let idx_j: u32 = jj.try_into().unwrap();
                    let idx: u32 = idx_i * 4 + idx_j;
                    flat.append(*a.at(idx));
                    cnt += 1;
                };
            };
            counts.append(cnt);
        };
    };

    assert!(counts.len() == 16);
    for k in 0_u32..16 {
        assert!(*counts.at(k) == *claim_counts.at(k));
    };
    let total: u32 = flat.len();
    assert!(total == claim_total);
    for k in 0_u32..claim_total {
        assert!(*flat.at(k) == *claim_flat.at(k));
    };
}
