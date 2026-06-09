// DS-1000 case 0366 — np.insert rows at runtime positions.
// a[i][j] = (i+1)*0.1 + (j+1)*0.01 (Float). Quantised ×2^32 to match Zinnia.
// pos=[10,20,30,40,50] → adj=[10,19,28,37,46]. element values 1000 + k*10 + j.
// Output rows = 105 (so 1050 cells).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let r: u32 = 100;
    let c: u32 = 10;
    let e: u32 = 5;
    let mut a: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < r {
        let mut j: u32 = 0;
        while j < c {
            // Driver uses a[i][j] = (i+1)*0.1 + (j+1)*0.01; quantised numerator/100.
            let raw_num: u128 = ((i + 1) * 10 + (j + 1)).into();
            a.append((raw_num * Q) / 100);
            j += 1;
        };
        i += 1;
    };
    let mut pos: Array<i64> = ArrayTrait::new();
    pos.append(10); pos.append(20); pos.append(30); pos.append(40); pos.append(50);
    let mut element: Array<u128> = ArrayTrait::new();
    let mut k: u32 = 0;
    while k < e {
        let mut j: u32 = 0;
        while j < c {
            // Driver uses element[k][j] = 1000 + k*10 + j (whole numbers).
            let raw: u128 = (1000 + k * 10 + j).into();
            element.append(raw * Q);
            j += 1;
        };
        k += 1;
    };
    let mut adj: Array<i64> = ArrayTrait::new();
    let mut kk: i64 = 0;
    while kk < 5 {
        let p: i64 = *pos.at(kk.try_into().unwrap());
        adj.append(p - kk);
        kk += 1;
    };
    let mut out: Array<u128> = ArrayTrait::new();
    let mut ki: u32 = 0;
    let mut src: u32 = 0;
    while src < r {
        while ki < e && *adj.at(ki.try_into().unwrap()) == src.into() {
            let base: u32 = ki * c;
            let mut jj: u32 = 0;
            while jj < c { out.append(*element.at((base + jj).try_into().unwrap())); jj += 1; };
            ki += 1;
        };
        let base: u32 = src * c;
        let mut jj: u32 = 0;
        while jj < c { out.append(*a.at((base + jj).try_into().unwrap())); jj += 1; };
        src += 1;
    };
    while ki < e {
        let base: u32 = ki * c;
        let mut jj: u32 = 0;
        while jj < c { out.append(*element.at((base + jj).try_into().unwrap())); jj += 1; };
        ki += 1;
    };
    // Total rows = 105 → 1050 cells.
    assert!(out.len() == 1050);
}
