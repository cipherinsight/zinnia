// DS-1000 case 0328 — pad-or-truncate to runtime length.
// kernel.py: if length > A.shape[0]: pad zeros; else A.copy() with A[length:]=0.
// A = [1,2,3,4,5], length = 8 → output [1,2,3,4,5,0,0,0].

#[executable]
pub fn main() {
    let n: u32 = 5;
    let length: u32 = 8;
    let mut a: Array<i64> = ArrayTrait::new();
    a.append(1); a.append(2); a.append(3); a.append(4); a.append(5);
    let mut expect: Array<i64> = ArrayTrait::new();
    expect.append(1); expect.append(2); expect.append(3); expect.append(4); expect.append(5);
    expect.append(0); expect.append(0); expect.append(0);
    let mut out: Array<i64> = ArrayTrait::new();
    if length > n {
        for i in 0_u32..length {
            if i < n { out.append(*a.at(i.try_into().unwrap())); } else { out.append(0); };
        };
    } else {
        for i in 0_u32..n {
            if i < length { out.append(*a.at(i.try_into().unwrap())); } else { out.append(0); };
        };
    };
    let mut k: u32 = 0;
    let total = if length > n { length } else { n };
    while k < total {
        assert!(*out.at(k.try_into().unwrap()) == *expect.at(k.try_into().unwrap()));
        k += 1;
    };
}
