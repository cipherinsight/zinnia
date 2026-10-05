// DS-1000 case 0489 — chained where with mutation on signed quantised floats.
// kernel.py: snapshot result = arr; arr[result<-10]=0; arr[result>=15]=30;
// arr[(-10 <= result) & (result < 15)] += 5. Concrete dims 2×2×2 (8 cells).
// Quantisation ×2^32 to match Zinnia. arr values: -20, -5, 0, 12, 14.9, 15, 20, 100.
// Thresholds: -10·Q, 15·Q. The `+5` becomes `+5·Q`. 14.9 = 149·Q/10.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let mut arr: Array<i128> = ArrayTrait::new();
    arr.append(-20 * Q);
    arr.append(-5 * Q);
    arr.append(0);
    arr.append(12 * Q);
    arr.append((149 * Q) / 10);  // 14.9
    arr.append(15 * Q);
    arr.append(20 * Q);
    arr.append(100 * Q);
    // Expected (unquantised): [0, 0, 5, 17, 19.9, 30, 30, 30].
    let mut expect: Array<i128> = ArrayTrait::new();
    expect.append(0);
    expect.append(0);
    expect.append(5 * Q);
    expect.append(17 * Q);
    expect.append((199 * Q) / 10);  // 19.9
    expect.append(30 * Q);
    expect.append(30 * Q);
    expect.append(30 * Q);
    let lo: i128 = -10 * Q;
    let hi: i128 = 15 * Q;
    let mut out: Array<i128> = ArrayTrait::new();
    for i in 0_usize..8 {
        let v: i128 = *arr.at(i);
        let new_v: i128 = if v < lo { 0 }
            else if v >= hi { 30 * Q }
            else { v + 5 * Q };
        out.append(new_v);
    };
    for i in 0_usize..8 {
        assert!(*out.at(i) == *expect.at(i));
    };
}
