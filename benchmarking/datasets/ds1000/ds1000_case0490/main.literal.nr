// DS-1000 case 0490: literal translation of kernel.py (LITERAL_BASELINES.md).
// where and boolean indexing use count/equality compaction; writes scan every position.
// Private-length loops run 50 slots with active flags; all four comparisons remain separate.
// @outputs: float[250]
fn main(arr: [Field; 250], n1: [Field; 5], n2: [Field; 5]) -> pub [Field; 250] {
    noir_hw::check_inputs(arr); noir_hw::check_inputs(n1); noir_hw::check_inputs(n2);
    let mut result = arr;
    for r in 0..5 {
        let mut a = [0; 50];
        for i in 0..50 { a[i] = result[r*50+i]; }
        let temp = a;
        let mut mask = [false; 50];
        for i in 0..50 { mask[i] = noir_hw::lt(temp[i], noir_hw::from_int(n1[r])); }
        let (count, pos) = positions(mask);
        for j in 0..50 {
            let active = noir_hw::lt(j as Field, count);
            for k in 0..50 { a[k] = if active & (pos[j] == k as Field) { 0 } else { a[k] }; }
        }
        let mut mask = [false; 50];
        for i in 0..50 { mask[i] = noir_hw::ge(temp[i], noir_hw::from_int(n2[r])); }
        let (count, pos) = positions(mask);
        for j in 0..50 {
            let active = noir_hw::lt(j as Field, count);
            for k in 0..50 { a[k] = if active & (pos[j] == k as Field) { 30*4294967296 } else { a[k] }; }
        }
        let mut lower = [false; 50];
        for i in 0..50 { lower[i] = noir_hw::ge(temp[i], noir_hw::from_int(n1[r])); }
        let mut upper = [false; 50];
        for i in 0..50 { upper[i] = noir_hw::lt(temp[i], noir_hw::from_int(n2[r])); }
        let mut mask = [false; 50];
        for i in 0..50 { mask[i] = lower[i] & upper[i]; }
        let (_, selected) = gather_mask(a, mask);
        let mut values = [0; 50];
        for j in 0..50 { values[j] = selected[j] + 5*4294967296; }
        let mut before: Field = 0;
        for k in 0..50 {
            for j in 0..50 { a[k] = if mask[k] & (before == j as Field) { values[j] } else { a[k] }; }
            before += mask[k] as Field;
        }
        for i in 0..50 { result[r*50+i] = a[i]; }
    }
    result
}
fn positions(mask: [bool; 50]) -> (Field, [Field; 50]) {
    let mut payload = [0; 50];
    for i in 0..50 { payload[i] = i as Field; }
    gather_mask(payload, mask)
}
fn gather_mask(payload: [Field; 50], mask: [bool; 50]) -> (Field, [Field; 50]) {
    let mut before = [0; 50]; let mut count: Field = 0;
    for i in 0..50 { before[i] = count; count += mask[i] as Field; }
    let mut result = [0; 50];
    for j in 0..50 {
        let mut acc: Field = 0;
        for i in 0..50 { acc += ((mask[i] & (before[i] == j as Field)) as Field) * payload[i]; }
        result[j] = acc;
    }
    (count, result)
}
