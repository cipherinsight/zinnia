// DS-1000 case 0504: literal translation of kernel.py (LITERAL_BASELINES.md).
// One standard indexed read per row; any reductions scan the fully built array in index order.
// @outputs: float[1]
fn main(a: [Field; 845], b: [Field; 169]) -> pub Field {
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    let mut arr = [0; 169];
    for i in 0..169 {
        let row = [a[i*5+0], a[i*5+1], a[i*5+2], a[i*5+3], a[i*5+4]];
        arr[i] = row[noir_hw::index_pos::<8>(b[i], 5)];
    }
    let mut total: Field = 0;
    for i in 0..169 { total += arr[i]; }
    total
}
