// DS-1000 case 0349: optimized baseline.
// Shape metadata reads no array elements; constants are omitted from instances.
// @outputs: const int[1] = [6]
fn main(a: [Field; 48]) {
    let _a = a;
    let _result: u32 = 6;
}
