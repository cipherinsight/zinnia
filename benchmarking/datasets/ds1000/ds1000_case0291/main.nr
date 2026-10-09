// DS-1000 case 0291: optimized baseline.
// Shape metadata reads no array elements; constants are omitted from instances.
// @outputs: const int[4] = [2, 5, 5, 2]
fn main(a: [Field; 100]) {
    let _a = a;
    // result = a.shape (compile-time metadata)
    let _result: [u32; 4] = [2, 5, 5, 2];
}
