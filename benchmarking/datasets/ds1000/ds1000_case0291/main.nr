// DS-1000 case 0291: optimized baseline.
// Shape metadata reads no array elements; constants are omitted from instances.
// @outputs: const int[4] = [2, 5, 5, 2]
fn main(a: [Field; 100]) {
    // result = a.shape (compile-time metadata)
    // Read a[0] to produce a non-empty witness (required by UltraHonk).
    let _a0: Field = a[0];
    let _result: [u32; 4] = [2, 5, 5, 2];
}
