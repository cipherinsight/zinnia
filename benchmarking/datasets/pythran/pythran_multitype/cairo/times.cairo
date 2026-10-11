// output: i64
use crate::io::ArgsTrait;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let n: i64 = args.scalar();
    let m: i64 = args.scalar();
    assert!(0 <= n && n <= 1024, "requires 0 <= n <= 1024");
    assert!(0 <= m && m <= 1024, "requires 0 <= m <= 1024");
    let mut out = args.output();
    out.append(n * m);
    out
}
