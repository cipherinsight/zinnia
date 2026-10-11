//! Shape-generic Q32 port of zkvm.rs::fibo.
// output: i64
use crate::io::ArgsTrait;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let n: i64 = args.scalar();
    assert!(n >= 0 && n <= 1024, "requires 0 <= n <= 1024");
    let mut a: i64 = 1;
    let mut b: i64 = 1;
    for _ in 0_u32..n.try_into().unwrap() {
        // Rust explicitly uses wrapping_add, including beyond n = 90.
        let sum: i128 = Into::<i64, i128>::into(a) + b.into();
        let next = if sum > 9223372036854775807 {
            sum - 18446744073709551616
        } else if sum < -9223372036854775808 {
            sum + 18446744073709551616
        } else {
            sum
        };
        b = a;
        a = next.try_into().unwrap();
    }
    let mut out = args.output();
    out.append(a);
    out
}
