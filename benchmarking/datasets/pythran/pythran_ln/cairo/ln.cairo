//! Shape-generic Q32 port of zkvm.rs::ln.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{ONE, Q32, Q32Trait};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let x: Array<Q32> = args.array1();
    let y: Array<Q32> = args.array1();
    let mut series = array![];
    for v in x.span() {
        let d = *v - ONE;
        let mut power = d;
        let mut sum = d;
        for n in 2..10_u32 {
            power = power * d;
            let term = power.div_int(n.into());
            sum = if n % 2 == 0 {
                sum - term
            } else {
                sum + term
            };
        }
        series.append(sum);
    }
    assert!(x.len() == y.len() || x.len() == 1, "assignment shape mismatch");
    let mut result = array![];
    for i in 0..y.len() {
        result.append(*series.at(if x.len() == 1 {
            0
        } else {
            i
        }));
    }
    let mut out = args.output();
    tensor(ref out, array![y.len()].span(), result.span());
    out
}
