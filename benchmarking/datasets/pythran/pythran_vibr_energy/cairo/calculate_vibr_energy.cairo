//! Shape-generic Q32 port of zkvm.rs::calculate_vibr_energy.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let harmonic: Array<Q32> = args.array1();
    let anharmonic: Array<Q32> = args.array1();
    let levels: Array<Q32> = args.array1();
    assert!(
        harmonic.len() == levels.len() && anharmonic.len() == levels.len(),
        "operands shape mismatch",
    );
    let mut result = array![];
    for j in 0..levels.len() {
        let i = *levels.at(j);
        result.append((-*harmonic.at(j) * i - *anharmonic.at(j) * (i * i)).exp());
    }
    let mut out = args.output();
    tensor(ref out, array![levels.len()].span(), result.span());
    out
}
