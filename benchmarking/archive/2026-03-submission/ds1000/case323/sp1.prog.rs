// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read inputs
    let degree: f64 = sp1_zkvm::io::read::<f64>();
    let result: f64 = sp1_zkvm::io::read::<f64>();

    let pi: f64 = 3.141592653589793;
    let rad: f64 = degree * pi / 180.0;
    let computed: f64 = rad.sin();

    // verify equality (with floating tolerance)
    assert!((result - computed).abs() < 1e-9);

    // sp1_zkvm::io::commit_slice(&output);
}
