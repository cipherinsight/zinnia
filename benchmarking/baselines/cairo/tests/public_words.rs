//! Read-only oracle: call the existing Rust native runner and print its words.
use std::path::Path;
use zkvm_kernels::{fixture::Fixture, host::run_native, unused_params};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let fixture = Fixture::load(Path::new(&a[2])).unwrap();
    let inputs = fixture.args(&a[3], unused_params(&a[1]), true).unwrap();
    println!("{:?}", run_native(&a[1], inputs).unwrap());
}
