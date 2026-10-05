//! RISC Zero guest support: the private-input reader and the `guest!` macro
//! that every per-case binary in `src/bin/` expands.
#![no_std]

pub use ds1000_zkvm;
pub use risc0_zkvm;

/// Reads the kernel parameters from RISC Zero's private input stream.
pub struct Risc0Input;

impl ds1000_zkvm::Input for Risc0Input {
    fn read<T: serde::de::DeserializeOwned>(&mut self) -> T {
        risc0_zkvm::guest::env::read()
    }
}

/// The guest for one case: read the private inputs, run the kernel, and
/// commit its output to the journal.
#[macro_export]
macro_rules! guest {
    ($case:ident) => {
        fn main() {
            let output = $crate::ds1000_zkvm::$case::main(&mut $crate::Risc0Input);
            $crate::risc0_zkvm::guest::env::commit(&output);
        }
    };
}
