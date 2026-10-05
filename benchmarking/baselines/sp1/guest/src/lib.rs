//! SP1 guest support: the private-input reader and the `guest!` macro that
//! every per-case binary in `src/bin/` expands.
#![no_std]

pub use ds1000_zkvm;
pub use sp1_zkvm;

/// Reads the kernel parameters from SP1's private input stream.
pub struct Sp1Input;

impl ds1000_zkvm::Input for Sp1Input {
    fn read<T: serde::de::DeserializeOwned>(&mut self) -> T {
        sp1_zkvm::io::read()
    }
}

/// The guest for one case: read the private inputs, run the kernel, and
/// commit its output as the public values.
#[macro_export]
macro_rules! guest {
    ($case:ident) => {
        $crate::sp1_zkvm::entrypoint!(main);

        fn main() {
            let output = $crate::ds1000_zkvm::$case::main(&mut $crate::Sp1Input);
            $crate::sp1_zkvm::io::commit(&output);
        }
    };
}
