//! SP1 guest support: the private-input reader and the `guest!` macro that
//! every per-case binary in `src/bin/` expands.
//!
//! Each kernel parameter arrives as one raw buffer of little-endian 64-bit
//! words (`sp1_zkvm::io::read_vec`, one copy), and the output is committed as
//! one raw buffer (`commit_slice`); see `ds1000_zkvm` for the layout.
#![no_std]

extern crate alloc;

pub use ds1000_zkvm;
pub use sp1_zkvm;

use alloc::vec::Vec;

/// Reads the kernel parameters from SP1's private input stream.
pub struct Sp1Input;

impl ds1000_zkvm::Input for Sp1Input {
    fn read_words(&mut self) -> Vec<i64> {
        let bytes = sp1_zkvm::io::read_vec();
        ds1000_zkvm::words_from_le_bytes(&bytes).expect("a parameter is a whole number of words")
    }
}

/// Commits `output` as the public values.
pub fn commit<T: ds1000_zkvm::Publics>(output: &T) {
    let words = output.encode_to_vec();
    sp1_zkvm::io::commit_slice(ds1000_zkvm::words_as_le_bytes(&words));
}

/// The guest for one case: read the private inputs, run the kernel, and
/// commit its output as the public values.
#[macro_export]
macro_rules! guest {
    ($case:ident) => {
        $crate::sp1_zkvm::entrypoint!(main);

        fn main() {
            let output = $crate::ds1000_zkvm::$case::main(&mut $crate::Sp1Input);
            $crate::commit(&output);
        }
    };
}
