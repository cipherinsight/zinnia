//! SP1 guest support: the private-input reader and the `guest!` macro that
//! every per-case binary in `src/bin/` expands.
//!
//! Each kernel parameter arrives as one raw buffer of little-endian 64-bit
//! words (`sp1_zkvm::io::read_vec`, one copy), and the output is committed as
//! one raw buffer (`commit_slice`); see `zkvm_kernels` for the layout.
#![no_std]

extern crate alloc;

pub use sp1_zkvm;
pub use zkvm_kernels;

use alloc::vec::Vec;

/// Reads the kernel parameters from SP1's private input stream.
pub struct Sp1Input;

impl zkvm_kernels::Input for Sp1Input {
    fn read_words(&mut self) -> Vec<i64> {
        let bytes = sp1_zkvm::io::read_vec();
        zkvm_kernels::words_from_le_bytes(&bytes).expect("a parameter is a whole number of words")
    }
}

/// Commits the case's words as the public values.
pub fn commit(words: &[i64]) {
    sp1_zkvm::io::commit_slice(zkvm_kernels::words_as_le_bytes(words));
}

/// The guest for one case: read the private inputs, run the kernel, and
/// commit its output (with the input shapes, for a shape-generic program).
#[macro_export]
macro_rules! guest {
    (fixed $($case:ident)::+) => {
        $crate::sp1_zkvm::entrypoint!(main);

        fn main() {
            let words = $crate::zkvm_kernels::run_fixed(&mut $crate::Sp1Input, |input| {
                $crate::zkvm_kernels::$($case)::+::main(input)
            });
            $crate::commit(&words);
        }
    };
    (shaped $($case:ident)::+) => {
        $crate::sp1_zkvm::entrypoint!(main);

        fn main() {
            let words = $crate::zkvm_kernels::run_shaped(&mut $crate::Sp1Input, |args| {
                $crate::zkvm_kernels::$($case)::+::main(args)
            });
            $crate::commit(&words);
        }
    };
}
