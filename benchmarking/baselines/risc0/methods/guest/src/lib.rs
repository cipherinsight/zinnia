//! RISC Zero guest support: the private-input reader and the `guest!` macro
//! that every per-case binary in `src/bin/` expands.
//!
//! Each kernel parameter arrives as a frame (`env::read_slice` of a `u32`
//! word count, then of the 64-bit words: two reads, as RISC Zero recommends
//! for bulk data instead of `env::read`'s per-element deserialization), and
//! the output is committed as one raw buffer (`env::commit_slice`); see
//! `ds1000_zkvm` for the layout.
#![no_std]

extern crate alloc;

pub use ds1000_zkvm;
pub use risc0_zkvm;

use alloc::{vec, vec::Vec};
use risc0_zkvm::guest::env;

/// Reads the kernel parameters from RISC Zero's private input stream.
pub struct Risc0Input;

impl ds1000_zkvm::Input for Risc0Input {
    fn read_words(&mut self) -> Vec<i64> {
        let mut len: u32 = 0;
        env::read_slice(core::slice::from_mut(&mut len));
        let mut words = vec![0i64; len as usize];
        env::read_slice(&mut words);
        words
    }
}

/// Commits `output` to the journal.
pub fn commit<T: ds1000_zkvm::Publics>(output: &T) {
    env::commit_slice(&output.encode_to_vec());
}

/// The guest for one case: read the private inputs, run the kernel, and
/// commit its output to the journal.
#[macro_export]
macro_rules! guest {
    ($case:ident) => {
        fn main() {
            let output = $crate::ds1000_zkvm::$case::main(&mut $crate::Risc0Input);
            $crate::commit(&output);
        }
    };
}
