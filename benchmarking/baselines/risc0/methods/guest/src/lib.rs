//! RISC Zero guest support: the private-input reader and the `guest!` macro
//! that every per-case binary in `src/bin/` expands.
//!
//! Each kernel parameter arrives as a frame (`env::read_slice` of a `u32`
//! word count, then of the 64-bit words: two reads, as RISC Zero recommends
//! for bulk data instead of `env::read`'s per-element deserialization), and
//! the output is committed as one raw buffer (`env::commit_slice`); see
//! `zkvm_kernels` for the layout.
#![no_std]

extern crate alloc;

pub use risc0_zkvm;
pub use zkvm_kernels;

use alloc::{vec, vec::Vec};
use risc0_zkvm::guest::env;

/// Reads the kernel parameters from RISC Zero's private input stream.
pub struct Risc0Input;

impl zkvm_kernels::Input for Risc0Input {
    fn read_words(&mut self) -> Vec<i64> {
        let mut len: u32 = 0;
        env::read_slice(core::slice::from_mut(&mut len));
        let mut words = vec![0i64; len as usize];
        env::read_slice(&mut words);
        words
    }
}

/// Commits the case's words to the journal.
pub fn commit(words: &[i64]) {
    env::commit_slice(words);
}

/// The guest for one case: read the private inputs, run the kernel, and
/// commit its output (with the input shapes, for a shape-generic program).
#[macro_export]
macro_rules! guest {
    (fixed $($case:ident)::+) => {
        fn main() {
            let words = $crate::zkvm_kernels::run_fixed(&mut $crate::Risc0Input, |input| {
                $crate::zkvm_kernels::$($case)::+::main(input)
            });
            $crate::commit(&words);
        }
    };
    (shaped $($case:ident)::+) => {
        fn main() {
            let words = $crate::zkvm_kernels::run_shaped(&mut $crate::Risc0Input, |args| {
                $crate::zkvm_kernels::$($case)::+::main(args)
            });
            $crate::commit(&words);
        }
    };
}
