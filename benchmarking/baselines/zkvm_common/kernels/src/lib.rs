//! The benchmark kernels as Q32 integer Rust programs, shared by the SP1 and
//! RISC Zero guests and run natively for the output check.
//!
//! Two kinds of program:
//!
//! * **Fixed-shape** (DS-1000 performance set): `datasets/ds1000/<case>/zkvm.rs`,
//!   compiled as the module `<case>`. It provides `pub type Output` and
//!   `pub fn main(input: &mut impl Input) -> Output`, which reads every kernel
//!   parameter in signature order and computes the kernel. Its shapes are
//!   constants of the program, as in `kernel.py`.
//! * **Shape-generic** (NPBench, Pythran): `datasets/{npbench,pythran}/<program>/zkvm.rs`,
//!   compiled as the module `<program>`, with one public submodule per entry
//!   function, each providing `Output` and
//!   `pub fn main(args: &mut Args<impl Input>) -> Output`. Arrays carry their
//!   shapes ([`Args`]); the guest commits the input shapes with the output.
//!
//! A case is a fixed-shape program or one entry function of a shape-generic
//! program (key `<program>/<function>`). The guest binaries call `main` and
//! commit the result ([`publics`]); the hosts decode it from the proof.
//!
//! A case imports [`prelude`]; NumPy behaviours that several cases share
//! live in [`np`].
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

pub use q32::{C32, Q32};

#[cfg(feature = "std")]
pub mod fixture;
#[cfg(feature = "std")]
pub mod host;
pub mod io;
pub mod np;
pub mod publics;

pub use io::{Args, Element, Input};
pub use publics::{
    decode_all, words_as_le_bytes, words_from_le_bytes, DecodeError, Public, Publics, Shaped,
};

/// What every case module imports.
pub mod prelude {
    pub use crate::np::{self, np_index};
    pub use crate::{Args, Input, Shaped};
    pub use alloc::{vec, vec::Vec};
    pub use ndarray::{
        s, Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayViewMut1,
        ArrayViewMut2, Axis, Zip,
    };
    pub use q32::{dot, int::floor_div, int::floor_mod, C32, Q32};
}

/// The words a fixed-shape program commits: its output.
pub fn run_fixed<I: Input, O: Publics>(input: &mut I, main: impl FnOnce(&mut I) -> O) -> Vec<i64> {
    main(input).encode_to_vec()
}

/// The words a shape-generic program commits: the input shapes it read,
/// then its output.
pub fn run_shaped<I: Input, O: Publics>(
    input: &mut I,
    main: impl FnOnce(&mut Args<I>) -> O,
) -> Vec<i64> {
    let mut args = Args::new(input);
    let output = main(&mut args);
    (args.into_shapes(), output).encode_to_vec()
}

/// A case's committed words, decoded.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// The dimensions of every array parameter, in order (shape-generic
    /// programs only).
    pub input_shapes: Vec<i64>,
    pub publics: Vec<Public>,
}

/// Declares the case modules and the dispatch functions over them. The
/// lists are generated into `cases.rs` by `zkvm_common/sync_cases.py`.
macro_rules! programs {
    (
        modules { $($module:ident => $path:literal;)* }
        fixed { $($fixed:ident $([unused $($fixed_unused:literal),*])?;)* }
        shaped { $($key:literal => $($entry:ident)::+ $([unused $($shaped_unused:literal),*])?;)* }
    ) => {
        $(
            #[path = $path]
            pub mod $module;
        )*

        /// Every case: fixed-shape programs by name, shape-generic entry
        /// functions as `<program>/<function>`.
        pub const CASES: &[&str] = &[$(stringify!($fixed),)* $($key,)*];

        /// Whether `case` is an entry function of a shape-generic program.
        pub fn is_shaped(case: &str) -> bool {
            [$($key),*].contains(&case)
        }

        /// Kernel parameters that `case` never reads, declared in its
        /// program by a line `// unused parameters: <names>`. Zinnia removes
        /// such inputs from the circuit, so the guest does not take them
        /// either: the hosts leave them out of the private input.
        pub fn unused_params(case: &str) -> &'static [&'static str] {
            match case {
                $(stringify!($fixed) => &[$($($fixed_unused),*)?],)*
                $($key => &[$($($shaped_unused),*)?],)*
                _ => &[],
            }
        }

        /// Decodes the words `case` committed; `None` if the case is
        /// unknown.
        pub fn decode_publics(case: &str, words: &[i64]) -> Option<Result<crate::Decoded, crate::DecodeError>> {
            match case {
                $(stringify!($fixed) => Some(crate::decode_all::<$fixed::Output>(words).map(|output| {
                    crate::Decoded { input_shapes: alloc::vec::Vec::new(), publics: crate::Publics::flatten(&output) }
                })),)*
                $($key => Some(crate::decode_all::<(alloc::vec::Vec<i64>, $($entry)::+::Output)>(words).map(
                    |(input_shapes, output)| crate::Decoded { input_shapes, publics: crate::Publics::flatten(&output) },
                )),)*
                _ => None,
            }
        }

        /// Runs `case` on `input` and returns the words it commits; `None`
        /// if the case is unknown.
        pub fn run(case: &str, input: &mut impl crate::Input) -> Option<alloc::vec::Vec<i64>> {
            match case {
                $(stringify!($fixed) => Some(crate::run_fixed(input, |i| $fixed::main(i))),)*
                $($key => Some(crate::run_shaped(input, |a| $($entry)::+::main(a))),)*
                _ => None,
            }
        }
    };
}

mod cases;
pub use cases::*;

/// The case for `program` (a case folder's name) and an optional entry
/// function: the fixed-shape program itself, or the program's entry function
/// (which may be omitted when the program has exactly one).
pub fn resolve_case(program: &str, function: Option<&str>) -> Result<&'static str, &'static str> {
    if let Some(&case) = CASES.iter().find(|&&c| c == program) {
        return match function {
            None => Ok(case),
            Some(_) => Err("a fixed-shape program has no entry functions to choose from"),
        };
    }
    let mut entries = CASES.iter().copied().filter(|c| {
        c.split_once('/').is_some_and(|(p, f)| p == program && function.is_none_or(|want| f == want))
    });
    match (entries.next(), entries.next()) {
        (Some(case), None) => Ok(case),
        (None, _) => Err("no program for this case folder and function"),
        (Some(_), Some(_)) => Err("the program has several entry functions: choose one with --function"),
    }
}

/// The guest binary of `case` (`/` is not allowed in binary names).
pub fn binary_name(case: &str) -> alloc::string::String {
    case.replace('/', "__")
}
