//! No-class (global) operators: abs, len, min, max, sum, any, all, range, print,
//! pow, list, tuple, poseidon_hash_builtin, merkle_verify.
//! Ports `zinnia/op_def/nocls/`.

pub mod abs;
pub mod all;
pub mod any;
pub mod len;
pub mod list;
pub mod max;
pub mod merkle_verify;
pub mod min;
pub mod poseidon_hash_builtin;
pub mod pow;
pub mod print;
pub mod range;
pub mod sum;
pub mod tuple;

pub use abs::AbsOp;
pub use all::AllOp;
pub use any::AnyOp;
pub use len::LenOp;
pub use list::ListOp;
pub use max::MaxOp;
pub use merkle_verify::MerkleVerifyOp;
pub use min::MinOp;
pub use poseidon_hash_builtin::PoseidonHashBuiltinOp;
pub use pow::PowOp;
pub use print::PrintOp;
pub use range::RangeOp;
pub use sum::SumOp;
pub use tuple::TupleOp;
