//! No-class (global) operators: abs, len, min, max, sum, any, all, range, print,
//! pow, list, tuple, poseidon_hash_builtin, merkle_verify.
//! Ports `zinnia/op_def/nocls/`.

pub mod abs;
pub mod len;
pub mod min;
pub mod max;
pub mod print;
pub mod list;
pub mod tuple;
pub mod pow;
pub mod sum;
pub mod any;
pub mod all;
pub mod range;
pub mod poseidon_hash_builtin;
pub mod merkle_verify;

pub use abs::AbsOp;
pub use len::LenOp;
pub use min::MinOp;
pub use max::MaxOp;
pub use print::PrintOp;
pub use list::ListOp;
pub use tuple::TupleOp;
pub use pow::PowOp;
pub use sum::SumOp;
pub use any::AnyOp;
pub use all::AllOp;
pub use range::RangeOp;
pub use poseidon_hash_builtin::PoseidonHashBuiltinOp;
pub use merkle_verify::MerkleVerifyOp;
