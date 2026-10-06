//! The array-operator lowering layer (see `eval-redo/LOWERING-LAYER.md`).
//!
//! Families of constructions, one module each. The framework (placeholders,
//! selection, emission) is added by its own branch; until then the scatter
//! family's constructions are generator functions the frontend calls.

pub mod scatter;
