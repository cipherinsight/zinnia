//! Gather family of the array-operator lowering layer: reads of an array at
//! runtime indices (element gathers, slices and crops with runtime bounds)
//! and the index computations that feed them (`argwhere` / `flatnonzero`
//! followed by `min` / `max`).
//!
//! Each construction is a generator from operands (and the facts it relies
//! on) to scalar IR, documented with its preconditions, postconditions,
//! cost and soundness argument:
//!
//! * [`scan`]: first / last set position of a flag vector by running
//!   products, recording the result's one-hot and thermometer;
//! * [`element`]: an element read at an index whose one-hot is recorded, as
//!   an inner product with that one-hot;
//! * [`window`]: a slice with runtime bounds as a shift by a shared one-hot
//!   per axis;
//! * [`vanishing`]: which cells a flag being 0 forces to 0 (lets a window
//!   skip the mask on padding that is 0 anyway);
//! * [`encoding`]: the index encodings (one-hot, thermometer) constructions
//!   record as postconditions and gathers consume.

pub mod element;
pub mod encoding;
pub mod scan;
pub mod vanishing;
pub mod window;

#[cfg(test)]
mod tests;
