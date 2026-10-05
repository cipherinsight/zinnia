//! The DS-1000 performance-set kernels as Cairo programs over Q32 integers.
//! Each case is `datasets/ds1000/<case>/main.cairo`, linked in as
//! `src/cases/<case>.cairo`, with one executable target per case.
pub mod cases;
pub mod q32;
