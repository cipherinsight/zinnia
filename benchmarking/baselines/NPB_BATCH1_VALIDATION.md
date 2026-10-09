# Batch 1 validation and numerical limits

The 19 entries listed as batch 1 in `NPB_BASELINES.md` have handwritten
halo2-base and Noir sources and generated input-B Prover files. These sources
were completed from the kernels and shared handwritten gadgets, without reading
Zinnia's emitted circuits, IR, or counts.

`log_likelihood` uses the algebraic identity
`log(exp(-s) / scale) = -s - log(scale)`, with `log(scale)` computed once.
Squares and the division defining `s` still use the shared Q32 rounding. This
avoids premature Q32 underflow of the density and substantially reduces the
circuit. With sixteen data values of 10, mean 0, and sigma 1, NumPy's density
is finite but smaller than a Q32 unit. Both baselines match the finite NumPy
log-likelihood. The regression is in `hw_check/domain_edges.json`, along with
rejection of zero and negative sigma; all three pass on both backends.

## Known numerical limitations

Passing the fixture and edge checks is not evidence of full-domain equivalence.
In particular, these remaining issues prevent that claim:

- The shared exp approximation can cross the signed Q32 upper boundary while
  the mathematical result still fits. For example, Q32 input code 92288378412
  (about 21.4875625474) has a real exponential of about 2147483540.797, below
  2^31, but the shared polynomial produces code 9223372114164340035, above
  2^63-1. The gadget rejects this value. Its source already documents the
  affected interval near 31 ln(2). Both backends use this approximation.
- Algebraic log-likelihood deliberately avoids Q32 density materialization,
  but does not emulate binary64 exponential underflow. For example, sixteen
  data values of 40, mean 0, and sigma 1 give s=800. Literal NumPy computes
  exp(-800)=0 and then log(0)=-inf; the algebraic circuit instead has a finite
  log-density. It therefore differs from the rule requiring rejection wherever
  the literal NumPy computation is nonfinite. A future numerical-domain policy
  or an explicit binary64 underflow check is needed before claiming that rule.
- As with other shared Q32 baselines, intermediate square/product/division
  window checks can reject inputs even when algebraic cancellation would make
  the final mathematical answer fit. No fixture-derived input restrictions
  were added to conceal these limitations.

The stable identity is preferable for the measured finite fixtures, but these
limits must accompany any claim that the implementation supports every input
allowed by the source signature. Validation here concerns correctness at the
checked inputs and proof soundness checks, not performance timings.
