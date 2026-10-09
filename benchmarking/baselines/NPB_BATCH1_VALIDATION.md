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

## Fixture validation (2026-10-09)

All 19 entries pass `hw_check.check` on both backends: 38 passing results.
The checker compares the A/B public outputs with NumPy, generates halo2 keys
with A and proves B, verifies honest proofs, and rejects tampered public values
and out-of-domain private inputs. All recorded public outputs are finite.
Halo2 reports zero redundant input checks. All 19 entries pass static lint
with zero errors; its warnings concern precondition widths/input loading in
covariance2 and the fixed three-coordinate rank in sumarray3d. The five new
Prover files were checked against the fixture-B writer and compiled Noir ABI.

Commands: from `benchmarking/baselines`, build the 19 batch-one IDs with
`HALO2_HW_CASES=<comma-separated IDs> cargo build --profile dev-fast` in
`halo2_hw`, then `python -m hw_check.check <IDs> --halo2-bin
halo2_hw/target/dev-fast/halo2_hw`. Heavy processes used the shared checker
lock, `memrun.sh` with 8G maximum, and two Cargo build jobs. No proving-time
measurements are inferred from the dev-fast validation build.

| Entry | Halo2 advice | Halo2 lookup | Noir gates |
|---|---:|---:|---:|
| `npbench_deep_learning_mlp_mlp__mlp` | 8,126 | 1,864 | 4,082 |
| `npbench_deep_learning_softmax_softmax__softmax` | 454,976 | 87,552 | 89,623 |
| `npbench_go_fast_go_fast__go_fast` | 14,766 | 3,152 | 5,748 |
| `pythran_log_likelihood__log_likelihood` | 3,226 | 814 | 3,304 |
| `pythran_vibr_energy__calculate_vibr_energy` | 8,624 | 1,856 | 4,223 |
| `npbench_polybench_covariance2_covariance2__kernel` | 22,042 | 4,456 | 9,145 |
| `npbench_polybench_gramschmidt_gramschmidt__kernel` | 71,286 | 17,700 | 17,852 |
| `npbench_polybench_symm_symm__kernel` | 66,487 | 17,044 | 13,596 |
| `npbench_polybench_trmm_trmm__kernel` | 29,285 | 7,096 | 8,507 |
| `pythran_deuxd_convolution__conv` | 9,071 | 1,256 | 4,911 |
| `pythran_sumarray3d__summation` | 71,440 | 17,280 | 14,204 |
| `npbench_polybench_bicg_bicg__kernel` | 8,836 | 2,244 | 4,479 |
| `npbench_polybench_k3mm_k3mm__kernel` | 75,323 | 14,256 | 19,559 |
| `pythran_create_grid__create_grid` | 432 | 128 | 3,908 |
| `pythran_l2norm__l2_norm` | 9,856 | 2,640 | 4,508 |
| `pythran_multi_export__a` | 46 | 14 | 2,822 |
| `pythran_pairwise__pairwise` | 26,496 | 5,464 | 7,922 |
| `pythran_reverse_cumsum__reverse_cumsum` | 496 | 140 | 2,887 |
| `pythran_wdist__slow_wdist` | 3,732 | 963 | 3,442 |
