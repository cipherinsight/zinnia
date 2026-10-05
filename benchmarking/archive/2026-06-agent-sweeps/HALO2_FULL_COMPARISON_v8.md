# Zinnia halo2 vs handwritten halo2-graph — full DS-1000 sweep (v2)

v2 methodology: a single long-lived Python process for the Zinnia sweep,
so the in-memory proving-key cache stays warm across cases. Per case:
one untimed warm-up prove (does keygen + 1 prove), then 3 timed warm
proves. `keygen` = first_prove − median(warm_proves); `prove` =
median(warm_proves). This matches HW's `cargo run prove` flow, which
loads `data/case*.pk` from disk and only times the prove step.

Both backends: precision_bits=32; lookup_bits=8 (Zinnia) / 6 (HW).

| Case | Zin k | Zin keygen | Zin prove | Zin proof | Zin v | HW k | HW prove | HW proof | HW adv | HW v |
|------|------:|-----------:|----------:|----------:|:-----:|-----:|---------:|---------:|-------:|:----:|
| 0291 | 10 | 32ms | 228ms | 1376B | ✓ | 10 | 106ms | 2532B | 2 | ✓ |
| 0298 ⚠️ | — | n/a | n/a | n/a | — | 10 | 3.94s | 17276B | 9269 | ✓ |
| 0300 | 13 | 782ms | 1.24s | 2880B | — | 10 | 6.77s | 34364B | 22164 | ✓ |
| 0301 | 10 | 79ms | 293ms | 2688B | ✓ | 10 | 1.97s | 4612B | 569 | ✓ |
| 0302 | 10 | 78ms | 343ms | 2688B | ✓ | 10 | 1.95s | 4612B | 569 | ✓ |
| 0303 | 10 | 147ms | 327ms | 2688B | ✓ | 10 | 1.53s | 7248B | 2640 | ✓ |
| 0304 | 10 | 106ms | 320ms | 2688B | ✓ | 10 | 1.88s | 7248B | 2464 | ✓ |
| 0309 | 10 | 159ms | 330ms | 2688B | ✓ | 10 | 2.73s | 10316B | 4655 | ✓ |
| 0310 | 10 | 124ms | 320ms | 2688B | ✓ | 10 | 3.33s | 10316B | 4655 | ✓ |
| 0311 | 10 | 89ms | 329ms | 2688B | ✓ | 10 | 2.61s | 11056B | 5172 | ✓ |
| 0312 | 10 | 88ms | 339ms | 2688B | ✓ | 10 | 4.49s | 11056B | 5172 | ✓ |
| 0313 | 10 | 0ms | 310ms | 2688B | ✓ | 10 | 1.88s | 10316B | 4655 | ✓ |
| 0314 | 10 | 106ms | 349ms | 2688B | ✓ | 10 | 5.58s | 24016B | 15798 | ✓ |
| 0317 | 10 | 38ms | 277ms | 1376B | ✓ | 10 | 907ms | 34468B | 1000 | ✓ |
| 0318 | — | n/a | n/a | n/a | — | 10 | 1.61s | 11796B | 6588 | ✓ |
| 0319 | — | n/a | n/a | n/a | — | 10 | 6.31s | 46992B | 27408 | ✓ |
| 0320 | 10 | 102ms | 326ms | 2688B | ✓ | 10 | 2.71s | 7612B | 3390 | ✓ |
| 0321 | 10 | 88ms | 314ms | 2688B | ✓ | 10 | 3.12s | 7612B | 3390 | ✓ |
| 0322 ⚠️ | 10 | 148ms | 297ms | 2688B | ✓ | 10 | 2.17s | 3940B | 655 | ✓ |
| 0323 | 10 | 130ms | 306ms | 2688B | ✓ | 10 | 2.96s | 13760B | 6999 | ✓ |
| 0324 | — | n/a | n/a | n/a | — | 10 | 2.94s | 12188B | 6847 | ✓ |
| 0325 | — | n/a | n/a | n/a | — | 10 | 4.37s | 22112B | 13608 | ✓ |
| 0326 | 10 | 112ms | 310ms | 2688B | ✓ | 10 | 5.28s | 14924B | 7369 | ✓ |
| 0329 | 15 | 2.52s | 2.75s | 3008B | ✓ | 10 | 399.51s | 1192044B | 967599 | ✓ |
| 0330 | 15 | 0ms | 2.77s | 3008B | ✓ | 10 | 324.13s | 1192044B | 967599 | ✓ |
| 0334 | 10 | 126ms | 298ms | 2688B | ✓ | 10 | 34.63s | 124492B | 82551 | ✓ |
| 0335 | 10 | 168ms | 326ms | 2688B | ✓ | 10 | 3.54s | 18364B | 9150 | ✓ |
| 0336 | 10 | 14ms | 240ms | 1376B | ✓ | 10 | 99ms | 2724B | 8 | ✓ |
| 0337 | 10 | 3ms | 250ms | 1376B | ✓ | 10 | 117ms | 2724B | 8 | ✓ |
| 0338 | 10 | 29ms | 232ms | 1376B | ✓ | 10 | 117ms | 2980B | 16 | ✓ |
| 0339 | 10 | 48ms | 242ms | 1376B | ✓ | 10 | 106ms | 2980B | 16 | ✓ |
| 0340 | 10 | 19ms | 246ms | 1376B | ✓ | 10 | 705ms | 3428B | 30 | ✓ |
| 0341 | 10 | 0ms | 248ms | 1376B | ✓ | 10 | 1.29s | 3428B | 30 | ✓ |
| 0342 | 10 | 6ms | 238ms | 1376B | ✓ | 10 | 1.04s | 3428B | 30 | ✓ |
| 0343 | 10 | 11ms | 241ms | 1376B | ✓ | 10 | 882ms | 3428B | 30 | ✓ |
| 0344 | 10 | 13ms | 248ms | 1376B | ✓ | 10 | 1.32s | 2500B | 1 | ✓ |
| 0345 | 10 | 137ms | 310ms | 2688B | ✓ | 10 | 2.22s | 6640B | 2497 | ✓ |
| 0346 | 10 | 121ms | 337ms | 2688B | ✓ | 10 | 2.23s | 6576B | 2177 | ✓ |
| 0347 | 10 | 132ms | 285ms | 2688B | ✓ | 10 | n/a | n/a | — | — |
| 0349 | 10 | 76ms | 237ms | 1376B | ✓ | 10 | 114ms | 2500B | 1 | ✓ |
| 0353 | 10 | 98ms | 311ms | 2688B | ✓ | 10 | 1.74s | 6640B | 2633 | ✓ |
| 0355 | 10 | 139ms | 301ms | 2688B | ✓ | 10 | 3.18s | 9120B | 4285 | ✓ |
| 0356 | 10 | 0ms | 359ms | 2688B | ✓ | 10 | 2.36s | 9120B | 4285 | ✓ |
| 0357 | 10 | 113ms | 321ms | 2688B | ✓ | 10 | 3.80s | 9120B | 4285 | ✓ |
| 0358 | 10 | 144ms | 313ms | 2688B | ✓ | 10 | 2.04s | 6540B | 1629 | ✓ |
| 0359 | 10 | 20ms | 231ms | 1376B | ✓ | 10 | 98ms | 2660B | 6 | ✓ |
| 0360 | 10 | 34ms | 226ms | 1376B | ✓ | 10 | 126ms | 2852B | 12 | ✓ |
| 0361 | 10 | 53ms | 238ms | 1376B | ✓ | 10 | 101ms | 3236B | 24 | ✓ |
| 0363 | 13 | 366ms | 1.24s | 2880B | ✓ | 10 | 4.79s | 21868B | 11470 | ✓ |
| 0364 | 16 | 4.69s | 5.02s | 3072B | ✓ | 10 | 8.21s | 64768B | 26766 | ✓ |
| 0365 | 13 | 0ms | 1.38s | 2880B | ✓ | 10 | 3.77s | 21868B | 11470 | ✓ |
| 0367 | 10 | 21ms | 232ms | 1376B | ✓ | 10 | 107ms | 2564B | 3 | ✓ |
| 0368 | 10 | 112ms | 323ms | 2688B | ✓ | 10 | 3.29s | 10316B | 4322 | ✓ |
| 0369 | 10 | 41ms | 350ms | 2688B | ✓ | 10 | 3.40s | 10316B | 4322 | ✓ |
| 0370 | 10 | 24ms | 296ms | 2688B | ✓ | 10 | 2.25s | 10316B | 4322 | ✓ |
| 0373 | 10 | 137ms | 311ms | 2688B | ✓ | 10 | 119.30s | 286048B | 202151 | ✓ |
| 0374 | 14 | 729ms | 1.90s | 2944B | ✗ | 10 | 134.00s | 328096B | 235263 | ✓ |
| 0375 | 14 | 717ms | 1.95s | 2944B | ✓ | 10 | 113.43s | 295264B | 208433 | ✓ |
| 0381 | 16 | 4.03s | 5.54s | 3072B | ✓ | 10 | 253.17s | 737372B | 564500 | ✓ |
| 0382 | 16 | 2.96s | 5.46s | 3072B | ✓ | 10 | 145.96s | 737372B | 564500 | ✓ |
| 0383 ⚠️ | 16 | 3.29s | 5.58s | 3072B | ✓ | 10 | 12.63s | 734492B | 564513 | ✓ |
| 0385 | 15 | 286ms | 2.08s | 1696B | ✓ | 15 | 1.40s | 642468B | 20000 | ✓ |
| 0387 | 15 | 7ms | 2.24s | 1696B | ✓ | 15 | 1.28s | 642468B | 20000 | ✓ |
| 0388 ⚠️ | 18 | 15.77s | 20.90s | 3200B | ✓ | 15 | 1.46s | 643276B | 20013 | ✓ |
| 0389 ⚠️ | 10 | 60ms | 340ms | 2688B | ✗ | 10 | 507ms | 23676B | 12266 | ✓ |
| 0390 ⚠️ | 18 | 13.48s | 22.11s | 3200B | ✓ | 15 | 16.52s | 791548B | 3400013 | ✓ |
| 0391 ⚠️ | 12 | 282ms | 999ms | 2816B | ✓ | 10 | 468ms | 24640B | 13828 | ✓ |
| 0392 ⚠️ | 12 | 365ms | 1.02s | 2816B | ✓ | 10 | 624ms | 31052B | 17208 | ✓ |
| 0393 ⚠️ | 13 | 55ms | 1.34s | 2880B | ✗ | 10 | 790ms | 41548B | 24070 | ✓ |
| 0398 ⚠️ | — | n/a | n/a | n/a | — | 10 | 236ms | 9408B | 3989 | ✓ |
| 0399 ⚠️ | — | n/a | n/a | n/a | — | 10 | 277ms | 10572B | 4957 | ✓ |
| 0400 | 10 | 39ms | 239ms | 1376B | ✓ | 10 | 79ms | 1288B | — | ✓ |
| 0401 | 10 | 34ms | 236ms | 1376B | ✓ | 10 | 82ms | 1288B | — | ✓ |
| 0402 | 10 | 19ms | 425ms | 2688B | ✓ | 10 | 113ms | 3308B | 76 | ✓ |
| 0403 | 10 | 38ms | 365ms | 2688B | ✓ | 10 | 114ms | 3308B | 55 | ✓ |
| 0409 | 10 | 198ms | 320ms | 2688B | ✓ | 10 | 182ms | 4812B | 288 | ✓ |
| 0412 | 10 | 203ms | 318ms | 2688B | ✓ | 10 | 191ms | 4164B | 951 | ✓ |
| 0413 | 10 | 39ms | 215ms | 1376B | ✓ | 10 | 84ms | 1288B | — | ✓ |
| 0414 | 10 | 109ms | 300ms | 2688B | ✓ | 10 | 302ms | 12284B | 6585 | ✓ |
| 0415 | 10 | 37ms | 339ms | 2688B | ✓ | 10 | 168ms | 3972B | 368 | ✓ |
| 0416 | 10 | 139ms | 334ms | 2688B | ✓ | 10 | 1.33s | 61584B | 39505 | ✓ |
| 0417 | 10 | 166ms | 300ms | 2688B | ✓ | 10 | 335ms | 12284B | 6585 | ✓ |
| 0418 | 10 | 135ms | 335ms | 2688B | ✓ | 10 | 1.28s | 61584B | 39505 | ✓ |
| 0419 | 10 | 131ms | 313ms | 2688B | ✓ | 10 | 1.27s | 61584B | 39505 | ✓ |
| 0422 | 14 | 835ms | 1.68s | 2944B | ✓ | 10 | 478ms | 25852B | 17750 | ✓ |
| 0428 | 11 | 201ms | 490ms | 2752B | ✓ | 10 | 1.47s | 73120B | 54990 | ✓ |
| 0429 | 11 | 87ms | 537ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0430 | 11 | 0ms | 517ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0431 | 10 | 60ms | 295ms | 2688B | ✓ | 10 | 306ms | 12412B | 6491 | ✓ |
| 0433 | 10 | 82ms | 315ms | 2688B | ✓ | 10 | 123ms | 3788B | 535 | ✓ |
| 0434 | — | n/a | n/a | n/a | — | 10 | 109ms | 3788B | 937 | ✓ |
| 0435 | 10 | 243ms | 290ms | 2688B | ✓ | 10 | 118ms | 3788B | 17 | ✓ |
| 0436 | 10 | 183ms | 293ms | 2688B | ✓ | 10 | 484ms | 22972B | 12250 | ✓ |
| 0437 | 10 | 112ms | 313ms | 2688B | ✓ | 10 | 524ms | 22972B | 12250 | ✓ |
| 0438 | 10 | 181ms | 320ms | 2688B | ✓ | — | n/a | n/a | — | — |
| 0439 | 15 | 1.89s | 3.00s | 3008B | ✓ | 11 | 6.22s | 292720B | 312200 | ✓ |
| 0441 | 10 | 125ms | 296ms | 2688B | ✓ | 10 | 112ms | 3308B | 77 | ✓ |
| 0442 | 10 | 106ms | 298ms | 2688B | ✓ | 10 | 111ms | 5112B | 1385 | ✓ |
| 0443 | 10 | 122ms | 292ms | 2688B | ✓ | 10 | 262ms | 15588B | 11037 | ✓ |
| 0444 | 10 | 141ms | 337ms | 2688B | ✓ | 10 | 469ms | 22188B | 16877 | ✓ |
| 0448 | 10 | 0ms | 257ms | 1376B | ✓ | 10 | 106ms | 3236B | 24 | ✓ |
| 0449 | 10 | 35ms | 227ms | 1376B | ✓ | 10 | 98ms | 3236B | 24 | ✓ |
| 0450 | 10 | 104ms | 393ms | 2688B | ✓ | 10 | 890ms | 65864B | 53821 | ✓ |
| 0451 | 10 | 15ms | 234ms | 1376B | ✓ | 10 | 107ms | 2500B | 1 | ✓ |
| 0452 | 10 | 123ms | 325ms | 2688B | ✓ | 10 | 1.12s | 56140B | 35641 | ✓ |
| 0453 | 10 | 117ms | 322ms | 2688B | ✓ | 10 | 1.23s | 60748B | 38506 | ✓ |
| 0454 | 10 | 134ms | 312ms | 2688B | ✓ | 10 | 1.20s | 56848B | 36931 | ✓ |
| 0459 | 10 | 138ms | 279ms | 2688B | ✓ | 10 | 185ms | 5708B | 1662 | ✓ |
| 0460 | 10 | 0ms | 280ms | 2688B | ✓ | 10 | 205ms | 5708B | 1662 | ✓ |
| 0462 | 10 | 105ms | 282ms | 2688B | ✓ | 10 | 269ms | 15588B | 10699 | ✓ |
| 0463 | 10 | 158ms | 310ms | 2688B | ✗ | 10 | 279ms | 15588B | 10699 | ✓ |
| 0467 ⚠️ | 10 | 0ms | 236ms | 1376B | ✓ | 10 | 133ms | 8184B | 1581 | ✓ |
| 0468 ⚠️ | 10 | 0ms | 240ms | 1376B | ✓ | 10 | 141ms | 8184B | 1581 | ✓ |
| 0469 | 10 | 100ms | 266ms | 2688B | ✓ | 10 | 201ms | 7612B | 3293 | ✓ |
| 0470 | 10 | 3ms | 257ms | 2688B | ✓ | 10 | 230ms | 7612B | 3293 | ✓ |
| 0472 | 11 | 0ms | 478ms | 1440B | ✓ | 11 | 211ms | 50852B | 1512 | ✓ |
| 0473 | 10 | 0ms | 249ms | 1376B | ✓ | 10 | 116ms | 3308B | 1 | ✓ |
| 0478 | 10 | 137ms | 290ms | 2688B | ✓ | 10 | 109ms | 3404B | 17 | ✓ |
| 0480 | 10 | 160ms | 306ms | 2688B | ✓ | 10 | 146ms | 3876B | 942 | ✓ |
| 0481 | 10 | 139ms | 289ms | 2688B | ✓ | 10 | 144ms | 5924B | 2392 | ✓ |
| 0485 | 11 | 222ms | 530ms | 2752B | ✓ | 10 | 577ms | 27696B | 14949 | ✓ |
| 0490 ⚠️ | — | n/a | n/a | n/a | — | 10 | 837ms | 48688B | 31715 | ✓ |
| 0491 | 10 | 125ms | 280ms | 2688B | ✓ | 10 | 214ms | 6416B | 2639 | ✓ |
| 0492 | 10 | 105ms | 321ms | 2688B | ✓ | 10 | 213ms | 6416B | 2639 | ✓ |
| 0500 | 10 | 39ms | 224ms | 1376B | ✓ | 10 | 108ms | 4196B | 54 | ✓ |
| 0501 | 11 | 204ms | 503ms | 2752B | ✓ | 10 | 156ms | 11012B | 2535 | ✓ |
| 0502 | 11 | 0ms | 543ms | 2752B | ✓ | 10 | 164ms | 11012B | 2535 | ✓ |
| 0503 | 12 | 197ms | 1.04s | 2816B | ✓ | 10 | 501ms | 24188B | 15717 | ✓ |
| 0504 | 12 | 278ms | 981ms | 2816B | ✓ | 10 | 444ms | 19520B | 16394 | ✓ |
| 0505 | 13 | 457ms | 1.29s | 2880B | ✓ | 10 | 466ms | 22556B | 19774 | ✓ |
| 0507 | 11 | 121ms | 563ms | 2752B | ✓ | 10 | 236ms | 9408B | 4357 | ✓ |
| 0508 | 14 | 546ms | 1.92s | 2944B | ✓ | 10 | 400ms | 21872B | 13154 | ✓ |
| 0509 | 12 | 300ms | 967ms | 2816B | ✓ | 10 | 837ms | 44912B | 28755 | ✓ |
| 0510 | 12 | 358ms | 1.00s | 2816B | ✓ | 10 | 789ms | 46076B | 29155 | ✓ |

## Summary

- Cases attempted: **134**
- Both sides verified (and apples-to-apples): **108**
- Zinnia-only verified: **4**
- HW-only verified: **8**
- Neither verified: **0**
- Excluded from headline (structurally unfair HW baseline): **14** (0298, 0322, 0383, 0388, 0389, 0390, 0391, 0392, 0393, 0398, 0399, 0467, 0468, 0490)

## Structurally unfair baselines (excluded ⚠️)

These cases were identified by sweeping every `target.rs` for
patterns where the handwritten halo2-graph baseline silently
implements a strictly weaker kernel than `kernel.py` declares.
Excluding them avoids letting HW's specialisation flatter the
comparison.

- **Category A — needs zkRAM (3 cases)**: the kernel uses a
  runtime `Integer` parameter inside indexing of a large static
  array (e.g. 100×200 patch extract with runtime `patch_size`).
  A faithful HW baseline in halo2-graph would need `select_from_idx`
  per access (~4N cells per access for an N-cell source). For a
  20k-cell input × 20k outputs that's 1.6B cells, k≥30, prove time
  in hours. **No tractable handwritten baseline exists; Zinnia's
  zkram is the only viable implementation.** HW dodges this by
  declaring `const PSZ = 4` (or similar) and asserting the runtime
  witness equals that constant.

- **Category B — runtime param pinned to a const (9 cases)**:
  the kernel declares an `Integer` parameter (or array of integers)
  but `target.rs` hardcodes a specific value and asserts the runtime
  witness equals it. A faithful HW baseline is *tractable* without
  zkRAM (typically O(N²) total work), but would cost ~5-50× the
  current specialised version. HW silently chose to specialise.

- **Category C — other silent kernel-contract downgrade (2 cases)**:
  HW changes the kernel contract in some non-equivalent way — e.g.
  using a host-provided claim as a compile-time loop bound (different
  proofs → different circuits), or truncating a variable-length output
  to a fixed-length witness.

| Case | Category & rationale |
|------|--------|
| 0298 | [C] HW reads claimed n_unique from input as compile-time loop bound; different proofs would yield different circuits |
| 0322 | [C] HW silently truncates output to first match + length; kernel returns variable-length list |
| 0383 | [B] HW hardcodes M=10 (top-N from 100-cell argsort); kernel has runtime N: Integer |
| 0388 | [A] HW hardcodes patch_size=4 for 100x200 patch-extract; faithful version needs zkRAM over 20k cells |
| 0389 | [B] HW hardcodes h=6, w=12 (reshape dims); kernel has runtime h, w: Integer |
| 0390 | [A] HW hardcodes patch_size=10 for 100x200 patch-extract; faithful version needs zkRAM over 20k cells |
| 0391 | [B] HW hardcodes low/high for column slice; kernel has runtime low, high: Integer |
| 0392 | [B] HW hardcodes low/high for row slice; kernel has runtime low, high: Integer |
| 0393 | [B] HW hardcodes low/high/clamp for column slice; kernel has runtime low, high: Integer |
| 0398 | [B] HW hardcodes N=8 for recurrence over array bounded at N<=64 |
| 0399 | [B] HW hardcodes N=8 for 3-tap recurrence over array bounded at N<=64 |
| 0467 | [B] HW hardcodes window size (3,3); kernel has runtime size: Integer[2] |
| 0468 | [B] HW hardcodes window size (3,3) col-major; kernel has runtime size: Integer[2] |
| 0490 | [A] HW hardcodes N=5, M=4, K=4 for array bounded N<=64,M<=64,K<=64; faithful version needs zkRAM |

- Zinnia / HW prove-time ratio (on 108 both-verified cases): median **0.91×**, range 0.00× – 4.80×
- Zinnia slower: **51 (47%)**; Zinnia faster: **57 (53%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 49 | 45% |
| Zinnia 0.5-1× | 8 | 7% |
| Zinnia 1-1.5× | 9 | 8% |
| Zinnia 1.5-3× | 35 | 32% |
| Zinnia 3-10× | 7 | 6% |
| Zinnia >10× | 0 | 0% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0508 | 1.92s | 400ms | 4.80× |
| 0402 | 425ms | 113ms | 3.75× |
| 0422 | 1.68s | 478ms | 3.52× |
| 0502 | 543ms | 164ms | 3.31× |
| 0501 | 503ms | 156ms | 3.23× |
| 0403 | 365ms | 114ms | 3.21× |
| 0400 | 239ms | 79ms | 3.01× |
| 0401 | 236ms | 82ms | 2.89× |
| 0505 | 1.29s | 466ms | 2.76× |
| 0442 | 298ms | 111ms | 2.68× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 311ms | 119.30s | 0.003× | 383.3× |
| 0329 | 2.75s | 399.51s | 0.007× | 145.2× |
| 0330 | 2.77s | 324.13s | 0.009× | 116.9× |
| 0334 | 298ms | 34.63s | 0.009× | 116.4× |
| 0375 | 1.95s | 113.43s | 0.017× | 58.2× |
| 0381 | 5.54s | 253.17s | 0.022× | 45.7× |
| 0382 | 5.46s | 145.96s | 0.037× | 26.7× |
| 0326 | 310ms | 5.28s | 0.059× | 17.1× |
| 0314 | 349ms | 5.58s | 0.062× | 16.0× |
| 0312 | 339ms | 4.49s | 0.075× | 13.3× |
