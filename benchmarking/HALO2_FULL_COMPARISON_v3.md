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
| 0291 | 10 | 140ms | 327ms | 2688B | ✓ | 10 | 1.09s | 2532B | 102 | ✓ |
| 0298 ⚠️ | — | n/a | n/a | n/a | — | 10 | 3.94s | 17276B | 9269 | ✓ |
| 0300 | 13 | 578ms | 1.32s | 2880B | — | 10 | 6.77s | 34364B | 22164 | ✓ |
| 0301 | 10 | 98ms | 317ms | 2688B | ✓ | 10 | 1.97s | 4612B | 569 | ✓ |
| 0302 | 10 | 89ms | 318ms | 2688B | ✓ | 10 | 1.95s | 4612B | 569 | ✓ |
| 0303 | 10 | 100ms | 362ms | 2688B | ✓ | 10 | 1.53s | 7248B | 2640 | ✓ |
| 0304 | 10 | 138ms | 324ms | 2688B | ✓ | 10 | 1.88s | 7248B | 2464 | ✓ |
| 0309 | 10 | 185ms | 316ms | 2688B | ✓ | 10 | 2.73s | 10316B | 4655 | ✓ |
| 0310 | 10 | 94ms | 327ms | 2688B | ✓ | 10 | 3.33s | 10316B | 4655 | ✓ |
| 0311 | 10 | 140ms | 344ms | 2688B | ✓ | 10 | 2.61s | 11056B | 5172 | ✓ |
| 0312 | 10 | 117ms | 346ms | 2688B | ✓ | 10 | 4.49s | 11056B | 5172 | ✓ |
| 0313 | 10 | 18ms | 343ms | 2688B | ✓ | 10 | 1.88s | 10316B | 4655 | ✓ |
| 0314 | 10 | 179ms | 358ms | 2688B | ✓ | 10 | 5.58s | 24016B | 15798 | ✓ |
| 0317 | 10 | 55ms | 284ms | 1376B | ✓ | 10 | 907ms | 34468B | 1000 | ✓ |
| 0318 | — | n/a | n/a | n/a | — | 10 | 1.61s | 11796B | 6588 | ✓ |
| 0319 | — | n/a | n/a | n/a | — | 10 | 6.31s | 46992B | 27408 | ✓ |
| 0320 | 10 | 101ms | 353ms | 2688B | ✓ | 10 | 2.71s | 7612B | 3390 | ✓ |
| 0321 | 10 | 139ms | 328ms | 2688B | ✓ | 10 | 3.12s | 7612B | 3390 | ✓ |
| 0322 ⚠️ | 10 | 119ms | 332ms | 2688B | ✓ | 10 | 2.17s | 3940B | 655 | ✓ |
| 0323 | 10 | 124ms | 327ms | 2688B | ✓ | 10 | 2.96s | 13760B | 6999 | ✓ |
| 0324 | — | n/a | n/a | n/a | — | 10 | 2.94s | 12188B | 6847 | ✓ |
| 0325 | — | n/a | n/a | n/a | — | 10 | 4.37s | 22112B | 13608 | ✓ |
| 0326 | 10 | 136ms | 337ms | 2688B | ✓ | 10 | 5.28s | 14924B | 7369 | ✓ |
| 0329 | 15 | 2.09s | 2.74s | 3008B | ✓ | 10 | 399.51s | 1192044B | 967599 | ✓ |
| 0330 | 15 | 0ms | 2.62s | 3008B | ✓ | 10 | 324.13s | 1192044B | 967599 | ✓ |
| 0334 | 10 | 169ms | 318ms | 2688B | ✓ | 10 | 34.63s | 124492B | 82551 | ✓ |
| 0335 | 10 | 152ms | 317ms | 2688B | ✓ | 10 | 3.54s | 18364B | 9150 | ✓ |
| 0336 | 10 | 109ms | 327ms | 2688B | ✓ | 10 | 595ms | 2724B | 64 | ✓ |
| 0337 | 10 | 133ms | 320ms | 2688B | ✓ | 10 | 1.07s | 2724B | 80 | ✓ |
| 0338 | 10 | 149ms | 307ms | 2688B | ✓ | 10 | 701ms | 2980B | 64 | ✓ |
| 0339 | 10 | 172ms | 325ms | 2688B | ✓ | 10 | 1.39s | 2980B | 80 | ✓ |
| 0340 | 10 | 0ms | 249ms | 1376B | ✓ | 10 | 705ms | 3428B | 30 | ✓ |
| 0341 | 10 | 0ms | 262ms | 1376B | ✓ | 10 | 1.29s | 3428B | 30 | ✓ |
| 0342 | 10 | 0ms | 263ms | 1376B | ✓ | 10 | 1.04s | 3428B | 30 | ✓ |
| 0343 | 10 | 25ms | 263ms | 1376B | ✓ | 10 | 882ms | 3428B | 30 | ✓ |
| 0344 | 10 | 21ms | 250ms | 1376B | ✓ | 10 | 1.32s | 2500B | 1 | ✓ |
| 0345 | 10 | 131ms | 361ms | 2688B | ✓ | 10 | 2.22s | 6640B | 2497 | ✓ |
| 0346 | 10 | 130ms | 314ms | 2688B | ✓ | 10 | 2.23s | 6576B | 2177 | ✓ |
| 0347 | 10 | 152ms | 337ms | 2688B | ✓ | 10 | n/a | n/a | — | — |
| 0349 | 10 | 130ms | 308ms | 2688B | ✓ | 10 | 582ms | 2500B | 49 | ✓ |
| 0353 | 10 | 120ms | 325ms | 2688B | ✓ | 10 | 1.74s | 6640B | 2633 | ✓ |
| 0355 | 10 | 90ms | 319ms | 2688B | ✓ | 10 | 3.18s | 9120B | 4285 | ✓ |
| 0356 | 10 | 0ms | 326ms | 2688B | ✓ | 10 | 2.36s | 9120B | 4285 | ✓ |
| 0357 | 10 | 145ms | 307ms | 2688B | ✓ | 10 | 3.80s | 9120B | 4285 | ✓ |
| 0358 | 10 | 108ms | 337ms | 2688B | ✓ | 10 | 2.04s | 6540B | 1629 | ✓ |
| 0359 | 10 | 158ms | 319ms | 2688B | ✓ | 10 | 2.09s | 2660B | 9 | ✓ |
| 0360 | 10 | 128ms | 328ms | 2688B | ✓ | 10 | 1.21s | 2852B | 16 | ✓ |
| 0361 | 10 | 129ms | 298ms | 2688B | ✓ | 10 | 1.56s | 3236B | 36 | ✓ |
| 0363 | 13 | 371ms | 1.46s | 2880B | ✓ | 10 | 4.79s | 21868B | 11470 | ✓ |
| 0364 | 16 | 3.14s | 5.26s | 3072B | ✓ | 10 | 8.21s | 64768B | 26766 | ✓ |
| 0365 | 13 | 74ms | 1.27s | 2880B | ✓ | 10 | 3.77s | 21868B | 11470 | ✓ |
| 0367 | 10 | 110ms | 336ms | 2688B | ✓ | 10 | 1.44s | 2564B | 9 | ✓ |
| 0368 | 10 | 133ms | 307ms | 2688B | ✓ | 10 | 3.29s | 10316B | 4322 | ✓ |
| 0369 | 10 | 132ms | 329ms | 2688B | ✓ | 10 | 3.40s | 10316B | 4322 | ✓ |
| 0370 | 10 | 0ms | 333ms | 2688B | ✓ | 10 | 2.25s | 10316B | 4322 | ✓ |
| 0373 | 10 | 119ms | 305ms | 2688B | ✓ | 10 | 119.30s | 286048B | 202151 | ✓ |
| 0374 | 14 | 812ms | 1.96s | 2944B | ✗ | 10 | 134.00s | 328096B | 235263 | ✓ |
| 0375 | 14 | 994ms | 2.03s | 2944B | ✓ | 10 | 113.43s | 295264B | 208433 | ✓ |
| 0381 | 16 | 3.76s | 5.84s | 3072B | ✓ | 10 | 253.17s | 737372B | 564500 | ✓ |
| 0382 | 16 | 2.66s | 6.42s | 3072B | ✓ | 10 | 145.96s | 737372B | 564500 | ✓ |
| 0383 ⚠️ | 16 | 3.33s | 5.81s | 3072B | ✓ | 10 | 12.63s | 734492B | 564513 | ✓ |
| 0385 | 15 | 328ms | 2.08s | 1696B | ✓ | 15 | 1.40s | 642468B | 20000 | ✓ |
| 0387 | 15 | 516ms | 2.21s | 1696B | ✓ | 15 | 1.28s | 642468B | 20000 | ✓ |
| 0388 ⚠️ | 18 | 20.20s | 23.09s | 3200B | ✓ | 15 | 1.46s | 643276B | 20013 | ✓ |
| 0389 ⚠️ | 10 | 171ms | 327ms | 2688B | ✗ | 10 | 507ms | 23676B | 12266 | ✓ |
| 0390 ⚠️ | 19 | 32.74s | 36.38s | 3264B | ✓ | 15 | 16.52s | 791548B | 3400013 | ✓ |
| 0391 ⚠️ | 13 | 564ms | 1.30s | 2880B | ✓ | 10 | 468ms | 24640B | 13828 | ✓ |
| 0392 ⚠️ | 13 | 358ms | 1.39s | 2880B | ✓ | 10 | 624ms | 31052B | 17208 | ✓ |
| 0393 ⚠️ | 13 | 0ms | 1.35s | 2880B | ✗ | 10 | 790ms | 41548B | 24070 | ✓ |
| 0398 ⚠️ | — | n/a | n/a | n/a | — | 10 | 236ms | 9408B | 3989 | ✓ |
| 0399 ⚠️ | — | n/a | n/a | n/a | — | 10 | 277ms | 10572B | 4957 | ✓ |
| 0400 | 10 | 79ms | 247ms | 1376B | ✓ | 10 | 79ms | 1288B | — | ✓ |
| 0401 | 10 | 0ms | 256ms | 1376B | ✓ | 10 | 82ms | 1288B | — | ✓ |
| 0402 | 13 | 454ms | 1.69s | 2880B | ✓ | 10 | 577ms | 45212B | 36940 | ✓ |
| 0403 | 13 | 0ms | 2.15s | 2880B | ✓ | 10 | 639ms | 45212B | 36919 | ✓ |
| 0409 | 10 | 112ms | 338ms | 2688B | ✓ | 10 | 182ms | 4812B | 288 | ✓ |
| 0412 | 10 | 139ms | 333ms | 2688B | ✓ | 10 | 191ms | 4164B | 951 | ✓ |
| 0413 | 10 | 91ms | 321ms | 2688B | ✓ | 10 | 118ms | 2468B | 10 | ✓ |
| 0414 | 10 | 147ms | 318ms | 2688B | ✓ | 10 | 302ms | 12284B | 6585 | ✓ |
| 0415 | 10 | 97ms | 302ms | 2688B | ✓ | 10 | 168ms | 3972B | 368 | ✓ |
| 0416 | 10 | 92ms | 366ms | 2688B | ✓ | 10 | 1.33s | 61584B | 39505 | ✓ |
| 0417 | 10 | 105ms | 344ms | 2688B | ✓ | 10 | 335ms | 12284B | 6585 | ✓ |
| 0418 | 10 | 118ms | 354ms | 2688B | ✓ | 10 | 1.28s | 61584B | 39505 | ✓ |
| 0419 | 10 | 177ms | 348ms | 2688B | ✓ | 10 | 1.27s | 61584B | 39505 | ✓ |
| 0422 | 14 | 1.02s | 1.82s | 2944B | ✓ | 10 | 478ms | 25852B | 17750 | ✓ |
| 0428 | 11 | 189ms | 521ms | 2752B | ✓ | 10 | 1.47s | 73120B | 54990 | ✓ |
| 0429 | 11 | 175ms | 518ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0430 | 11 | 45ms | 484ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0431 | 10 | 130ms | 318ms | 2688B | ✓ | 10 | 306ms | 12412B | 6491 | ✓ |
| 0433 | 10 | 91ms | 332ms | 2688B | ✓ | 10 | 123ms | 3788B | 535 | ✓ |
| 0434 | — | n/a | n/a | n/a | — | 10 | 109ms | 3788B | 937 | ✓ |
| 0435 | 10 | 123ms | 331ms | 2688B | ✓ | 10 | 118ms | 3788B | 17 | ✓ |
| 0436 | 10 | 119ms | 357ms | 2688B | ✓ | 10 | 484ms | 22972B | 12250 | ✓ |
| 0437 | 10 | 165ms | 319ms | 2688B | ✓ | 10 | 524ms | 22972B | 12250 | ✓ |
| 0438 | 10 | 123ms | 338ms | 2688B | ✓ | — | n/a | n/a | — | — |
| 0439 | 15 | 2.01s | 2.82s | 3008B | ✓ | 11 | 6.22s | 292720B | 312200 | ✓ |
| 0441 | 10 | 126ms | 318ms | 2688B | ✓ | 10 | 112ms | 3308B | 77 | ✓ |
| 0442 | 10 | 80ms | 336ms | 2688B | ✓ | 10 | 111ms | 5112B | 1385 | ✓ |
| 0443 | 10 | 113ms | 320ms | 2688B | ✓ | 10 | 262ms | 15588B | 11037 | ✓ |
| 0444 | 10 | 143ms | 325ms | 2688B | ✓ | 10 | 469ms | 22188B | 16877 | ✓ |
| 0448 | 10 | 15ms | 261ms | 1376B | ✓ | 10 | 106ms | 3236B | 24 | ✓ |
| 0449 | 10 | 3ms | 241ms | 1376B | ✓ | 10 | 98ms | 3236B | 24 | ✓ |
| 0450 | 10 | 182ms | 341ms | 2688B | ✓ | 10 | 890ms | 65864B | 53821 | ✓ |
| 0451 | 10 | 67ms | 232ms | 1376B | ✓ | 10 | 107ms | 2500B | 1 | ✓ |
| 0452 | 10 | 112ms | 333ms | 2688B | ✓ | 10 | 1.12s | 56140B | 35641 | ✓ |
| 0453 | 10 | 185ms | 319ms | 2688B | ✓ | 10 | 1.23s | 60748B | 38506 | ✓ |
| 0454 | 10 | 171ms | 299ms | 2688B | ✓ | 10 | 1.20s | 56848B | 36931 | ✓ |
| 0459 | 10 | 111ms | 330ms | 2688B | ✓ | 10 | 173ms | 5708B | 1667 | ✓ |
| 0460 | 10 | 11ms | 306ms | 2688B | ✓ | 10 | 196ms | 5708B | 1667 | ✓ |
| 0462 | 10 | 165ms | 292ms | 2688B | ✓ | 10 | 269ms | 15588B | 10699 | ✓ |
| 0463 | 10 | 133ms | 318ms | 2688B | ✗ | 10 | 279ms | 15588B | 10699 | ✓ |
| 0467 ⚠️ | 10 | 136ms | 314ms | 2688B | ✓ | 10 | 133ms | 8184B | 1581 | ✓ |
| 0468 ⚠️ | 10 | 0ms | 345ms | 2688B | ✓ | 10 | 141ms | 8184B | 1581 | ✓ |
| 0469 | 10 | 146ms | 306ms | 2688B | ✓ | 10 | 201ms | 7612B | 3293 | ✓ |
| 0470 | 10 | 39ms | 312ms | 2688B | ✓ | 10 | 230ms | 7612B | 3293 | ✓ |
| 0472 | 11 | 160ms | 827ms | 2752B | ✓ | 11 | 293ms | 55508B | 9072 | ✓ |
| 0473 | 10 | 134ms | 332ms | 2688B | ✓ | 10 | 101ms | 3308B | 412 | ✓ |
| 0478 | 10 | 107ms | 337ms | 2688B | ✓ | 10 | 109ms | 3404B | 17 | ✓ |
| 0480 | 10 | 99ms | 335ms | 2688B | ✓ | 10 | 146ms | 3876B | 942 | ✓ |
| 0481 | 10 | 177ms | 314ms | 2688B | ✓ | 10 | 144ms | 5924B | 2392 | ✓ |
| 0485 | 11 | 196ms | 592ms | 2752B | ✓ | 10 | 577ms | 27696B | 14949 | ✓ |
| 0490 ⚠️ | — | n/a | n/a | n/a | — | 10 | 837ms | 48688B | 31715 | ✓ |
| 0491 | 12 | 135ms | 913ms | 2816B | ✓ | 10 | 214ms | 6416B | 2639 | ✓ |
| 0492 | 12 | 11ms | 1.00s | 2816B | ✓ | 10 | 213ms | 6416B | 2639 | ✓ |
| 0500 | 10 | 49ms | 244ms | 1376B | ✓ | 10 | 108ms | 4196B | 54 | ✓ |
| 0501 | 12 | 227ms | 1.11s | 2816B | ✓ | 10 | 142ms | 9848B | 1859 | ✓ |
| 0502 | 12 | 0ms | 1.11s | 2816B | ✓ | 10 | 154ms | 9848B | 1859 | ✓ |
| 0503 | 12 | 307ms | 1.12s | 2816B | ✓ | 10 | 332ms | 24980B | 14534 | ✓ |
| 0504 | 12 | 192ms | 1.18s | 2816B | ✓ | 10 | 323ms | 20768B | 15211 | ✓ |
| 0505 | 13 | 445ms | 1.36s | 2880B | ✓ | 10 | 375ms | 24260B | 18591 | ✓ |
| 0507 | 11 | 200ms | 554ms | 2752B | ✓ | 10 | 236ms | 9408B | 4357 | ✓ |
| 0508 | 14 | 786ms | 1.69s | 2944B | ✓ | 10 | 400ms | 21872B | 13154 | ✓ |
| 0509 | 12 | 216ms | 1.03s | 2816B | ✓ | 10 | 837ms | 44912B | 28755 | ✓ |
| 0510 | 12 | 385ms | 1.02s | 2816B | ✓ | 10 | 789ms | 46076B | 29155 | ✓ |

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

- Zinnia / HW prove-time ratio (on 108 both-verified cases): median **0.37×**, range 0.00× – 7.84×
- Zinnia slower: **45 (42%)**; Zinnia faster: **63 (58%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 57 | 53% |
| Zinnia 0.5-1× | 6 | 6% |
| Zinnia 1-1.5× | 10 | 9% |
| Zinnia 1.5-3× | 20 | 19% |
| Zinnia 3-10× | 15 | 14% |
| Zinnia >10× | 0 | 0% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0501 | 1.11s | 142ms | 7.84× |
| 0502 | 1.11s | 154ms | 7.22× |
| 0492 | 1.00s | 213ms | 4.71× |
| 0491 | 913ms | 214ms | 4.27× |
| 0508 | 1.69s | 400ms | 4.22× |
| 0422 | 1.82s | 478ms | 3.80× |
| 0504 | 1.18s | 323ms | 3.65× |
| 0505 | 1.36s | 375ms | 3.62× |
| 0503 | 1.12s | 332ms | 3.38× |
| 0403 | 2.15s | 639ms | 3.36× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 305ms | 119.30s | 0.003× | 390.6× |
| 0329 | 2.74s | 399.51s | 0.007× | 146.0× |
| 0330 | 2.62s | 324.13s | 0.008× | 123.8× |
| 0334 | 318ms | 34.63s | 0.009× | 109.0× |
| 0375 | 2.03s | 113.43s | 0.018× | 55.8× |
| 0381 | 5.84s | 253.17s | 0.023× | 43.4× |
| 0382 | 6.42s | 145.96s | 0.044× | 22.7× |
| 0326 | 337ms | 5.28s | 0.064× | 15.7× |
| 0314 | 358ms | 5.58s | 0.064× | 15.6× |
| 0312 | 346ms | 4.49s | 0.077× | 13.0× |
