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
| 0291 | 10 | 30ms | 364ms | 2688B | ✓ | 10 | 1.09s | 2532B | 102 | ✓ |
| 0298 ⚠️ | — | n/a | n/a | n/a | — | 10 | 3.94s | 17276B | 9269 | ✓ |
| 0300 | 13 | 683ms | 1.40s | 2880B | — | 10 | 6.77s | 34364B | 22164 | ✓ |
| 0301 | 10 | 132ms | 292ms | 2688B | ✓ | 10 | 1.97s | 4612B | 569 | ✓ |
| 0302 | 10 | 103ms | 338ms | 2688B | ✓ | 10 | 1.95s | 4612B | 569 | ✓ |
| 0303 | 10 | 198ms | 342ms | 2688B | ✓ | 10 | 1.53s | 7248B | 2640 | ✓ |
| 0304 | 10 | 91ms | 345ms | 2688B | ✓ | 10 | 1.88s | 7248B | 2464 | ✓ |
| 0309 | 10 | 96ms | 330ms | 2688B | ✓ | 10 | 2.73s | 10316B | 4655 | ✓ |
| 0310 | 10 | 126ms | 348ms | 2688B | ✓ | 10 | 3.33s | 10316B | 4655 | ✓ |
| 0311 | 10 | 120ms | 337ms | 2688B | ✓ | 10 | 2.61s | 11056B | 5172 | ✓ |
| 0312 | 10 | 147ms | 308ms | 2688B | ✓ | 10 | 4.49s | 11056B | 5172 | ✓ |
| 0313 | 10 | 0ms | 320ms | 2688B | ✓ | 10 | 1.88s | 10316B | 4655 | ✓ |
| 0314 | 10 | 148ms | 371ms | 2688B | ✓ | 10 | 5.58s | 24016B | 15798 | ✓ |
| 0317 | 10 | 31ms | 283ms | 1376B | ✓ | 10 | 907ms | 34468B | 1000 | ✓ |
| 0318 | — | n/a | n/a | n/a | — | 10 | 1.61s | 11796B | 6588 | ✓ |
| 0319 | — | n/a | n/a | n/a | — | 10 | 6.31s | 46992B | 27408 | ✓ |
| 0320 | 10 | 167ms | 317ms | 2688B | ✓ | 10 | 2.71s | 7612B | 3390 | ✓ |
| 0321 | 10 | 128ms | 330ms | 2688B | ✓ | 10 | 3.12s | 7612B | 3390 | ✓ |
| 0322 ⚠️ | 10 | 113ms | 311ms | 2688B | ✓ | 10 | 2.17s | 3940B | 655 | ✓ |
| 0323 | 10 | 160ms | 327ms | 2688B | ✓ | 10 | 2.96s | 13760B | 6999 | ✓ |
| 0324 | — | n/a | n/a | n/a | — | 10 | 2.94s | 12188B | 6847 | ✓ |
| 0325 | — | n/a | n/a | n/a | — | 10 | 4.37s | 22112B | 13608 | ✓ |
| 0326 | 10 | 215ms | 307ms | 2688B | ✓ | 10 | 5.28s | 14924B | 7369 | ✓ |
| 0329 | 15 | 2.20s | 2.86s | 3008B | ✓ | 10 | 399.51s | 1192044B | 967599 | ✓ |
| 0330 | 15 | 90ms | 2.88s | 3008B | ✓ | 10 | 324.13s | 1192044B | 967599 | ✓ |
| 0334 | 10 | 130ms | 340ms | 2688B | ✓ | 10 | 34.63s | 124492B | 82551 | ✓ |
| 0335 | 10 | 164ms | 307ms | 2688B | ✓ | 10 | 3.54s | 18364B | 9150 | ✓ |
| 0336 | 10 | 92ms | 323ms | 2688B | ✓ | 10 | 595ms | 2724B | 64 | ✓ |
| 0337 | 10 | 90ms | 323ms | 2688B | ✓ | 10 | 1.07s | 2724B | 80 | ✓ |
| 0338 | 10 | 157ms | 310ms | 2688B | ✓ | 10 | 701ms | 2980B | 64 | ✓ |
| 0339 | 10 | 89ms | 317ms | 2688B | ✓ | 10 | 1.39s | 2980B | 80 | ✓ |
| 0340 | 10 | 43ms | 251ms | 1376B | ✓ | 10 | 705ms | 3428B | 30 | ✓ |
| 0341 | 10 | 2ms | 254ms | 1376B | ✓ | 10 | 1.29s | 3428B | 30 | ✓ |
| 0342 | 10 | 0ms | 243ms | 1376B | ✓ | 10 | 1.04s | 3428B | 30 | ✓ |
| 0343 | 10 | 6ms | 256ms | 1376B | ✓ | 10 | 882ms | 3428B | 30 | ✓ |
| 0344 | 10 | 37ms | 237ms | 1376B | ✓ | 10 | 1.32s | 2500B | 1 | ✓ |
| 0345 | 10 | 152ms | 308ms | 2688B | ✓ | 10 | 2.22s | 6640B | 2497 | ✓ |
| 0346 | 10 | 81ms | 343ms | 2688B | ✓ | 10 | 2.23s | 6576B | 2177 | ✓ |
| 0347 | 10 | 71ms | 333ms | 2688B | ✓ | 10 | n/a | n/a | — | — |
| 0349 | 10 | 116ms | 320ms | 2688B | ✓ | 10 | 582ms | 2500B | 49 | ✓ |
| 0353 | 10 | 83ms | 298ms | 2688B | ✓ | 10 | 1.74s | 6640B | 2633 | ✓ |
| 0355 | 10 | 171ms | 306ms | 2688B | ✓ | 10 | 3.18s | 9120B | 4285 | ✓ |
| 0356 | 10 | 0ms | 349ms | 2688B | ✓ | 10 | 2.36s | 9120B | 4285 | ✓ |
| 0357 | 10 | 122ms | 331ms | 2688B | ✓ | 10 | 3.80s | 9120B | 4285 | ✓ |
| 0358 | 10 | 79ms | 324ms | 2688B | ✓ | 10 | 2.04s | 6540B | 1629 | ✓ |
| 0359 | 10 | 82ms | 311ms | 2688B | ✓ | 10 | 2.09s | 2660B | 9 | ✓ |
| 0360 | 10 | 198ms | 307ms | 2688B | ✓ | 10 | 1.21s | 2852B | 16 | ✓ |
| 0361 | 10 | 143ms | 291ms | 2688B | ✓ | 10 | 1.56s | 3236B | 36 | ✓ |
| 0363 | 13 | 555ms | 1.36s | 2880B | ✓ | 10 | 4.79s | 21868B | 11470 | ✓ |
| 0364 | 16 | 4.05s | 5.12s | 3072B | ✓ | 10 | 8.21s | 64768B | 26766 | ✓ |
| 0365 | 13 | 60ms | 1.25s | 2880B | ✓ | 10 | 3.77s | 21868B | 11470 | ✓ |
| 0367 | 10 | 130ms | 297ms | 2688B | ✓ | 10 | 1.44s | 2564B | 9 | ✓ |
| 0368 | 10 | 136ms | 304ms | 2688B | ✓ | 10 | 3.29s | 10316B | 4322 | ✓ |
| 0369 | 10 | 108ms | 347ms | 2688B | ✓ | 10 | 3.40s | 10316B | 4322 | ✓ |
| 0370 | 10 | 20ms | 311ms | 2688B | ✓ | 10 | 2.25s | 10316B | 4322 | ✓ |
| 0373 | 10 | 105ms | 317ms | 2688B | ✓ | 10 | 119.30s | 286048B | 202151 | ✓ |
| 0374 | 14 | 893ms | 1.83s | 2944B | ✗ | 10 | 134.00s | 328096B | 235263 | ✓ |
| 0375 | 14 | 602ms | 1.97s | 2944B | ✓ | 10 | 113.43s | 295264B | 208433 | ✓ |
| 0381 | 16 | 2.89s | 5.76s | 3072B | ✓ | 10 | 253.17s | 737372B | 564500 | ✓ |
| 0382 | 16 | 5.47s | 5.74s | 3072B | ✓ | 10 | 145.96s | 737372B | 564500 | ✓ |
| 0383 ⚠️ | 16 | 2.99s | 5.51s | 3072B | ✓ | 10 | 12.63s | 734492B | 564513 | ✓ |
| 0385 | 15 | 127ms | 2.45s | 1696B | ✓ | 15 | 1.40s | 642468B | 20000 | ✓ |
| 0387 | 15 | 43ms | 2.36s | 1696B | ✓ | 15 | 1.28s | 642468B | 20000 | ✓ |
| 0388 ⚠️ | 18 | 23.36s | 22.69s | 3200B | ✓ | 15 | 1.46s | 643276B | 20013 | ✓ |
| 0389 ⚠️ | 10 | 87ms | 332ms | 2688B | ✗ | 10 | 507ms | 23676B | 12266 | ✓ |
| 0390 ⚠️ | 19 | 38.02s | 35.52s | 3264B | ✓ | 15 | 16.52s | 791548B | 3400013 | ✓ |
| 0391 ⚠️ | 13 | 500ms | 1.38s | 2880B | ✓ | 10 | 468ms | 24640B | 13828 | ✓ |
| 0392 ⚠️ | 13 | 560ms | 1.35s | 2880B | ✓ | 10 | 624ms | 31052B | 17208 | ✓ |
| 0393 ⚠️ | 13 | 0ms | 1.41s | 2880B | ✗ | 10 | 790ms | 41548B | 24070 | ✓ |
| 0398 ⚠️ | — | n/a | n/a | n/a | — | 10 | 236ms | 9408B | 3989 | ✓ |
| 0399 ⚠️ | — | n/a | n/a | n/a | — | 10 | 277ms | 10572B | 4957 | ✓ |
| 0400 | 10 | 21ms | 237ms | 1376B | ✓ | 10 | 79ms | 1288B | — | ✓ |
| 0401 | 10 | 0ms | 235ms | 1376B | ✓ | 10 | 82ms | 1288B | — | ✓ |
| 0402 | 13 | 454ms | 1.98s | 2880B | ✓ | 10 | 577ms | 45212B | 36940 | ✓ |
| 0403 | 13 | 270ms | 2.11s | 2880B | ✓ | 10 | 639ms | 45212B | 36919 | ✓ |
| 0409 | 10 | 114ms | 318ms | 2688B | ✓ | 10 | 182ms | 4812B | 288 | ✓ |
| 0412 | 10 | 122ms | 305ms | 2688B | ✓ | 10 | 191ms | 4164B | 951 | ✓ |
| 0413 | 10 | 150ms | 304ms | 2688B | ✓ | 10 | 118ms | 2468B | 10 | ✓ |
| 0414 | 10 | 113ms | 311ms | 2688B | ✓ | 10 | 302ms | 12284B | 6585 | ✓ |
| 0415 | 10 | 151ms | 299ms | 2688B | ✓ | 10 | 168ms | 3972B | 368 | ✓ |
| 0416 | 10 | 188ms | 339ms | 2688B | ✓ | 10 | 1.33s | 61584B | 39505 | ✓ |
| 0417 | 10 | 123ms | 337ms | 2688B | ✓ | 10 | 335ms | 12284B | 6585 | ✓ |
| 0418 | 10 | 108ms | 350ms | 2688B | ✓ | 10 | 1.28s | 61584B | 39505 | ✓ |
| 0419 | 10 | 114ms | 340ms | 2688B | ✓ | 10 | 1.27s | 61584B | 39505 | ✓ |
| 0422 | 14 | 835ms | 1.84s | 2944B | ✓ | 10 | 478ms | 25852B | 17750 | ✓ |
| 0428 | 11 | 138ms | 553ms | 2752B | ✓ | 10 | 1.47s | 73120B | 54990 | ✓ |
| 0429 | 11 | 231ms | 556ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0430 | 11 | 0ms | 555ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0431 | 10 | 105ms | 306ms | 2688B | ✓ | 10 | 306ms | 12412B | 6491 | ✓ |
| 0433 | 10 | 114ms | 315ms | 2688B | ✓ | 10 | 123ms | 3788B | 535 | ✓ |
| 0434 | — | n/a | n/a | n/a | — | 10 | 109ms | 3788B | 937 | ✓ |
| 0435 | 10 | 121ms | 335ms | 2688B | ✓ | 10 | 118ms | 3788B | 17 | ✓ |
| 0436 | 10 | 120ms | 327ms | 2688B | ✓ | 10 | 484ms | 22972B | 12250 | ✓ |
| 0437 | 10 | 170ms | 310ms | 2688B | ✓ | 10 | 524ms | 22972B | 12250 | ✓ |
| 0438 | 10 | 165ms | 314ms | 2688B | ✓ | — | n/a | n/a | — | — |
| 0439 | 15 | 1.65s | 3.10s | 3008B | ✓ | 11 | 6.22s | 292720B | 312200 | ✓ |
| 0441 | 10 | 183ms | 303ms | 2688B | ✓ | 10 | 112ms | 3308B | 77 | ✓ |
| 0442 | 10 | 95ms | 309ms | 2688B | ✓ | 10 | 111ms | 5112B | 1385 | ✓ |
| 0443 | 10 | 88ms | 334ms | 2688B | ✓ | 10 | 262ms | 15588B | 11037 | ✓ |
| 0444 | 10 | 191ms | 304ms | 2688B | ✓ | 10 | 469ms | 22188B | 16877 | ✓ |
| 0448 | 10 | 5ms | 249ms | 1376B | ✓ | 10 | 106ms | 3236B | 24 | ✓ |
| 0449 | 10 | 16ms | 235ms | 1376B | ✓ | 10 | 98ms | 3236B | 24 | ✓ |
| 0450 | 10 | 118ms | 347ms | 2688B | ✓ | 10 | 890ms | 65864B | 53821 | ✓ |
| 0451 | 10 | 18ms | 244ms | 1376B | ✓ | 10 | 107ms | 2500B | 1 | ✓ |
| 0452 | 10 | 150ms | 332ms | 2688B | ✓ | 10 | 1.12s | 56140B | 35641 | ✓ |
| 0453 | 10 | 117ms | 326ms | 2688B | ✓ | 10 | 1.23s | 60748B | 38506 | ✓ |
| 0454 | 10 | 156ms | 315ms | 2688B | ✓ | 10 | 1.20s | 56848B | 36931 | ✓ |
| 0459 | 10 | 99ms | 329ms | 2688B | ✓ | 10 | 173ms | 5708B | 1667 | ✓ |
| 0460 | 10 | 6ms | 322ms | 2688B | ✓ | 10 | 196ms | 5708B | 1667 | ✓ |
| 0462 | 10 | 139ms | 318ms | 2688B | ✓ | 10 | 269ms | 15588B | 10699 | ✓ |
| 0463 | 10 | 120ms | 327ms | 2688B | ✗ | 10 | 279ms | 15588B | 10699 | ✓ |
| 0467 ⚠️ | 10 | 127ms | 323ms | 2688B | ✓ | 10 | 133ms | 8184B | 1581 | ✓ |
| 0468 ⚠️ | 10 | 4ms | 325ms | 2688B | ✓ | 10 | 141ms | 8184B | 1581 | ✓ |
| 0469 | 10 | 130ms | 301ms | 2688B | ✓ | 10 | 201ms | 7612B | 3293 | ✓ |
| 0470 | 10 | 24ms | 308ms | 2688B | ✓ | 10 | 230ms | 7612B | 3293 | ✓ |
| 0472 | 11 | 280ms | 738ms | 2752B | ✓ | 11 | 293ms | 55508B | 9072 | ✓ |
| 0473 | 10 | 141ms | 303ms | 2688B | ✓ | 10 | 101ms | 3308B | 412 | ✓ |
| 0478 | 10 | 134ms | 307ms | 2688B | ✓ | 10 | 109ms | 3404B | 17 | ✓ |
| 0480 | 10 | 107ms | 341ms | 2688B | ✓ | 10 | 146ms | 3876B | 942 | ✓ |
| 0481 | 10 | 146ms | 297ms | 2688B | ✓ | 10 | 144ms | 5924B | 2392 | ✓ |
| 0485 | 11 | 131ms | 521ms | 2752B | ✓ | 10 | 577ms | 27696B | 14949 | ✓ |
| 0490 ⚠️ | — | n/a | n/a | n/a | — | 10 | 837ms | 48688B | 31715 | ✓ |
| 0491 | 10 | 141ms | 299ms | 2688B | ✓ | 10 | 214ms | 6416B | 2639 | ✓ |
| 0492 | 10 | 0ms | 339ms | 2688B | ✓ | 10 | 213ms | 6416B | 2639 | ✓ |
| 0500 | 10 | 21ms | 246ms | 1376B | ✓ | 10 | 108ms | 4196B | 54 | ✓ |
| 0501 | 11 | 160ms | 568ms | 2752B | ✓ | 10 | 156ms | 11012B | 2535 | ✓ |
| 0502 | 11 | 0ms | 592ms | 2752B | ✓ | 10 | 164ms | 11012B | 2535 | ✓ |
| 0503 | 12 | 173ms | 988ms | 2816B | ✓ | 10 | 501ms | 24188B | 15717 | ✓ |
| 0504 | 12 | 282ms | 1.15s | 2816B | ✓ | 10 | 444ms | 19520B | 16394 | ✓ |
| 0505 | 13 | 380ms | 1.28s | 2880B | ✓ | 10 | 466ms | 22556B | 19774 | ✓ |
| 0507 | 11 | 195ms | 563ms | 2752B | ✓ | 10 | 236ms | 9408B | 4357 | ✓ |
| 0508 | 14 | 928ms | 1.88s | 2944B | ✓ | 10 | 400ms | 21872B | 13154 | ✓ |
| 0509 | 12 | 563ms | 965ms | 2816B | ✓ | 10 | 837ms | 44912B | 28755 | ✓ |
| 0510 | 12 | 523ms | 972ms | 2816B | ✓ | 10 | 789ms | 46076B | 29155 | ✓ |

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

- Zinnia / HW prove-time ratio (on 108 both-verified cases): median **0.38×**, range 0.00× – 4.70×
- Zinnia slower: **43 (40%)**; Zinnia faster: **65 (60%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 57 | 53% |
| Zinnia 0.5-1× | 8 | 7% |
| Zinnia 1-1.5× | 9 | 8% |
| Zinnia 1.5-3× | 28 | 26% |
| Zinnia 3-10× | 6 | 6% |
| Zinnia >10× | 0 | 0% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0508 | 1.88s | 400ms | 4.70× |
| 0422 | 1.84s | 478ms | 3.86× |
| 0501 | 568ms | 156ms | 3.65× |
| 0502 | 592ms | 164ms | 3.61× |
| 0402 | 1.98s | 577ms | 3.44× |
| 0403 | 2.11s | 639ms | 3.30× |
| 0400 | 237ms | 79ms | 2.99× |
| 0473 | 303ms | 101ms | 2.99× |
| 0401 | 235ms | 82ms | 2.88× |
| 0435 | 335ms | 118ms | 2.84× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 317ms | 119.30s | 0.003× | 376.1× |
| 0329 | 2.86s | 399.51s | 0.007× | 139.5× |
| 0330 | 2.88s | 324.13s | 0.009× | 112.6× |
| 0334 | 340ms | 34.63s | 0.010× | 101.8× |
| 0375 | 1.97s | 113.43s | 0.017× | 57.5× |
| 0381 | 5.76s | 253.17s | 0.023× | 43.9× |
| 0382 | 5.74s | 145.96s | 0.039× | 25.4× |
| 0326 | 307ms | 5.28s | 0.058× | 17.2× |
| 0314 | 371ms | 5.58s | 0.067× | 15.0× |
| 0312 | 308ms | 4.49s | 0.069× | 14.6× |
