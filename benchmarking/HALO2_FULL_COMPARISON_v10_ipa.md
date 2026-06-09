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
| 0291 | 10 | 0ms | 248ms | 1376B | ✓ | 10 | 189ms | 1616B | 2 | ✓ |
| 0298 ⚠️ | — | n/a | n/a | n/a | — | 10 | 464ms | 6064B | 9269 | ✓ |
| 0300 | 13 | 782ms | 1.24s | 2880B | — | 10 | 764ms | 10576B | 22164 | ✓ |
| 0301 | 10 | 79ms | 293ms | 2688B | ✓ | 10 | 249ms | 2640B | 569 | ✓ |
| 0302 | 10 | 78ms | 343ms | 2688B | ✓ | 10 | 225ms | 2640B | 569 | ✓ |
| 0303 | 10 | 147ms | 327ms | 2688B | ✓ | 10 | 293ms | 3504B | 2640 | ✓ |
| 0304 | 10 | 106ms | 320ms | 2688B | ✓ | 10 | 260ms | 3504B | 2464 | ✓ |
| 0309 | 10 | 159ms | 330ms | 2688B | ✓ | 10 | 362ms | 3728B | 4655 | ✓ |
| 0310 | 10 | 124ms | 320ms | 2688B | ✓ | 10 | 343ms | 3728B | 4655 | ✓ |
| 0311 | 10 | 89ms | 329ms | 2688B | ✓ | 10 | 346ms | 3984B | 5172 | ✓ |
| 0312 | 10 | 88ms | 339ms | 2688B | ✓ | 10 | 348ms | 3984B | 5172 | ✓ |
| 0313 | 10 | 0ms | 310ms | 2688B | ✓ | 10 | 334ms | 3728B | 4655 | ✓ |
| 0314 | 10 | 106ms | 349ms | 2688B | ✓ | 10 | 619ms | 7696B | 15798 | ✓ |
| 0317 | 10 | 38ms | 277ms | 1376B | ✓ | 10 | 180ms | 33552B | 1000 | ✓ |
| 0318 | — | n/a | n/a | n/a | — | 10 | 306ms | 5392B | 6588 | ✓ |
| 0319 | — | n/a | n/a | n/a | — | 10 | 738ms | 23536B | 27408 | ✓ |
| 0320 | 10 | 102ms | 326ms | 2688B | ✓ | 10 | 257ms | 3056B | 3390 | ✓ |
| 0321 | 10 | 88ms | 314ms | 2688B | ✓ | 10 | 321ms | 3056B | 3390 | ✓ |
| 0322 ⚠️ | 10 | 148ms | 297ms | 2688B | ✓ | 10 | 242ms | 1968B | 655 | ✓ |
| 0323 | 10 | 130ms | 306ms | 2688B | ✓ | 10 | 420ms | 4656B | 6999 | ✓ |
| 0324 | — | n/a | n/a | n/a | — | 10 | 355ms | 4304B | 6847 | ✓ |
| 0325 | — | n/a | n/a | n/a | — | 10 | 626ms | 7088B | 13608 | ✓ |
| 0326 | 10 | 112ms | 310ms | 2688B | ✓ | 10 | 450ms | 5008B | 7369 | ✓ |
| 0329 | 15 | 2.52s | 2.75s | 3008B | ✓ | 10 | 21.78s | 346736B | 967599 | ✓ |
| 0330 | 15 | 0ms | 2.77s | 3008B | ✓ | 10 | 21.74s | 346736B | 967599 | ✓ |
| 0334 | 10 | 126ms | 298ms | 2688B | ✓ | 10 | 2.60s | 36912B | 82551 | ✓ |
| 0335 | 10 | 168ms | 326ms | 2688B | ✓ | 10 | 523ms | 7152B | 9150 | ✓ |
| 0336 | 10 | 27ms | 235ms | 1376B | ✓ | 10 | 168ms | 1808B | 8 | ✓ |
| 0337 | 10 | 22ms | 229ms | 1376B | ✓ | 10 | 167ms | 1808B | 8 | ✓ |
| 0338 | 10 | 0ms | 268ms | 1376B | ✓ | 10 | 203ms | 2064B | 16 | ✓ |
| 0339 | 10 | 59ms | 219ms | 1376B | ✓ | 10 | 192ms | 2064B | 16 | ✓ |
| 0340 | 10 | 19ms | 246ms | 1376B | ✓ | 10 | 179ms | 2512B | 30 | ✓ |
| 0341 | 10 | 0ms | 248ms | 1376B | ✓ | 10 | 162ms | 2512B | 30 | ✓ |
| 0342 | 10 | 6ms | 238ms | 1376B | ✓ | 10 | 156ms | 2512B | 30 | ✓ |
| 0343 | 10 | 11ms | 241ms | 1376B | ✓ | 10 | 157ms | 2512B | 30 | ✓ |
| 0344 | 10 | 13ms | 248ms | 1376B | ✓ | 10 | 176ms | 1584B | 1 | ✓ |
| 0345 | 10 | 137ms | 310ms | 2688B | ✓ | 10 | 244ms | 2896B | 2497 | ✓ |
| 0346 | 10 | 121ms | 337ms | 2688B | ✓ | 10 | 268ms | 2832B | 2177 | ✓ |
| 0347 | 10 | 132ms | 285ms | 2688B | ✓ | 10 | 481ms | 6512B | 11841 | ✗ |
| 0349 | 10 | 38ms | 227ms | 1376B | ✓ | 10 | 162ms | 1584B | 1 | ✓ |
| 0353 | 10 | 98ms | 311ms | 2688B | ✓ | 10 | 256ms | 2896B | 2633 | ✓ |
| 0355 | 10 | 139ms | 301ms | 2688B | ✓ | 10 | 272ms | 4080B | 4285 | ✓ |
| 0356 | 10 | 0ms | 359ms | 2688B | ✓ | 10 | 283ms | 4080B | 4285 | ✓ |
| 0357 | 10 | 113ms | 321ms | 2688B | ✓ | 10 | 286ms | 4080B | 4285 | ✓ |
| 0358 | 10 | 144ms | 313ms | 2688B | ✓ | 10 | 263ms | 3280B | 1629 | ✓ |
| 0359 | 10 | 39ms | 205ms | 1376B | ✓ | 10 | 174ms | 1744B | 6 | ✓ |
| 0360 | 10 | 38ms | 239ms | 1376B | ✓ | 10 | 187ms | 1936B | 12 | ✓ |
| 0361 | 10 | 22ms | 229ms | 1376B | ✓ | 10 | 158ms | 2320B | 24 | ✓ |
| 0363 | 13 | 366ms | 1.24s | 2880B | ✓ | 10 | 473ms | 9360B | 11470 | ✓ |
| 0364 | 16 | 4.69s | 5.02s | 3072B | ✓ | 10 | 652ms | 42704B | 26766 | ✓ |
| 0365 | 13 | 0ms | 1.38s | 2880B | ✓ | 10 | 537ms | 9360B | 11470 | ✓ |
| 0367 | 10 | 47ms | 246ms | 1376B | ✓ | 10 | 164ms | 1648B | 3 | ✓ |
| 0368 | 10 | 112ms | 323ms | 2688B | ✓ | 10 | 350ms | 3728B | 4322 | ✓ |
| 0369 | 10 | 41ms | 350ms | 2688B | ✓ | 10 | 385ms | 3728B | 4322 | ✓ |
| 0370 | 10 | 24ms | 296ms | 2688B | ✓ | 10 | 324ms | 3728B | 4322 | ✓ |
| 0373 | 10 | 137ms | 311ms | 2688B | ✓ | 10 | 5.58s | 82768B | 202151 | ✓ |
| 0374 | 14 | 729ms | 1.90s | 2944B | ✗ | 10 | 6.49s | 93744B | 235263 | ✓ |
| 0375 | 14 | 717ms | 1.95s | 2944B | ✓ | 10 | 5.60s | 85328B | 208433 | ✓ |
| 0381 | 16 | 4.03s | 5.54s | 3072B | ✓ | 10 | 13.80s | 214192B | 564500 | ✓ |
| 0382 | 16 | 2.96s | 5.46s | 3072B | ✓ | 10 | 13.87s | 214192B | 564500 | ✓ |
| 0383 ⚠️ | 16 | 3.29s | 5.58s | 3072B | ✓ | 10 | 14.16s | 211312B | 564513 | ✓ |
| 0385 | 15 | 34ms | 2.36s | 1696B | ✓ | 15 | 1.13s | 641872B | 20000 | ✓ |
| 0387 | 15 | 916ms | 2.33s | 1696B | ✓ | 15 | 1.01s | 641872B | 20000 | ✓ |
| 0388 ⚠️ | 18 | 17.67s | 20.40s | 3200B | ✓ | 15 | 1.13s | 642064B | 20013 | ✓ |
| 0389 ⚠️ | 10 | 60ms | 340ms | 2688B | ✗ | 10 | 577ms | 9136B | 12266 | ✓ |
| 0390 ⚠️ | 18 | 13.82s | 22.43s | 3200B | ✓ | 15 | 17.53s | 683760B | 3400013 | ✓ |
| 0391 ⚠️ | 12 | 284ms | 1.01s | 2816B | ✓ | 10 | 520ms | 9616B | 13828 | ✓ |
| 0392 ⚠️ | 12 | 255ms | 1.10s | 2816B | ✓ | 10 | 598ms | 11888B | 17208 | ✓ |
| 0393 ⚠️ | 13 | 55ms | 1.34s | 2880B | ✗ | 10 | 884ms | 15728B | 24070 | ✓ |
| 0398 ⚠️ | — | n/a | n/a | n/a | — | 10 | 310ms | 3632B | 3989 | ✓ |
| 0399 ⚠️ | — | n/a | n/a | n/a | — | 10 | 346ms | 3984B | 4957 | ✓ |
| 0400 | 10 | 35ms | 231ms | 1376B | ✓ | 10 | 144ms | 1136B | — | ✓ |
| 0401 | 10 | 14ms | 219ms | 1376B | ✓ | 10 | 145ms | 1136B | — | ✓ |
| 0402 | 10 | 114ms | 369ms | 2688B | ✓ | 10 | 181ms | 1776B | 76 | ✓ |
| 0403 | 10 | 148ms | 334ms | 2688B | ✓ | 10 | 177ms | 1776B | 55 | ✓ |
| 0409 | 10 | 134ms | 300ms | 2688B | ✓ | 10 | 199ms | 3280B | 288 | ✓ |
| 0412 | 10 | 106ms | 306ms | 2688B | ✓ | 10 | 242ms | 2192B | 951 | ✓ |
| 0413 | 10 | 18ms | 238ms | 1376B | ✓ | 10 | 159ms | 1136B | — | ✓ |
| 0414 | 10 | 62ms | 295ms | 2688B | ✓ | 10 | 385ms | 4400B | 6585 | ✓ |
| 0415 | 10 | 129ms | 310ms | 2688B | ✓ | 10 | 223ms | 2000B | 368 | ✓ |
| 0416 | 10 | 139ms | 334ms | 2688B | ✓ | 10 | 1.20s | 18640B | 39505 | ✓ |
| 0417 | 10 | 166ms | 300ms | 2688B | ✓ | 10 | 326ms | 4400B | 6585 | ✓ |
| 0418 | 10 | 135ms | 335ms | 2688B | ✓ | 10 | 1.23s | 18640B | 39505 | ✓ |
| 0419 | 10 | 131ms | 313ms | 2688B | ✓ | 10 | 1.32s | 18640B | 39505 | ✓ |
| 0422 | 14 | 1.17s | 1.72s | 2944B | ✓ | 10 | 568ms | 9456B | 17750 | ✓ |
| 0428 | 11 | 201ms | 490ms | 2752B | ✓ | 10 | 1.49s | 21840B | 54990 | ✓ |
| 0429 | 11 | 87ms | 537ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0430 | 11 | 0ms | 517ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0431 | 10 | 204ms | 284ms | 2688B | ✓ | 10 | 378ms | 4528B | 6491 | ✓ |
| 0433 | 10 | 211ms | 286ms | 2688B | ✓ | 10 | 169ms | 2256B | 535 | ✓ |
| 0434 | — | n/a | n/a | n/a | — | 10 | 172ms | 2256B | 937 | ✓ |
| 0435 | 10 | 156ms | 279ms | 2688B | ✓ | 10 | 168ms | 2256B | 17 | ✓ |
| 0436 | 10 | 183ms | 293ms | 2688B | ✓ | 10 | 550ms | 8432B | 12250 | ✓ |
| 0437 | 10 | 112ms | 313ms | 2688B | ✓ | 10 | 533ms | 8432B | 12250 | ✓ |
| 0438 | 10 | 181ms | 320ms | 2688B | ✓ | — | n/a | n/a | — | — |
| 0439 | 15 | 1.89s | 3.00s | 3008B | ✓ | 11 | 6.36s | 128912B | 312200 | ✓ |
| 0441 | 10 | 79ms | 301ms | 2688B | ✓ | 10 | 171ms | 1776B | 77 | ✓ |
| 0442 | 10 | 118ms | 300ms | 2688B | ✓ | 10 | 190ms | 2768B | 1385 | ✓ |
| 0443 | 10 | 80ms | 316ms | 2688B | ✓ | 10 | 292ms | 5936B | 11037 | ✓ |
| 0444 | 10 | 141ms | 337ms | 2688B | ✓ | 10 | 547ms | 7824B | 16877 | ✓ |
| 0448 | 10 | 0ms | 244ms | 1376B | ✓ | 10 | 165ms | 2320B | 24 | ✓ |
| 0449 | 10 | 40ms | 233ms | 1376B | ✓ | 10 | 164ms | 2320B | 24 | ✓ |
| 0450 | 10 | 104ms | 393ms | 2688B | ✓ | 10 | 1.04s | 21296B | 53821 | ✓ |
| 0451 | 10 | 38ms | 225ms | 1376B | ✓ | 10 | 167ms | 1584B | 1 | ✓ |
| 0452 | 10 | 123ms | 325ms | 2688B | ✓ | 10 | 1.12s | 17008B | 35641 | ✓ |
| 0453 | 10 | 117ms | 322ms | 2688B | ✓ | 10 | 1.26s | 18288B | 38506 | ✓ |
| 0454 | 10 | 134ms | 312ms | 2688B | ✓ | 10 | 1.19s | 17232B | 36931 | ✓ |
| 0459 | 10 | 108ms | 306ms | 2688B | ✓ | 10 | 265ms | 2448B | 1662 | ✓ |
| 0460 | 10 | 0ms | 288ms | 2688B | ✓ | 10 | 243ms | 2448B | 1662 | ✓ |
| 0462 | 10 | 176ms | 290ms | 2688B | ✓ | 10 | 333ms | 5936B | 10699 | ✓ |
| 0463 | 10 | 158ms | 310ms | 2688B | ✗ | 10 | 349ms | 5936B | 10699 | ✓ |
| 0467 ⚠️ | 10 | 25ms | 219ms | 1376B | ✓ | 10 | 204ms | 5840B | 1581 | ✓ |
| 0468 ⚠️ | 10 | 35ms | 233ms | 1376B | ✓ | 10 | 181ms | 5840B | 1581 | ✓ |
| 0469 | 10 | 0ms | 291ms | 2688B | ✓ | 10 | 279ms | 3056B | 3293 | ✓ |
| 0470 | 10 | 151ms | 297ms | 2688B | ✓ | 10 | 282ms | 3056B | 3293 | ✓ |
| 0472 | 11 | 19ms | 499ms | 1440B | ✓ | 11 | 217ms | 50000B | 1512 | ✓ |
| 0473 | 10 | 39ms | 226ms | 1376B | ✓ | 10 | 185ms | 1776B | 1 | ✓ |
| 0478 | 10 | 95ms | 282ms | 2688B | ✓ | 10 | 162ms | 1872B | 17 | ✓ |
| 0480 | 10 | 181ms | 293ms | 2688B | ✓ | 10 | 217ms | 1904B | 942 | ✓ |
| 0481 | 10 | 254ms | 307ms | 2688B | ✓ | 10 | 225ms | 2768B | 2392 | ✓ |
| 0485 | 11 | 222ms | 530ms | 2752B | ✓ | 10 | 605ms | 10640B | 14949 | ✓ |
| 0490 ⚠️ | — | n/a | n/a | n/a | — | 10 | 1.01s | 16464B | 31715 | ✓ |
| 0491 | 10 | 138ms | 314ms | 2688B | ✓ | 10 | 250ms | 2672B | 2639 | ✓ |
| 0492 | 10 | 231ms | 281ms | 2688B | ✓ | 10 | 274ms | 2672B | 2639 | ✓ |
| 0500 | 10 | 22ms | 229ms | 1376B | ✓ | 10 | 165ms | 3280B | 54 | ✓ |
| 0501 | 10 | 181ms | 307ms | 2688B | ✓ | 10 | 215ms | 7856B | 2535 | ✓ |
| 0502 | 10 | 14ms | 318ms | 2688B | ✓ | 10 | 216ms | 7856B | 2535 | ✓ |
| 0503 | 12 | 292ms | 1.08s | 2816B | ✓ | 10 | 417ms | 11856B | 15717 | ✓ |
| 0504 | 12 | 334ms | 944ms | 2816B | ✓ | 10 | 434ms | 6704B | 16394 | ✓ |
| 0505 | 13 | 681ms | 1.24s | 2880B | ✓ | 10 | 522ms | 7632B | 19774 | ✓ |
| 0507 | 11 | 355ms | 487ms | 2752B | ✓ | 10 | 285ms | 4368B | 4357 | ✓ |
| 0508 | 14 | 780ms | 1.92s | 2944B | ✓ | 10 | 516ms | 9616B | 13154 | ✓ |
| 0509 | 12 | 369ms | 985ms | 2816B | ✓ | 10 | 835ms | 16016B | 28755 | ✓ |
| 0510 | 12 | 322ms | 1.00s | 2816B | ✓ | 10 | 925ms | 16368B | 29155 | ✓ |

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

- Zinnia / HW prove-time ratio (on 108 both-verified cases): median **1.23×**, range 0.06× – 7.70×
- Zinnia slower: **71 (66%)**; Zinnia faster: **37 (34%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 16 | 15% |
| Zinnia 0.5-1× | 21 | 19% |
| Zinnia 1-1.5× | 43 | 40% |
| Zinnia 1.5-3× | 25 | 23% |
| Zinnia 3-10× | 3 | 3% |
| Zinnia >10× | 0 | 0% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0364 | 5.02s | 652ms | 7.70× |
| 0508 | 1.92s | 516ms | 3.73× |
| 0422 | 1.72s | 568ms | 3.03× |
| 0363 | 1.24s | 473ms | 2.63× |
| 0503 | 1.08s | 417ms | 2.60× |
| 0365 | 1.38s | 537ms | 2.57× |
| 0505 | 1.24s | 522ms | 2.37× |
| 0387 | 2.33s | 1.01s | 2.30× |
| 0472 | 499ms | 217ms | 2.30× |
| 0504 | 944ms | 434ms | 2.17× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 311ms | 5.58s | 0.056× | 17.9× |
| 0334 | 298ms | 2.60s | 0.115× | 8.7× |
| 0329 | 2.75s | 21.78s | 0.126× | 7.9× |
| 0330 | 2.77s | 21.74s | 0.128× | 7.8× |
| 0419 | 313ms | 1.32s | 0.237× | 4.2× |
| 0453 | 322ms | 1.26s | 0.256× | 3.9× |
| 0454 | 312ms | 1.19s | 0.263× | 3.8× |
| 0418 | 335ms | 1.23s | 0.273× | 3.7× |
| 0416 | 334ms | 1.20s | 0.277× | 3.6× |
| 0452 | 325ms | 1.12s | 0.289× | 3.5× |
