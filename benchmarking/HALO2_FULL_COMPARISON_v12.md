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
| 0291 | 4 | 17ms | 17ms | 992B | ✓ | 7 | 98ms | 1424B | 2 | ✓ |
| 0298 ⚠️ | — | n/a | n/a | n/a | — | 7 | 1.13s | 35248B | 9269 | ✓ |
| 0300 | 13 | 691ms | 1.43s | 2880B | — | 7 | 2.43s | 80496B | 22164 | ✓ |
| 0301 | 7 | 69ms | 108ms | 2496B | ✓ | 7 | 187ms | 4496B | 569 | ✓ |
| 0302 | 7 | 60ms | 91ms | 2496B | ✓ | 7 | 188ms | 4496B | 569 | ✓ |
| 0303 | 8 | 76ms | 136ms | 2560B | ✓ | 7 | 369ms | 12144B | 2640 | ✓ |
| 0304 | 9 | 60ms | 235ms | 2624B | ✓ | 7 | 336ms | 11216B | 2464 | ✓ |
| 0309 | 8 | 97ms | 142ms | 2560B | ✓ | 7 | 618ms | 18512B | 4655 | ✓ |
| 0310 | 8 | 63ms | 141ms | 2560B | ✓ | 7 | 621ms | 18512B | 4655 | ✓ |
| 0311 | 8 | 89ms | 144ms | 2560B | ✓ | 7 | 587ms | 20048B | 5172 | ✓ |
| 0312 | 8 | 92ms | 151ms | 2560B | ✓ | 7 | 689ms | 20048B | 5172 | ✓ |
| 0313 | 8 | 23ms | 135ms | 2560B | ✓ | 7 | 589ms | 18512B | 4655 | ✓ |
| 0314 | 10 | 119ms | 332ms | 2688B | ✓ | 7 | 1.62s | 56464B | 15798 | ✓ |
| 0317 | 10 | 24ms | 319ms | 1376B | ✓ | 10 | 169ms | 33552B | 1000 | ✓ |
| 0318 | — | n/a | n/a | n/a | — | 7 | 550ms | 24208B | 6588 | ✓ |
| 0319 | — | n/a | n/a | n/a | — | 9 | 854ms | 33328B | 27408 | ✓ |
| 0320 | 8 | 60ms | 135ms | 2560B | ✓ | 7 | 427ms | 13776B | 3390 | ✓ |
| 0321 | 8 | 71ms | 130ms | 2560B | ✓ | 7 | 461ms | 13776B | 3390 | ✓ |
| 0322 ⚠️ | 6 | 40ms | 67ms | 2432B | ✓ | 7 | 194ms | 4176B | 655 | ✓ |
| 0323 | 9 | 71ms | 198ms | 2624B | ✓ | 7 | 853ms | 27824B | 6999 | ✓ |
| 0324 | — | n/a | n/a | n/a | — | 7 | 897ms | 27600B | 6847 | ✓ |
| 0325 | — | n/a | n/a | n/a | — | 7 | 1.53s | 52944B | 13608 | ✓ |
| 0326 | 9 | 94ms | 200ms | 2624B | ✓ | 7 | 960ms | 29456B | 7369 | ✓ |
| 0329 | 15 | 1.77s | 2.90s | 3008B | ✓ | 7 | 262.82s | 3194864B | 967599 | ✓ |
| 0330 | 15 | 0ms | 2.97s | 3008B | ✓ | 7 | 88.83s | 3194864B | 967599 | ✓ |
| 0334 | 8 | 88ms | 129ms | 2560B | ✓ | 7 | 6.35s | 315856B | 82551 | ✓ |
| 0335 | 9 | 63ms | 202ms | 2624B | ✓ | 7 | 776ms | 36464B | 9150 | ✓ |
| 0336 | 4 | 4ms | 19ms | 992B | ✓ | 7 | 64ms | 1616B | 8 | ✓ |
| 0337 | 4 | 2ms | 17ms | 992B | ✓ | 7 | 68ms | 1616B | 8 | ✓ |
| 0338 | 5 | 5ms | 24ms | 1056B | ✓ | 7 | 67ms | 1872B | 16 | ✓ |
| 0339 | 5 | 1ms | 29ms | 1056B | ✓ | 7 | 69ms | 1872B | 16 | ✓ |
| 0340 | 6 | 9ms | 53ms | 1120B | ✓ | 7 | 68ms | 2320B | 30 | ✓ |
| 0341 | 6 | 0ms | 49ms | 1120B | ✓ | 7 | 70ms | 2320B | 30 | ✓ |
| 0342 | 6 | 2ms | 47ms | 1120B | ✓ | 7 | 65ms | 2320B | 30 | ✓ |
| 0343 | 6 | 11ms | 40ms | 1120B | ✓ | 7 | 66ms | 2320B | 30 | ✓ |
| 0344 | 4 | 1ms | 16ms | 992B | ✓ | 7 | 67ms | 1392B | 1 | ✓ |
| 0345 | 9 | 163ms | 197ms | 2624B | ✓ | 7 | 255ms | 10000B | 2497 | ✓ |
| 0346 | 9 | 39ms | 260ms | 2624B | ✓ | 7 | 241ms | 8656B | 2177 | ✓ |
| 0347 | 9 | 126ms | 200ms | 2624B | ✓ | 7 | 989ms | 45360B | 11841 | ✗ |
| 0349 | 4 | 2ms | 17ms | 992B | ✓ | 7 | 68ms | 1392B | 1 | ✓ |
| 0353 | 9 | 78ms | 206ms | 2624B | ✓ | 7 | 305ms | 11184B | 2633 | ✓ |
| 0355 | 6 | 50ms | 63ms | 2432B | ✓ | 7 | 365ms | 16048B | 4285 | ✓ |
| 0356 | 6 | 22ms | 54ms | 2432B | ✓ | 7 | 376ms | 16048B | 4285 | ✓ |
| 0357 | 6 | 40ms | 53ms | 2432B | ✓ | 7 | 376ms | 16048B | 4285 | ✓ |
| 0358 | 8 | 41ms | 146ms | 2560B | ✓ | 7 | 181ms | 7120B | 1629 | ✓ |
| 0359 | 4 | 3ms | 17ms | 992B | ✓ | 7 | 66ms | 1552B | 6 | ✓ |
| 0360 | 5 | 5ms | 28ms | 1056B | ✓ | 7 | 71ms | 1744B | 12 | ✓ |
| 0361 | 5 | 0ms | 32ms | 1056B | ✓ | 7 | 67ms | 2128B | 24 | ✓ |
| 0363 | 10 | 0ms | 421ms | 2688B | ✓ | 8 | 560ms | 23152B | 11470 | ✓ |
| 0364 | 12 | 343ms | 1.13s | 2816B | ✓ | 10 | 485ms | 42704B | 26766 | ✓ |
| 0365 | 10 | 0ms | 396ms | 2688B | ✓ | 8 | 549ms | 23152B | 11470 | ✓ |
| 0367 | 4 | 2ms | 17ms | 992B | ✓ | 7 | 67ms | 1456B | 3 | ✓ |
| 0368 | 9 | 91ms | 213ms | 2624B | ✓ | 7 | 437ms | 17584B | 4322 | ✓ |
| 0369 | 9 | 96ms | 185ms | 2624B | ✓ | 7 | 444ms | 17584B | 4322 | ✓ |
| 0370 | 9 | 0ms | 211ms | 2624B | ✓ | 7 | 449ms | 17584B | 4322 | ✓ |
| 0373 | 6 | 60ms | 62ms | 2432B | ✓ | 7 | 14.70s | 743248B | 202151 | ✓ |
| 0374 | 14 | 815ms | 1.78s | 2944B | ✗ | 7 | 16.56s | 853936B | 235263 | ✓ |
| 0375 | 14 | 715ms | 1.91s | 2944B | ✓ | 7 | 15.10s | 764112B | 208433 | ✓ |
| 0381 | 16 | 3.60s | 5.79s | 3072B | ✓ | 8 | 19.69s | 894320B | 564500 | ✓ |
| 0382 | 16 | 3.60s | 5.52s | 3072B | ✓ | 8 | 19.66s | 894320B | 564500 | ✓ |
| 0383 ⚠️ | 16 | 3.91s | 5.56s | 3072B | ✓ | 8 | 19.58s | 891664B | 564513 | ✓ |
| 0385 | 15 | 141ms | 2.33s | 1696B | ✓ | 15 | 1.17s | 641872B | 20000 | ✓ |
| 0387 | 15 | 0ms | 2.50s | 1696B | ✓ | 15 | 1.16s | 641872B | 20000 | ✓ |
| 0388 ⚠️ | 18 | 17.20s | 20.86s | 3200B | ✓ | 15 | 1.25s | 642064B | 20013 | ✓ |
| 0389 ⚠️ | 7 | 86ms | 109ms | 2496B | ✗ | 7 | 1.03s | 49264B | 12266 | ✓ |
| 0390 ⚠️ | 18 | 13.12s | 22.62s | 3200B | ✓ | 15 | 18.73s | 683760B | 3400013 | ✓ |
| 0391 ⚠️ | 12 | 276ms | 1.01s | 2816B | ✓ | 7 | 1.12s | 55440B | 13828 | ✓ |
| 0392 ⚠️ | 12 | 342ms | 1.03s | 2816B | ✓ | 7 | 1.39s | 68400B | 17208 | ✓ |
| 0393 ⚠️ | 12 | 0ms | 1.11s | 2816B | ✗ | 8 | 1.08s | 46736B | 24070 | ✓ |
| 0398 ⚠️ | — | n/a | n/a | n/a | — | 7 | 414ms | 16688B | 3989 | ✓ |
| 0399 ⚠️ | — | n/a | n/a | n/a | — | 7 | 487ms | 20176B | 4957 | ✓ |
| 0400 | 4 | 8ms | 22ms | 992B | ✓ | 7 | 56ms | 944B | — | ✓ |
| 0401 | 4 | 2ms | 19ms | 992B | ✓ | 7 | 59ms | 944B | — | ✓ |
| 0402 | 6 | 33ms | 129ms | 2432B | ✓ | 7 | 74ms | 1584B | 76 | ✓ |
| 0403 | 6 | 19ms | 138ms | 2432B | ✓ | 7 | 81ms | 1584B | 55 | ✓ |
| 0409 | 7 | 49ms | 85ms | 2496B | ✓ | 7 | 87ms | 3792B | 288 | ✓ |
| 0412 | 7 | 53ms | 90ms | 2496B | ✓ | 7 | 178ms | 5328B | 951 | ✓ |
| 0413 | 4 | 0ms | 19ms | 992B | ✓ | 7 | 58ms | 944B | — | ✓ |
| 0414 | 7 | 65ms | 113ms | 2496B | ✓ | 7 | 634ms | 26416B | 6585 | ✓ |
| 0415 | 7 | 73ms | 101ms | 2496B | ✓ | 7 | 118ms | 2928B | 368 | ✓ |
| 0416 | 9 | 101ms | 212ms | 2624B | ✓ | 7 | 3.07s | 151920B | 39505 | ✓ |
| 0417 | 7 | 64ms | 111ms | 2496B | ✓ | 7 | 610ms | 26416B | 6585 | ✓ |
| 0418 | 9 | 135ms | 224ms | 2624B | ✓ | 7 | 3.13s | 151920B | 39505 | ✓ |
| 0419 | 9 | 171ms | 221ms | 2624B | ✓ | 7 | 3.18s | 151920B | 39505 | ✓ |
| 0422 | 14 | 875ms | 1.90s | 2944B | ✓ | 7 | 1.10s | 59504B | 17750 | ✓ |
| 0428 | 11 | 254ms | 461ms | 2752B | ✓ | 7 | 3.71s | 190224B | 54990 | ✓ |
| 0429 | 11 | 255ms | 517ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0430 | 11 | 4ms | 537ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0431 | 9 | 135ms | 194ms | 2624B | ✓ | 7 | 608ms | 26064B | 6491 | ✓ |
| 0433 | 6 | 48ms | 68ms | 2432B | ✓ | 7 | 108ms | 3472B | 535 | ✓ |
| 0434 | — | n/a | n/a | n/a | — | 7 | 129ms | 4880B | 937 | ✓ |
| 0435 | 4 | 6ms | 29ms | 2304B | ✓ | 7 | 73ms | 2064B | 17 | ✓ |
| 0436 | 8 | 14ms | 255ms | 2560B | ✓ | 7 | 998ms | 48080B | 12250 | ✓ |
| 0437 | 8 | 151ms | 138ms | 2560B | ✓ | 7 | 1.00s | 48080B | 12250 | ✓ |
| 0438 | 9 | 104ms | 197ms | 2624B | ✓ | — | n/a | n/a | — | — |
| 0439 | 15 | 1.14s | 2.78s | 3008B | ✓ | 11 | 3.21s | 128912B | 312200 | ✓ |
| 0441 | 5 | 42ms | 49ms | 2368B | ✓ | 7 | 74ms | 1584B | 77 | ✓ |
| 0442 | 9 | 89ms | 196ms | 2624B | ✓ | 7 | 153ms | 6448B | 1385 | ✓ |
| 0443 | 9 | 62ms | 209ms | 2624B | ✓ | 7 | 624ms | 38128B | 11037 | ✓ |
| 0444 | 9 | 96ms | 200ms | 2624B | ✓ | 7 | 1.04s | 54608B | 16877 | ✓ |
| 0448 | 5 | 2ms | 30ms | 1056B | ✓ | 7 | 69ms | 2128B | 24 | ✓ |
| 0449 | 5 | 3ms | 27ms | 1056B | ✓ | 7 | 67ms | 2128B | 24 | ✓ |
| 0450 | 9 | 81ms | 243ms | 2624B | ✓ | 7 | 2.66s | 177744B | 53821 | ✓ |
| 0451 | 4 | 1ms | 17ms | 992B | ✓ | 7 | 68ms | 1392B | 1 | ✓ |
| 0452 | 8 | 69ms | 135ms | 2560B | ✓ | 7 | 2.86s | 137136B | 35641 | ✓ |
| 0453 | 9 | 102ms | 209ms | 2624B | ✓ | 7 | 2.93s | 147952B | 38506 | ✓ |
| 0454 | 8 | 57ms | 136ms | 2560B | ✓ | 7 | 2.99s | 141552B | 36931 | ✓ |
| 0459 | 5 | 30ms | 61ms | 2368B | ✓ | 7 | 240ms | 7952B | 1662 | ✓ |
| 0460 | 5 | 1ms | 55ms | 2368B | ✓ | 7 | 237ms | 7952B | 1662 | ✓ |
| 0462 | 9 | 93ms | 208ms | 2624B | ✓ | 7 | 613ms | 37072B | 10699 | ✓ |
| 0463 | 9 | 110ms | 220ms | 2624B | ✗ | 7 | 619ms | 37072B | 10699 | ✓ |
| 0467 ⚠️ | 5 | 4ms | 28ms | 1056B | ✓ | 7 | 173ms | 10224B | 1581 | ✓ |
| 0468 ⚠️ | 5 | 0ms | 28ms | 1056B | ✓ | 7 | 174ms | 10224B | 1581 | ✓ |
| 0469 | 4 | 13ms | 25ms | 2304B | ✓ | 7 | 369ms | 13904B | 3293 | ✓ |
| 0470 | 4 | 0ms | 26ms | 2304B | ✓ | 7 | 367ms | 13904B | 3293 | ✓ |
| 0472 | 11 | 0ms | 546ms | 1440B | ✓ | 11 | 239ms | 50000B | 1512 | ✓ |
| 0473 | 4 | 1ms | 19ms | 992B | ✓ | 7 | 75ms | 1584B | 1 | ✓ |
| 0478 | 4 | 15ms | 34ms | 2304B | ✓ | 7 | 75ms | 1680B | 17 | ✓ |
| 0480 | 7 | 96ms | 96ms | 2496B | ✓ | 7 | 159ms | 4688B | 942 | ✓ |
| 0481 | 7 | 15ms | 109ms | 2496B | ✓ | 7 | 216ms | 9616B | 2392 | ✓ |
| 0485 | 11 | 222ms | 528ms | 2752B | ✓ | 7 | 1.26s | 60304B | 14949 | ✓ |
| 0490 ⚠️ | — | n/a | n/a | n/a | — | 7 | 2.33s | 119184B | 31715 | ✓ |
| 0491 | 9 | 83ms | 204ms | 2624B | ✓ | 7 | 320ms | 11312B | 2639 | ✓ |
| 0492 | 9 | 16ms | 182ms | 2624B | ✓ | 7 | 314ms | 11312B | 2639 | ✓ |
| 0500 | 6 | 14ms | 39ms | 1120B | ✓ | 7 | 65ms | 3088B | 54 | ✓ |
| 0501 | 10 | 96ms | 336ms | 2688B | ✓ | 8 | 175ms | 10544B | 2535 | ✓ |
| 0502 | 10 | 15ms | 326ms | 2688B | ✓ | 8 | 169ms | 10544B | 2535 | ✓ |
| 0503 | 12 | 183ms | 1.11s | 2816B | ✓ | 8 | 504ms | 26832B | 15717 | ✓ |
| 0504 | 12 | 263ms | 1.09s | 2816B | ✓ | 7 | 853ms | 46640B | 16394 | ✓ |
| 0505 | 13 | 435ms | 1.30s | 2880B | ✓ | 7 | 1.01s | 55856B | 19774 | ✓ |
| 0507 | 11 | 188ms | 524ms | 2752B | ✓ | 7 | 359ms | 16208B | 4357 | ✓ |
| 0508 | 14 | 632ms | 1.98s | 2944B | ✓ | 7 | 846ms | 46000B | 13154 | ✓ |
| 0509 | 12 | 377ms | 1.09s | 2816B | ✓ | 7 | 2.05s | 105168B | 28755 | ✓ |
| 0510 | 12 | 449ms | 1.01s | 2816B | ✓ | 7 | 2.06s | 106096B | 29155 | ✓ |

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

- Zinnia / HW prove-time ratio (on 108 both-verified cases): median **0.38×**, range 0.00× – 2.34×
- Zinnia slower: **17 (16%)**; Zinnia faster: **91 (84%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 68 | 63% |
| Zinnia 0.5-1× | 23 | 21% |
| Zinnia 1-1.5× | 5 | 5% |
| Zinnia 1.5-3× | 12 | 11% |
| Zinnia 3-10× | 0 | 0% |
| Zinnia >10× | 0 | 0% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0364 | 1.13s | 485ms | 2.34× |
| 0508 | 1.98s | 846ms | 2.34× |
| 0472 | 546ms | 239ms | 2.29× |
| 0503 | 1.11s | 504ms | 2.20× |
| 0387 | 2.50s | 1.16s | 2.14× |
| 0385 | 2.33s | 1.17s | 2.00× |
| 0502 | 326ms | 169ms | 1.93× |
| 0501 | 336ms | 175ms | 1.92× |
| 0317 | 319ms | 169ms | 1.89× |
| 0402 | 129ms | 74ms | 1.76× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 62ms | 14.70s | 0.004× | 236.8× |
| 0329 | 2.90s | 262.82s | 0.011× | 90.6× |
| 0334 | 129ms | 6.35s | 0.020× | 49.3× |
| 0330 | 2.97s | 88.83s | 0.033× | 29.9× |
| 0454 | 136ms | 2.99s | 0.046× | 21.9× |
| 0452 | 135ms | 2.86s | 0.047× | 21.2× |
| 0469 | 25ms | 369ms | 0.068× | 14.7× |
| 0416 | 212ms | 3.07s | 0.069× | 14.5× |
| 0419 | 221ms | 3.18s | 0.070× | 14.4× |
| 0470 | 26ms | 367ms | 0.070× | 14.2× |
