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
| 0291 | 4 | 0ms | 25ms | 1344B | ✓ | 7 | 87ms | 1488B | 2 | ✓ |
| 0300 | 12 | 0ms | 446ms | 3328B | ✓ | 12 | 485ms | 4080B | 22164 | ✓ |
| 0301 | 6 | 0ms | 90ms | 2176B | ✓ | 7 | 185ms | 4464B | 569 | ✓ |
| 0302 | 7 | 0ms | 95ms | 1536B | ✓ | 8 | 160ms | 3280B | 569 | ✓ |
| 0303 | 9 | 0ms | 177ms | 2880B | ✓ | 10 | 221ms | 3504B | 2640 | ✓ |
| 0304 | 9 | 0ms | 166ms | 2912B | ✓ | 10 | 220ms | 3504B | 2464 | ✓ |
| 0309 | 10 | 17ms | 205ms | 3200B | ✓ | 10 | 285ms | 3728B | 4655 | ✓ |
| 0310 | 10 | 12ms | 200ms | 3200B | ✓ | 10 | 287ms | 3728B | 4655 | ✓ |
| 0311 | 10 | 10ms | 207ms | 3200B | ✓ | 7 | 808ms | 19888B | 5172 | ✓ |
| 0312 | 10 | 0ms | 213ms | 3200B | ✓ | 10 | 300ms | 3952B | 5172 | ✓ |
| 0313 | 10 | 0ms | 212ms | 3200B | ✓ | 10 | 286ms | 3728B | 4655 | ✓ |
| 0314 | 11 | 13ms | 307ms | 3616B | ✓ | 12 | 388ms | 3184B | 15798 | ✓ |
| 0317 | 10 | 3ms | 151ms | 1536B | ✓ | 10 | 163ms | 33552B | 1000 | ✓ |
| 0318 | 12 | 0ms | 530ms | 4480B | ✓ | 10 | 269ms | 5392B | 6588 | ✓ |
| 0319 | 11 | 18ms | 476ms | 6112B | ✓ | 12 | 415ms | 16272B | 27408 | ✓ |
| 0320 | 10 | 0ms | 173ms | 2624B | ✓ | 10 | 238ms | 3056B | 3390 | ✓ |
| 0321 | 10 | 0ms | 190ms | 2624B | ✓ | 10 | 240ms | 3056B | 3390 | ✓ |
| 0322 ⚠️ | 9 | 0ms | 155ms | 2560B | ✓ | 7 | 193ms | 4144B | 655 | ✓ |
| 0323 | 8 | 0ms | 151ms | 2816B | ✓ | 11 | 335ms | 3440B | 6999 | ✓ |
| 0324 | 8 | 0ms | 134ms | 2816B | ✓ | 11 | 306ms | 3088B | 6847 | ✓ |
| 0325 | 9 | 0ms | 159ms | 2560B | ✓ | 7 | 2.08s | 52592B | 13608 | ✓ |
| 0326 | 8 | 7ms | 142ms | 3072B | ✓ | 11 | 342ms | 3440B | 7369 | ✓ |
| 0329 | 11 | 33ms | 394ms | 4768B | ✓ | 13 | 5.96s | 45744B | 967599 | ✓ |
| 0330 | 11 | 9ms | 415ms | 4768B | ✓ | 13 | 5.96s | 45744B | 967599 | ✓ |
| 0334 | 8 | 0ms | 160ms | 3360B | ✓ | 9 | 3.41s | 71568B | 82551 | ✓ |
| 0335 | 10 | 0ms | 338ms | 5504B | ✓ | 11 | 352ms | 5360B | 9150 | ✓ |
| 0336 | 5 | 0ms | 69ms | 1216B | ✓ | 7 | 85ms | 1616B | 8 | ✓ |
| 0337 | 5 | 0ms | 71ms | 1216B | ✓ | 7 | 86ms | 1616B | 8 | ✓ |
| 0338 | 5 | 0ms | 77ms | 1568B | ✓ | 7 | 87ms | 1872B | 16 | ✓ |
| 0339 | 6 | 3ms | 81ms | 1280B | ✓ | 7 | 76ms | 1872B | 16 | ✓ |
| 0340 | 6 | 10ms | 73ms | 1280B | ✓ | 7 | 90ms | 2320B | 30 | ✓ |
| 0341 | 6 | 16ms | 72ms | 1280B | ✓ | 7 | 66ms | 2320B | 30 | ✓ |
| 0342 | 6 | 19ms | 84ms | 1280B | ✓ | 7 | 74ms | 2320B | 30 | ✓ |
| 0343 | 6 | 0ms | 80ms | 1280B | ✓ | 7 | 85ms | 2320B | 30 | ✓ |
| 0344 | 4 | 0ms | 43ms | 1152B | ✓ | 7 | 84ms | 1392B | 1 | ✓ |
| 0345 | 8 | 10ms | 175ms | 3968B | ✓ | 9 | 226ms | 3728B | 2497 | ✓ |
| 0346 | 9 | 0ms | 175ms | 2912B | ✓ | 8 | 264ms | 5104B | 2177 | ✓ |
| 0347 | 8 | 0ms | 177ms | 3936B | ✓ | 12 | 363ms | 2800B | 11689 | ✓ |
| 0349 | 4 | 0ms | 28ms | 1344B | ✓ | 7 | 79ms | 1392B | 1 | ✓ |
| 0353 | 10 | 0ms | 209ms | 3200B | ✓ | 7 | 467ms | 10992B | 2633 | ✓ |
| 0355 | 8 | 29ms | 122ms | 2496B | ✓ | 7 | 609ms | 15888B | 4285 | ✓ |
| 0356 | 9 | 0ms | 158ms | 2304B | ✓ | 10 | 248ms | 4080B | 4285 | ✓ |
| 0357 | 9 | 0ms | 131ms | 1792B | ✓ | 10 | 257ms | 4080B | 4285 | ✓ |
| 0358 | 8 | 0ms | 172ms | 3616B | ✓ | 7 | 254ms | 7024B | 1629 | ✓ |
| 0359 | 5 | 0ms | 67ms | 1216B | ✓ | 7 | 86ms | 1552B | 6 | ✓ |
| 0360 | 5 | 0ms | 66ms | 1216B | ✓ | 7 | 94ms | 1744B | 12 | ✓ |
| 0361 | 5 | 0ms | 85ms | 1568B | ✓ | 7 | 86ms | 2128B | 24 | ✓ |
| 0363 | 10 | 3ms | 281ms | 4320B | ✓ | 11 | 383ms | 7216B | 11470 | ✓ |
| 0364 | 11 | 4ms | 468ms | 5536B | ✓ | 12 | 479ms | 36240B | 26781 | ✓ |
| 0365 | 10 | 7ms | 281ms | 4320B | ✓ | 11 | 370ms | 7216B | 11470 | ✓ |
| 0367 | 4 | 0ms | 39ms | 1152B | ✓ | 7 | 82ms | 1456B | 3 | ✓ |
| 0368 | 6 | 9ms | 96ms | 2176B | ✓ | 7 | 723ms | 17392B | 4322 | ✓ |
| 0369 | 7 | 6ms | 86ms | 1888B | ✓ | 7 | 721ms | 17392B | 4322 | ✓ |
| 0370 | 7 | 0ms | 100ms | 1888B | ✓ | 7 | 725ms | 17392B | 4322 | ✓ |
| 0373 | 6 | 10ms | 84ms | 1824B | ✓ | 10 | 3.53s | 82672B | 202151 | ✓ |
| 0374 | 13 | 0ms | 988ms | 6400B | ✓ | 10 | 3.14s | 93584B | 235263 | ✓ |
| 0375 | 7 | 0ms | 106ms | 1344B | ✓ | 10 | 2.84s | 85200B | 208433 | ✓ |
| 0381 | 13 | 0ms | 2.25s | 18432B | ✓ | 13 | 3.31s | 30672B | 564500 | ✓ |
| 0382 | 13 | 0ms | 2.24s | 18432B | ✓ | 11 | 4.61s | 108656B | 564500 | ✓ |
| 0385 | 15 | 23ms | 935ms | 1856B | ✓ | 15 | 883ms | 641872B | 20000 | ✓ |
| 0387 | 15 | 0ms | 948ms | 1856B | ✓ | 15 | 821ms | 641872B | 20000 | ✓ |
| 0398 ⚠️ | 9 | 0ms | 125ms | 1792B | ✓ | 7 | 673ms | 16560B | 3989 | ✓ |
| 0399 ⚠️ | 8 | 0ms | 117ms | 2496B | ✓ | 10 | 294ms | 3984B | 4957 | ✓ |
| 0400 | 4 | 1ms | 34ms | 736B | ✓ | 7 | 70ms | 944B | — | ✓ |
| 0401 | 4 | 0ms | 42ms | 736B | ✓ | 7 | 60ms | 944B | — | ✓ |
| 0402 | 9 | 0ms | 140ms | 1792B | ✓ | 7 | 128ms | 2256B | 211 | ✓ |
| 0403 | 8 | 14ms | 130ms | 2496B | ✓ | 7 | 125ms | 2256B | 190 | ✓ |
| 0409 | 8 | 25ms | 104ms | 1952B | ✓ | 7 | 119ms | 3792B | 288 | ✓ |
| 0412 | 9 | 0ms | 177ms | 3072B | ✓ | 8 | 185ms | 3376B | 951 | ✓ |
| 0413 | 4 | 6ms | 40ms | 736B | ✓ | 7 | 63ms | 944B | — | ✓ |
| 0414 | 8 | 7ms | 145ms | 3072B | ✓ | 7 | 1.04s | 26128B | 6585 | ✓ |
| 0415 | 9 | 0ms | 144ms | 2336B | ✓ | 7 | 134ms | 2928B | 368 | ✓ |
| 0416 | 8 | 0ms | 176ms | 3392B | ✓ | 13 | 618ms | 4656B | 39505 | ✓ |
| 0417 | 8 | 0ms | 147ms | 3072B | ✓ | 10 | 313ms | 4400B | 6585 | ✓ |
| 0418 | 8 | 0ms | 156ms | 3424B | ✓ | 8 | 3.07s | 70480B | 39505 | ✓ |
| 0419 | 8 | 0ms | 152ms | 3040B | ✓ | 9 | 1.73s | 35376B | 39505 | ✓ |
| 0422 | 11 | 6ms | 328ms | 3616B | ✓ | 12 | 422ms | 4944B | 17750 | ✓ |
| 0428 | 9 | 0ms | 175ms | 2560B | ✓ | 12 | 780ms | 6896B | 54990 | ✓ |
| 0429 | 9 | 0ms | 167ms | 2560B | ✓ | 9 | 2.06s | 42832B | 54990 | ✓ |
| 0430 | 9 | 0ms | 156ms | 2560B | ✓ | 8 | 3.42s | 87664B | 54990 | ✓ |
| 0431 | 9 | 0ms | 175ms | 2912B | ✓ | 7 | 990ms | 25904B | 6491 | ✓ |
| 0433 | 8 | 0ms | 126ms | 2496B | ✓ | 7 | 144ms | 3472B | 535 | ✓ |
| 0434 | 8 | 0ms | 150ms | 3072B | ✓ | 8 | 164ms | 3184B | 937 | ✓ |
| 0435 | 5 | 15ms | 80ms | 1760B | ✓ | 7 | 98ms | 2064B | 17 | ✓ |
| 0436 | 10 | 0ms | 206ms | 3168B | ✓ | 11 | 401ms | 5904B | 12250 | ✓ |
| 0437 | 10 | 1ms | 214ms | 3168B | ✓ | 11 | 403ms | 5904B | 12250 | ✓ |
| 0438 | 8 | 9ms | 130ms | 2848B | ✓ | 12 | 531ms | 4432B | 26942 | ✓ |
| 0439 | 12 | 0ms | 530ms | 4000B | ✓ | 13 | 2.37s | 81776B | 312200 | ✓ |
| 0441 | 6 | 0ms | 92ms | 1824B | ✓ | 7 | 91ms | 1584B | 77 | ✓ |
| 0442 | 10 | 0ms | 247ms | 3488B | ✓ | 8 | 186ms | 4048B | 1385 | ✓ |
| 0443 | 10 | 0ms | 242ms | 3488B | ✓ | 12 | 300ms | 3248B | 11037 | ✓ |
| 0444 | 10 | 0ms | 300ms | 4704B | ✓ | 11 | 384ms | 5104B | 16877 | ✓ |
| 0448 | 6 | 0ms | 85ms | 1280B | ✓ | 7 | 84ms | 2128B | 24 | ✓ |
| 0449 | 5 | 0ms | 84ms | 1568B | ✓ | 7 | 79ms | 2128B | 24 | ✓ |
| 0450 | 10 | 4ms | 250ms | 3520B | ✓ | 13 | 563ms | 4944B | 53821 | ✓ |
| 0451 | 12 | 19ms | 212ms | 1856B | ✓ | 7 | 80ms | 1392B | 1 | ✓ |
| 0452 | 9 | 14ms | 169ms | 3136B | ✓ | 12 | 636ms | 5968B | 35641 | ✓ |
| 0453 | 9 | 0ms | 171ms | 2912B | ✓ | 9 | 1.66s | 34160B | 38506 | ✓ |
| 0454 | 10 | 3ms | 196ms | 2976B | ✓ | 12 | 645ms | 6320B | 36931 | ✓ |
| 0459 | 7 | 0ms | 104ms | 1536B | ✓ | 7 | 360ms | 7888B | 1662 | ✓ |
| 0460 | 6 | 0ms | 94ms | 1824B | ✓ | 9 | 207ms | 2960B | 1662 | ✓ |
| 0462 | 11 | 2ms | 225ms | 2496B | ✓ | 12 | 297ms | 3248B | 10699 | ✓ |
| 0463 | 10 | 6ms | 244ms | 3488B | ✓ | 12 | 303ms | 3248B | 10699 | ✓ |
| 0469 | 5 | 0ms | 93ms | 2112B | ✓ | 7 | 575ms | 13776B | 3293 | ✓ |
| 0470 | 6 | 0ms | 84ms | 1472B | ✓ | 7 | 584ms | 13776B | 3293 | ✓ |
| 0472 | 11 | 18ms | 181ms | 1600B | ✓ | 11 | 177ms | 50000B | 1512 | ✓ |
| 0473 | 4 | 8ms | 35ms | 1344B | ✓ | 7 | 98ms | 1584B | 1 | ✓ |
| 0478 | 4 | 2ms | 24ms | 1152B | ✓ | 7 | 93ms | 1680B | 17 | ✓ |
| 0480 | 7 | 0ms | 99ms | 2592B | ✓ | 8 | 176ms | 2896B | 942 | ✓ |
| 0481 | 9 | 5ms | 162ms | 2912B | ✓ | 7 | 373ms | 9616B | 2392 | ✓ |
| 0485 | 9 | 12ms | 211ms | 3488B | ✓ | 11 | 464ms | 7920B | 14949 | ✓ |
| 0490 ⚠️ | 12 | 9ms | 476ms | 3904B | ✓ | 7 | 2.44s | 118096B | 31715 | ✓ |
| 0491 | 10 | 0ms | 173ms | 2400B | ✓ | 10 | 229ms | 2672B | 2639 | ✓ |
| 0492 | 9 | 0ms | 164ms | 2912B | ✓ | 10 | 227ms | 2672B | 2639 | ✓ |
| 0500 | 6 | 1ms | 83ms | 1632B | ✓ | 7 | 75ms | 3088B | 54 | ✓ |
| 0501 | 10 | 0ms | 212ms | 2784B | ✓ | 10 | 190ms | 7856B | 2535 | ✓ |
| 0502 | 10 | 2ms | 199ms | 2784B | ✓ | 9 | 206ms | 8848B | 2535 | ✓ |
| 0503 | 10 | 0ms | 320ms | 5248B | ✓ | 11 | 379ms | 9616B | 15717 | ✓ |
| 0504 | 10 | 0ms | 321ms | 5472B | ✓ | 11 | 386ms | 4432B | 16394 | ✓ |
| 0505 | 10 | 0ms | 389ms | 6400B | ✓ | 8 | 1.04s | 26064B | 19774 | ✓ |
| 0507 | 10 | 0ms | 309ms | 4928B | ✓ | 7 | 915ms | 25648B | 8404 | ✓ |
| 0508 | 11 | 0ms | 500ms | 6112B | ✓ | 12 | 585ms | 8592B | 44885 | ✓ |
| 0509 | 11 | 0ms | 553ms | 7264B | ✓ | 13 | 754ms | 7536B | 60486 | ✓ |
| 0510 | 12 | 0ms | 565ms | 4800B | ✓ | 13 | 704ms | 7536B | 60886 | ✓ |

## Summary

- Cases attempted: **124**
- Both sides verified (and apples-to-apples): **120**
- Zinnia-only verified: **0**
- HW-only verified: **0**
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

- Zinnia / HW prove-time ratio (on 120 both-verified cases): median **0.69×**, range 0.02× – 2.65×
- Zinnia slower: **18 (15%)**; Zinnia faster: **102 (85%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 48 | 40% |
| Zinnia 0.5-1× | 54 | 45% |
| Zinnia 1-1.5× | 16 | 13% |
| Zinnia 1.5-3× | 2 | 2% |
| Zinnia 3-10× | 0 | 0% |
| Zinnia >10× | 0 | 0% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0451 | 212ms | 80ms | 2.65× |
| 0318 | 530ms | 269ms | 1.97× |
| 0442 | 247ms | 186ms | 1.32× |
| 0387 | 948ms | 821ms | 1.16× |
| 0319 | 476ms | 415ms | 1.15× |
| 0342 | 84ms | 74ms | 1.14× |
| 0501 | 212ms | 190ms | 1.12× |
| 0500 | 83ms | 75ms | 1.11× |
| 0341 | 72ms | 66ms | 1.10× |
| 0402 | 140ms | 128ms | 1.09× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 84ms | 3.53s | 0.024× | 41.8× |
| 0375 | 106ms | 2.84s | 0.037× | 26.8× |
| 0430 | 156ms | 3.42s | 0.046× | 22.0× |
| 0334 | 160ms | 3.41s | 0.047× | 21.3× |
| 0418 | 156ms | 3.07s | 0.051× | 19.7× |
| 0329 | 394ms | 5.96s | 0.066× | 15.1× |
| 0330 | 415ms | 5.96s | 0.070× | 14.4× |
| 0325 | 159ms | 2.08s | 0.076× | 13.1× |
| 0429 | 167ms | 2.06s | 0.081× | 12.4× |
| 0419 | 152ms | 1.73s | 0.088× | 11.4× |
