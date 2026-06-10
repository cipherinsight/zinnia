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
| 0291 | 4 | 0ms | 26ms | 1344B | ✓ | 7 | 78ms | 1488B | 2 | ✓ |
| 0300 | 12 | 0ms | 548ms | 4608B | ✓ | 12 | 462ms | 4080B | 22164 | ✓ |
| 0301 | 7 | 3ms | 65ms | 1536B | ✓ | 7 | 175ms | 4464B | 569 | ✓ |
| 0302 | 7 | 0ms | 69ms | 1536B | ✓ | 7 | 179ms | 4464B | 569 | ✓ |
| 0303 | 10 | 0ms | 168ms | 2624B | ✓ | 7 | 472ms | 12048B | 2640 | ✓ |
| 0304 | 11 | 0ms | 374ms | 4768B | ✓ | 10 | 226ms | 3504B | 2464 | ✓ |
| 0309 | 11 | 0ms | 213ms | 2688B | ✓ | 11 | 281ms | 2736B | 4655 | ✓ |
| 0310 | 11 | 16ms | 221ms | 2688B | ✓ | 10 | 276ms | 3728B | 4655 | ✓ |
| 0311 | 11 | 0ms | 239ms | 2688B | ✓ | 11 | 275ms | 2768B | 5172 | ✓ |
| 0312 | 11 | 0ms | 220ms | 2688B | ✓ | 10 | 288ms | 3952B | 5172 | ✓ |
| 0313 | 11 | 0ms | 223ms | 2688B | ✓ | 7 | 730ms | 18352B | 4655 | ✓ |
| 0314 | 12 | 0ms | 391ms | 3328B | ✓ | 12 | 390ms | 3184B | 15798 | ✓ |
| 0317 | 10 | 7ms | 113ms | 1536B | ✓ | 10 | 159ms | 33552B | 1000 | ✓ |
| 0318 | 11 | 123ms | 523ms | 10496B | ✓ | 7 | 812ms | 24208B | 6588 | ✓ |
| 0319 | 13 | 0ms | 11.81s | 131616B | ✓ | 12 | 425ms | 16272B | 27408 | ✓ |
| 0320 | 10 | 0ms | 146ms | 2976B | ✓ | 7 | 561ms | 13648B | 3390 | ✓ |
| 0321 | 10 | 2ms | 145ms | 2976B | ✓ | 10 | 246ms | 3056B | 3390 | ✓ |
| 0322 ⚠️ | 8 | 0ms | 122ms | 3968B | ✓ | 8 | 168ms | 2608B | 655 | ✓ |
| 0323 | 8 | 4ms | 95ms | 2848B | ✓ | 10 | 349ms | 4656B | 6999 | ✓ |
| 0324 | 8 | 0ms | 101ms | 2848B | ✓ | 11 | 302ms | 3088B | 6847 | ✓ |
| 0325 | 9 | 0ms | 107ms | 2560B | ✓ | 11 | 406ms | 4368B | 13608 | ✓ |
| 0326 | 9 | 0ms | 108ms | 2560B | ✓ | 7 | 1.18s | 29264B | 7369 | ✓ |
| 0329 | 11 | 0ms | 457ms | 4992B | ✓ | 13 | 6.07s | 45744B | 967599 | ✓ |
| 0330 | 11 | 137ms | 310ms | 4992B | ✓ | 13 | 5.96s | 45744B | 967599 | ✓ |
| 0334 | 9 | 5ms | 95ms | 2368B | ✓ | 9 | 2.97s | 71568B | 82551 | ✓ |
| 0335 | 11 | 2ms | 249ms | 3968B | ✓ | 11 | 359ms | 5360B | 9150 | ✓ |
| 0336 | 5 | 0ms | 31ms | 1216B | ✓ | 7 | 75ms | 1616B | 8 | ✓ |
| 0337 | 5 | 4ms | 33ms | 1216B | ✓ | 7 | 83ms | 1616B | 8 | ✓ |
| 0338 | 6 | 0ms | 41ms | 1280B | ✓ | 7 | 73ms | 1872B | 16 | ✓ |
| 0339 | 5 | 0ms | 44ms | 1568B | ✓ | 7 | 75ms | 1872B | 16 | ✓ |
| 0340 | 6 | 0ms | 56ms | 1280B | ✓ | 7 | 69ms | 2320B | 30 | ✓ |
| 0341 | 6 | 7ms | 59ms | 1280B | ✓ | 7 | 82ms | 2320B | 30 | ✓ |
| 0342 | 6 | 3ms | 57ms | 1280B | ✓ | 7 | 82ms | 2320B | 30 | ✓ |
| 0343 | 6 | 0ms | 59ms | 1280B | ✓ | 7 | 103ms | 2320B | 30 | ✓ |
| 0344 | 4 | 0ms | 25ms | 1152B | ✓ | 7 | 78ms | 1392B | 1 | ✓ |
| 0345 | 10 | 0ms | 226ms | 2976B | ✓ | 9 | 228ms | 3728B | 2497 | ✓ |
| 0346 | 9 | 19ms | 235ms | 4032B | ✓ | 7 | 336ms | 8560B | 2177 | ✓ |
| 0347 | 10 | 6ms | 214ms | 2976B | ✓ | 7 | 1.71s | 44272B | 11689 | ✓ |
| 0349 | 4 | 1ms | 26ms | 1344B | ✓ | 7 | 87ms | 1392B | 1 | ✓ |
| 0353 | 11 | 24ms | 243ms | 2688B | ✓ | 7 | 469ms | 10992B | 2633 | ✓ |
| 0355 | 9 | 13ms | 149ms | 2560B | ✓ | 7 | 590ms | 15888B | 4285 | ✓ |
| 0356 | 9 | 0ms | 161ms | 2560B | ✓ | 7 | 599ms | 15888B | 4285 | ✓ |
| 0357 | 9 | 0ms | 146ms | 2560B | ✓ | 10 | 258ms | 4080B | 4285 | ✓ |
| 0358 | 11 | 0ms | 395ms | 4768B | ✓ | 7 | 258ms | 7024B | 1629 | ✓ |
| 0359 | 5 | 3ms | 39ms | 1216B | ✓ | 7 | 80ms | 1552B | 6 | ✓ |
| 0360 | 5 | 1ms | 36ms | 1216B | ✓ | 7 | 88ms | 1744B | 12 | ✓ |
| 0361 | 5 | 3ms | 47ms | 1568B | ✓ | 7 | 71ms | 2128B | 24 | ✓ |
| 0363 | 12 | 4ms | 355ms | 3104B | ✓ | 11 | 383ms | 7216B | 11470 | ✓ |
| 0364 | 12 | 0ms | 562ms | 4800B | ✓ | 12 | 450ms | 36240B | 26781 | ✓ |
| 0365 | 11 | 0ms | 386ms | 4320B | ✓ | 12 | 352ms | 6000B | 11470 | ✓ |
| 0367 | 4 | 0ms | 24ms | 1152B | ✓ | 7 | 85ms | 1456B | 3 | ✓ |
| 0368 | 8 | 0ms | 87ms | 1600B | ✓ | 10 | 271ms | 3728B | 4322 | ✓ |
| 0369 | 7 | 0ms | 84ms | 1888B | ✓ | 7 | 675ms | 17392B | 4322 | ✓ |
| 0370 | 7 | 1ms | 80ms | 1888B | ✓ | 10 | 279ms | 3728B | 4322 | ✓ |
| 0373 | 7 | 3ms | 68ms | 1536B | ✓ | 10 | 2.83s | 82672B | 202151 | ✓ |
| 0374 | 11 | 110ms | 1.62s | 35296B | ✓ | 10 | 3.25s | 93584B | 235263 | ✓ |
| 0375 | 13 | 0ms | 1.03s | 8704B | ✓ | 10 | 2.88s | 85200B | 208433 | ✓ |
| 0381 | 12 | 0ms | 5.19s | 63936B | ✓ | 13 | 3.35s | 30672B | 564500 | ✓ |
| 0382 | 12 | 0ms | 5.12s | 63936B | ✓ | 11 | 4.59s | 108656B | 564500 | ✓ |
| 0385 | 15 | 0ms | 968ms | 1856B | ✓ | 15 | 832ms | 641872B | 20000 | ✓ |
| 0387 | 15 | 0ms | 917ms | 1856B | ✓ | 15 | 865ms | 641872B | 20000 | ✓ |
| 0398 ⚠️ | 9 | 0ms | 120ms | 1792B | ✓ | 7 | 691ms | 16560B | 3989 | ✓ |
| 0399 ⚠️ | 8 | 0ms | 119ms | 2496B | ✓ | 10 | 293ms | 3984B | 4957 | ✓ |
| 0400 | 4 | 5ms | 26ms | 736B | ✓ | 7 | 70ms | 944B | — | ✓ |
| 0401 | 4 | 0ms | 35ms | 736B | ✓ | 7 | 71ms | 944B | — | ✓ |
| 0402 | 9 | 1ms | 116ms | 1792B | ✓ | 7 | 127ms | 2256B | 211 | ✓ |
| 0403 | 8 | 3ms | 127ms | 2496B | ✓ | 7 | 125ms | 2256B | 190 | ✓ |
| 0409 | 8 | 22ms | 91ms | 1952B | ✓ | 7 | 112ms | 3792B | 288 | ✓ |
| 0412 | 9 | 0ms | 185ms | 3712B | ✓ | 7 | 231ms | 5264B | 951 | ✓ |
| 0413 | 4 | 5ms | 26ms | 736B | ✓ | 7 | 61ms | 944B | — | ✓ |
| 0414 | 9 | 0ms | 143ms | 2560B | ✓ | 11 | 304ms | 3184B | 6585 | ✓ |
| 0415 | 9 | 0ms | 153ms | 2560B | ✓ | 7 | 140ms | 2928B | 368 | ✓ |
| 0416 | 9 | 0ms | 176ms | 2912B | ✓ | 12 | 669ms | 6448B | 39505 | ✓ |
| 0417 | 9 | 7ms | 141ms | 2560B | ✓ | 11 | 306ms | 3184B | 6585 | ✓ |
| 0418 | 9 | 0ms | 159ms | 2912B | ✓ | 12 | 658ms | 6448B | 39505 | ✓ |
| 0419 | 9 | 10ms | 161ms | 2912B | ✓ | 7 | 3.28s | 150576B | 39505 | ✓ |
| 0422 | 10 | 39ms | 301ms | 5248B | ✓ | 12 | 410ms | 4944B | 17750 | ✓ |
| 0428 | 10 | 0ms | 182ms | 2624B | ✓ | 12 | 754ms | 6896B | 54990 | ✓ |
| 0429 | 9 | 0ms | 196ms | 3488B | ✓ | 9 | 2.04s | 42832B | 54990 | ✓ |
| 0430 | 10 | 0ms | 186ms | 2624B | ✓ | 12 | 796ms | 6896B | 54990 | ✓ |
| 0431 | 10 | 0ms | 176ms | 2624B | ✓ | 11 | 296ms | 3312B | 6491 | ✓ |
| 0433 | 9 | 0ms | 147ms | 2560B | ✓ | 7 | 146ms | 3472B | 535 | ✓ |
| 0434 | 9 | 0ms | 180ms | 3456B | ✓ | 9 | 154ms | 2544B | 937 | ✓ |
| 0435 | 5 | 6ms | 67ms | 1760B | ✓ | 7 | 91ms | 2064B | 17 | ✓ |
| 0436 | 10 | 0ms | 196ms | 3200B | ✓ | 7 | 1.81s | 47664B | 12250 | ✓ |
| 0437 | 10 | 0ms | 204ms | 3200B | ✓ | 12 | 384ms | 4688B | 12250 | ✓ |
| 0438 | 9 | 1ms | 129ms | 2336B | ✓ | 12 | 533ms | 4432B | 26942 | ✓ |
| 0439 | 11 | 0ms | 511ms | 6144B | ✓ | 13 | 2.33s | 81776B | 312200 | ✓ |
| 0441 | 7 | 3ms | 65ms | 1536B | ✓ | 7 | 81ms | 1584B | 77 | ✓ |
| 0442 | 10 | 0ms | 337ms | 4896B | ✓ | 9 | 171ms | 3056B | 1385 | ✓ |
| 0443 | 11 | 75ms | 247ms | 3552B | ✓ | 12 | 304ms | 3248B | 11037 | ✓ |
| 0444 | 11 | 0ms | 410ms | 4864B | ✓ | 7 | 1.97s | 54160B | 16877 | ✓ |
| 0448 | 6 | 6ms | 37ms | 1280B | ✓ | 7 | 83ms | 2128B | 24 | ✓ |
| 0449 | 5 | 0ms | 36ms | 1568B | ✓ | 7 | 83ms | 2128B | 24 | ✓ |
| 0450 | 9 | 0ms | 321ms | 7456B | ✓ | 13 | 549ms | 4944B | 53821 | ✓ |
| 0451 | 4 | 1ms | 23ms | 1344B | ✓ | 7 | 84ms | 1392B | 1 | ✓ |
| 0452 | 10 | 1ms | 165ms | 2624B | ✓ | 12 | 630ms | 5968B | 35641 | ✓ |
| 0453 | 9 | 0ms | 150ms | 3136B | ✓ | 12 | 647ms | 6320B | 38506 | ✓ |
| 0454 | 10 | 0ms | 203ms | 3200B | ✓ | 12 | 645ms | 6320B | 36931 | ✓ |
| 0459 | 6 | 2ms | 54ms | 1824B | ✓ | 7 | 348ms | 7888B | 1662 | ✓ |
| 0460 | 6 | 8ms | 50ms | 1824B | ✓ | 7 | 336ms | 7888B | 1662 | ✓ |
| 0462 | 10 | 25ms | 226ms | 4896B | ✓ | 11 | 301ms | 4240B | 10699 | ✓ |
| 0463 | 10 | 3ms | 219ms | 4928B | ✓ | 10 | 338ms | 5936B | 10699 | ✓ |
| 0469 | 6 | 0ms | 40ms | 1472B | ✓ | 10 | 241ms | 3056B | 3293 | ✓ |
| 0470 | 6 | 1ms | 41ms | 1472B | ✓ | 7 | 572ms | 13776B | 3293 | ✓ |
| 0472 | 11 | 2ms | 123ms | 1600B | ✓ | 11 | 214ms | 50000B | 1512 | ✓ |
| 0473 | 4 | 4ms | 27ms | 1344B | ✓ | 7 | 95ms | 1584B | 1 | ✓ |
| 0478 | 4 | 0ms | 29ms | 1152B | ✓ | 7 | 99ms | 1680B | 17 | ✓ |
| 0480 | 7 | 0ms | 185ms | 3872B | ✓ | 8 | 172ms | 2896B | 942 | ✓ |
| 0481 | 10 | 66ms | 143ms | 2624B | ✓ | 9 | 205ms | 3408B | 2392 | ✓ |
| 0485 | 9 | 0ms | 186ms | 3488B | ✓ | 7 | 2.22s | 59792B | 14949 | ✓ |
| 0490 ⚠️ | 13 | 50ms | 708ms | 4544B | ✓ | 12 | 528ms | 7184B | 31715 | ✓ |
| 0491 | 9 | 3ms | 170ms | 3136B | ✓ | 7 | 476ms | 11248B | 2639 | ✓ |
| 0492 | 9 | 1ms | 168ms | 3136B | ✓ | 10 | 220ms | 2672B | 2639 | ✓ |
| 0500 | 7 | 11ms | 60ms | 1344B | ✓ | 7 | 81ms | 3088B | 54 | ✓ |
| 0501 | 10 | 11ms | 170ms | 2784B | ✓ | 10 | 206ms | 7856B | 2535 | ✓ |
| 0502 | 10 | 2ms | 174ms | 2784B | ✓ | 10 | 205ms | 7856B | 2535 | ✓ |
| 0503 | 10 | 24ms | 336ms | 6048B | ✓ | 12 | 385ms | 8528B | 15717 | ✓ |
| 0504 | 11 | 19ms | 321ms | 4160B | ✓ | 11 | 392ms | 4432B | 16394 | ✓ |
| 0505 | 12 | 0ms | 397ms | 3328B | ✓ | 7 | 1.85s | 55216B | 19774 | ✓ |
| 0507 | 12 | 16ms | 483ms | 4256B | ✓ | 10 | 309ms | 5328B | 8404 | ✓ |
| 0508 | 12 | 47ms | 1.39s | 20928B | ✓ | 12 | 588ms | 8592B | 44885 | ✓ |
| 0509 | 11 | 0ms | 1.72s | 40768B | ✓ | 12 | 727ms | 9904B | 60486 | ✓ |
| 0510 | 11 | 0ms | 2.51s | 40832B | ✓ | 13 | 757ms | 7536B | 60886 | ✓ |

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

- Zinnia / HW prove-time ratio (on 120 both-verified cases): median **0.52×**, range 0.02× – 27.78×
- Zinnia slower: **21 (18%)**; Zinnia faster: **99 (82%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 59 | 49% |
| Zinnia 0.5-1× | 40 | 33% |
| Zinnia 1-1.5× | 12 | 10% |
| Zinnia 1.5-3× | 7 | 6% |
| Zinnia 3-10× | 1 | 1% |
| Zinnia >10× | 1 | 1% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0319 | 11.81s | 425ms | 27.78× |
| 0510 | 2.51s | 757ms | 3.31× |
| 0509 | 1.72s | 727ms | 2.36× |
| 0508 | 1.39s | 588ms | 2.36× |
| 0442 | 337ms | 171ms | 1.97× |
| 0304 | 374ms | 226ms | 1.66× |
| 0507 | 483ms | 309ms | 1.56× |
| 0381 | 5.19s | 3.35s | 1.55× |
| 0358 | 395ms | 258ms | 1.53× |
| 0364 | 562ms | 450ms | 1.25× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 68ms | 2.83s | 0.024× | 41.6× |
| 0334 | 95ms | 2.97s | 0.032× | 31.3× |
| 0419 | 161ms | 3.28s | 0.049× | 20.3× |
| 0330 | 310ms | 5.96s | 0.052× | 19.2× |
| 0470 | 41ms | 572ms | 0.072× | 13.9× |
| 0329 | 457ms | 6.07s | 0.075× | 13.3× |
| 0485 | 186ms | 2.22s | 0.084× | 12.0× |
| 0326 | 108ms | 1.18s | 0.092× | 10.9× |
| 0429 | 196ms | 2.04s | 0.096× | 10.4× |
| 0436 | 196ms | 1.81s | 0.108× | 9.2× |
