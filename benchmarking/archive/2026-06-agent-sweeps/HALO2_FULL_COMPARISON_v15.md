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
| 0291 | 4 | 25ms | 19ms | 992B | ✓ | 7 | 71ms | 1424B | 2 | ✓ |
| 0298 ⚠️ | — | n/a | n/a | n/a | — | 7 | 825ms | 34896B | 9269 | ✓ |
| 0300 | 13 | 1.95s | 469ms | 2880B | — | 7 | 1.71s | 79984B | 22164 | ✓ |
| 0301 | 7 | 45ms | 104ms | 2496B | ✓ | 7 | 149ms | 4464B | 569 | ✓ |
| 0302 | 7 | 42ms | 106ms | 2496B | ✓ | 7 | 137ms | 4464B | 569 | ✓ |
| 0303 | 8 | 59ms | 124ms | 2560B | ✓ | 7 | 310ms | 12048B | 2640 | ✓ |
| 0304 | 9 | 70ms | 145ms | 2624B | ✓ | 7 | 296ms | 11088B | 2464 | ✓ |
| 0309 | 8 | 55ms | 123ms | 2560B | ✓ | 7 | 477ms | 18352B | 4655 | ✓ |
| 0310 | 8 | 56ms | 119ms | 2560B | ✓ | 7 | 462ms | 18352B | 4655 | ✓ |
| 0311 | 8 | 65ms | 120ms | 2560B | ✓ | 7 | 490ms | 19888B | 5172 | ✓ |
| 0312 | 8 | 61ms | 119ms | 2560B | ✓ | 7 | 502ms | 19888B | 5172 | ✓ |
| 0313 | 8 | 4ms | 118ms | 2560B | ✓ | 7 | 464ms | 18352B | 4655 | ✓ |
| 0314 | 10 | 107ms | 180ms | 2688B | ✓ | 7 | 1.25s | 55952B | 15798 | ✓ |
| 0317 | 10 | 0ms | 146ms | 1376B | ✓ | 10 | 157ms | 33552B | 1000 | ✓ |
| 0318 | — | n/a | n/a | n/a | — | 7 | 528ms | 24208B | 6588 | ✓ |
| 0319 | — | n/a | n/a | n/a | — | 9 | 645ms | 33328B | 27408 | ✓ |
| 0320 | 8 | 112ms | 88ms | 2560B | ✓ | 7 | 345ms | 13648B | 3390 | ✓ |
| 0321 | 8 | 55ms | 81ms | 2560B | ✓ | 7 | 373ms | 13648B | 3390 | ✓ |
| 0322 ⚠️ | 6 | 24ms | 57ms | 2432B | ✓ | 7 | 145ms | 4144B | 655 | ✓ |
| 0323 | 9 | 45ms | 94ms | 2624B | ✓ | 7 | 704ms | 27600B | 6999 | ✓ |
| 0324 | — | n/a | n/a | n/a | — | 7 | 666ms | 27376B | 6847 | ✓ |
| 0325 | — | n/a | n/a | n/a | — | 7 | 1.22s | 52592B | 13608 | ✓ |
| 0326 | 9 | 43ms | 95ms | 2624B | ✓ | 7 | 740ms | 29264B | 7369 | ✓ |
| 0329 | 15 | 7.34s | 1.69s | 3008B | ✓ | 7 | 87.76s | 3167760B | 967599 | ✓ |
| 0330 | 15 | 95ms | 1.85s | 3008B | ✓ | 7 | 90.25s | 3167760B | 967599 | ✓ |
| 0334 | 8 | 58ms | 125ms | 2560B | ✓ | 7 | 6.36s | 313072B | 82551 | ✓ |
| 0335 | 9 | 68ms | 147ms | 2624B | ✓ | 7 | 820ms | 36080B | 9150 | ✓ |
| 0336 | 4 | 5ms | 22ms | 992B | ✓ | 7 | 68ms | 1616B | 8 | ✓ |
| 0337 | 4 | 6ms | 22ms | 992B | ✓ | 7 | 74ms | 1616B | 8 | ✓ |
| 0338 | 5 | 2ms | 43ms | 1056B | ✓ | 7 | 75ms | 1872B | 16 | ✓ |
| 0339 | 5 | 0ms | 46ms | 1056B | ✓ | 7 | 71ms | 1872B | 16 | ✓ |
| 0340 | 6 | 18ms | 64ms | 1120B | ✓ | 7 | 77ms | 2320B | 30 | ✓ |
| 0341 | 6 | 5ms | 48ms | 1120B | ✓ | 7 | 68ms | 2320B | 30 | ✓ |
| 0342 | 6 | 1ms | 45ms | 1120B | ✓ | 7 | 66ms | 2320B | 30 | ✓ |
| 0343 | 6 | 7ms | 45ms | 1120B | ✓ | 7 | 75ms | 2320B | 30 | ✓ |
| 0344 | 4 | 1ms | 21ms | 992B | ✓ | 7 | 67ms | 1392B | 1 | ✓ |
| 0345 | 9 | 80ms | 142ms | 2624B | ✓ | 7 | 273ms | 9872B | 2497 | ✓ |
| 0346 | 9 | 77ms | 143ms | 2624B | ✓ | 7 | 246ms | 8560B | 2177 | ✓ |
| 0347 | 9 | 70ms | 142ms | 2624B | ✓ | 7 | 1.05s | 44912B | 11841 | ✗ |
| 0349 | 4 | 3ms | 22ms | 992B | ✓ | 7 | 68ms | 1392B | 1 | ✓ |
| 0353 | 9 | 86ms | 144ms | 2624B | ✓ | 7 | 315ms | 10992B | 2633 | ✓ |
| 0355 | 6 | 38ms | 94ms | 2432B | ✓ | 7 | 395ms | 15888B | 4285 | ✓ |
| 0356 | 6 | 2ms | 94ms | 2432B | ✓ | 7 | 398ms | 15888B | 4285 | ✓ |
| 0357 | 6 | 31ms | 95ms | 2432B | ✓ | 7 | 391ms | 15888B | 4285 | ✓ |
| 0358 | 8 | 60ms | 119ms | 2560B | ✓ | 7 | 190ms | 7024B | 1629 | ✓ |
| 0359 | 4 | 1ms | 23ms | 992B | ✓ | 7 | 65ms | 1552B | 6 | ✓ |
| 0360 | 5 | 3ms | 40ms | 1056B | ✓ | 7 | 68ms | 1744B | 12 | ✓ |
| 0361 | 5 | 10ms | 40ms | 1056B | ✓ | 7 | 65ms | 2128B | 24 | ✓ |
| 0363 | 10 | 153ms | 176ms | 2688B | ✓ | 8 | 579ms | 23088B | 11470 | ✓ |
| 0364 | 12 | 218ms | 383ms | 2816B | ✓ | 10 | 451ms | 42672B | 26781 | ✓ |
| 0365 | 10 | 43ms | 170ms | 2688B | ✓ | 8 | 592ms | 23088B | 11470 | ✓ |
| 0367 | 4 | 6ms | 21ms | 992B | ✓ | 7 | 68ms | 1456B | 3 | ✓ |
| 0368 | 9 | 81ms | 141ms | 2624B | ✓ | 7 | 451ms | 17392B | 4322 | ✓ |
| 0369 | 9 | 65ms | 142ms | 2624B | ✓ | 7 | 443ms | 17392B | 4322 | ✓ |
| 0370 | 9 | 0ms | 142ms | 2624B | ✓ | 7 | 462ms | 17392B | 4322 | ✓ |
| 0373 | 6 | 61ms | 91ms | 2432B | ✓ | 7 | 15.34s | 738544B | 202151 | ✓ |
| 0374 | 14 | 740ms | 1.17s | 2944B | ✗ | 7 | 18.10s | 848272B | 235263 | ✓ |
| 0375 | 14 | 707ms | 1.14s | 2944B | ✓ | 7 | 15.76s | 758768B | 208433 | ✓ |
| 0381 | 16 | 14.46s | 3.67s | 3072B | ✓ | 8 | 21.86s | 891472B | 564500 | ✓ |
| 0382 | 16 | 2.12s | 3.38s | 3072B | ✓ | 8 | 21.56s | 891472B | 564500 | ✓ |
| 0383 ⚠️ | 16 | 1.77s | 3.68s | 3072B | ✓ | 8 | 21.31s | 888752B | 564513 | ✓ |
| 0385 | 15 | 384ms | 1.47s | 1696B | ✓ | 15 | 872ms | 641872B | 20000 | ✓ |
| 0387 | 15 | 301ms | 1.37s | 1696B | ✓ | 15 | 771ms | 641872B | 20000 | ✓ |
| 0388 ⚠️ | 18 | 55.14s | 11.16s | 3200B | ✓ | 15 | 990ms | 642064B | 20013 | ✓ |
| 0389 ⚠️ | 7 | 76ms | 72ms | 2496B | ✗ | 7 | 1.10s | 48752B | 12266 | ✓ |
| 0390 ⚠️ | 18 | 5.37s | 12.24s | 3200B | ✓ | 15 | 24.77s | 683728B | 3400013 | ✓ |
| 0391 ⚠️ | 12 | 190ms | 310ms | 2816B | ✓ | 7 | 1.20s | 54864B | 13828 | ✓ |
| 0392 ⚠️ | 12 | 245ms | 316ms | 2816B | ✓ | 7 | 1.46s | 67664B | 17208 | ✓ |
| 0393 ⚠️ | 12 | 24ms | 373ms | 2816B | ✗ | 8 | 1.16s | 46608B | 24070 | ✓ |
| 0398 ⚠️ | — | n/a | n/a | n/a | — | 7 | 426ms | 16560B | 3989 | ✓ |
| 0399 ⚠️ | — | n/a | n/a | n/a | — | 7 | 509ms | 20016B | 4957 | ✓ |
| 0400 | 4 | 14ms | 21ms | 992B | ✓ | 7 | 62ms | 944B | — | ✓ |
| 0401 | 4 | 0ms | 22ms | 992B | ✓ | 7 | 55ms | 944B | — | ✓ |
| 0402 | 6 | 53ms | 140ms | 2432B | ✓ | 7 | 99ms | 2256B | 211 | ✓ |
| 0403 | 6 | 36ms | 154ms | 2432B | ✓ | 7 | 100ms | 2256B | 190 | ✓ |
| 0409 | 7 | 64ms | 90ms | 2496B | ✓ | 7 | 93ms | 3792B | 288 | ✓ |
| 0412 | 7 | 41ms | 61ms | 2496B | ✓ | 7 | 174ms | 5264B | 951 | ✓ |
| 0413 | 4 | 4ms | 17ms | 992B | ✓ | 7 | 59ms | 944B | — | ✓ |
| 0414 | 7 | 36ms | 71ms | 2496B | ✓ | 7 | 668ms | 26128B | 6585 | ✓ |
| 0415 | 7 | 28ms | 62ms | 2496B | ✓ | 7 | 111ms | 2928B | 368 | ✓ |
| 0416 | 9 | 54ms | 94ms | 2624B | ✓ | 7 | 3.19s | 150576B | 39505 | ✓ |
| 0417 | 7 | 29ms | 63ms | 2496B | ✓ | 7 | 699ms | 26128B | 6585 | ✓ |
| 0418 | 9 | 54ms | 97ms | 2624B | ✓ | 7 | 3.31s | 150576B | 39505 | ✓ |
| 0419 | 9 | 60ms | 94ms | 2624B | ✓ | 7 | 3.12s | 150576B | 39505 | ✓ |
| 0422 | 14 | 464ms | 1.08s | 2944B | ✓ | 7 | 1.20s | 59024B | 17750 | ✓ |
| 0428 | 11 | 140ms | 215ms | 2752B | ✓ | 7 | 3.81s | 188560B | 54990 | ✓ |
| 0429 | 11 | 142ms | 212ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0430 | 11 | 24ms | 217ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0431 | 9 | 80ms | 134ms | 2624B | ✓ | 7 | 642ms | 25904B | 6491 | ✓ |
| 0433 | 6 | 33ms | 91ms | 2432B | ✓ | 7 | 115ms | 3472B | 535 | ✓ |
| 0434 | — | n/a | n/a | n/a | — | 7 | 147ms | 4880B | 937 | ✓ |
| 0435 | 4 | 14ms | 40ms | 2304B | ✓ | 7 | 78ms | 2064B | 17 | ✓ |
| 0436 | 8 | 56ms | 124ms | 2560B | ✓ | 7 | 1.08s | 47664B | 12250 | ✓ |
| 0437 | 8 | 59ms | 122ms | 2560B | ✓ | 7 | 1.05s | 47664B | 12250 | ✓ |
| 0438 | 9 | 77ms | 137ms | 2624B | ✓ | — | n/a | n/a | — | — |
| 0439 | 15 | 1.37s | 1.99s | 3008B | ✓ | 11 | 3.06s | 128848B | 312200 | ✓ |
| 0441 | 5 | 38ms | 72ms | 2368B | ✓ | 7 | 73ms | 1584B | 77 | ✓ |
| 0442 | 9 | 69ms | 140ms | 2624B | ✓ | 7 | 178ms | 6448B | 1385 | ✓ |
| 0443 | 9 | 70ms | 139ms | 2624B | ✓ | 7 | 784ms | 38128B | 11037 | ✓ |
| 0444 | 9 | 72ms | 141ms | 2624B | ✓ | 7 | 1.16s | 54160B | 16877 | ✓ |
| 0448 | 5 | 3ms | 36ms | 1056B | ✓ | 7 | 69ms | 2128B | 24 | ✓ |
| 0449 | 5 | 2ms | 39ms | 1056B | ✓ | 7 | 68ms | 2128B | 24 | ✓ |
| 0450 | 9 | 120ms | 166ms | 2624B | ✓ | 7 | 3.24s | 177744B | 53821 | ✓ |
| 0451 | 4 | 2ms | 28ms | 992B | ✓ | 7 | 64ms | 1392B | 1 | ✓ |
| 0452 | 8 | 116ms | 128ms | 2560B | ✓ | 7 | 2.89s | 135920B | 35641 | ✓ |
| 0453 | 9 | 114ms | 141ms | 2624B | ✓ | 7 | 3.18s | 147088B | 38506 | ✓ |
| 0454 | 8 | 55ms | 123ms | 2560B | ✓ | 7 | 2.82s | 140656B | 36931 | ✓ |
| 0459 | 5 | 45ms | 67ms | 2368B | ✓ | 7 | 244ms | 7888B | 1662 | ✓ |
| 0460 | 5 | 0ms | 65ms | 2368B | ✓ | 7 | 256ms | 7888B | 1662 | ✓ |
| 0462 | 9 | 63ms | 148ms | 2624B | ✓ | 7 | 772ms | 37072B | 10699 | ✓ |
| 0463 | 9 | 77ms | 141ms | 2624B | ✗ | 7 | 755ms | 37072B | 10699 | ✓ |
| 0467 ⚠️ | 5 | 11ms | 37ms | 1056B | ✓ | 7 | 188ms | 10224B | 1581 | ✓ |
| 0468 ⚠️ | 5 | 8ms | 40ms | 1056B | ✓ | 7 | 195ms | 10224B | 1581 | ✓ |
| 0469 | 4 | 17ms | 52ms | 2304B | ✓ | 7 | 370ms | 13776B | 3293 | ✓ |
| 0470 | 4 | 0ms | 55ms | 2304B | ✓ | 7 | 361ms | 13776B | 3293 | ✓ |
| 0472 | 11 | 36ms | 274ms | 1440B | ✓ | 11 | 199ms | 50000B | 1512 | ✓ |
| 0473 | 4 | 12ms | 37ms | 992B | ✓ | 7 | 77ms | 1584B | 1 | ✓ |
| 0478 | 4 | 28ms | 60ms | 2304B | ✓ | 7 | 79ms | 1680B | 17 | ✓ |
| 0480 | 7 | 84ms | 108ms | 2496B | ✓ | 7 | 159ms | 4624B | 942 | ✓ |
| 0481 | 7 | 41ms | 108ms | 2496B | ✓ | 7 | 237ms | 9616B | 2392 | ✓ |
| 0485 | 11 | 112ms | 175ms | 2752B | ✓ | 7 | 1.30s | 59792B | 14949 | ✓ |
| 0490 ⚠️ | — | n/a | n/a | n/a | — | 7 | 2.39s | 118096B | 31715 | ✓ |
| 0491 | 9 | 112ms | 101ms | 2624B | ✓ | 7 | 308ms | 11248B | 2639 | ✓ |
| 0492 | 9 | 1ms | 96ms | 2624B | ✓ | 7 | 320ms | 11248B | 2639 | ✓ |
| 0500 | 6 | 4ms | 32ms | 1120B | ✓ | 7 | 69ms | 3088B | 54 | ✓ |
| 0501 | 10 | 82ms | 129ms | 2688B | ✓ | 8 | 201ms | 10544B | 2535 | ✓ |
| 0502 | 10 | 21ms | 133ms | 2688B | ✓ | 8 | 204ms | 10544B | 2535 | ✓ |
| 0503 | 12 | 175ms | 309ms | 2816B | ✓ | 8 | 575ms | 26640B | 15717 | ✓ |
| 0504 | 12 | 231ms | 360ms | 2816B | ✓ | 7 | 954ms | 46096B | 16394 | ✓ |
| 0505 | 13 | 218ms | 610ms | 2880B | ✓ | 7 | 1.12s | 55216B | 19774 | ✓ |
| 0507 | 11 | 177ms | 234ms | 2752B | ✓ | 7 | 384ms | 15984B | 4357 | ✓ |
| 0508 | 14 | 675ms | 1.18s | 2944B | ✓ | 7 | 929ms | 45392B | 13154 | ✓ |
| 0509 | 12 | 240ms | 359ms | 2816B | ✓ | 7 | 2.12s | 104112B | 28755 | ✓ |
| 0510 | 12 | 279ms | 352ms | 2816B | ✓ | 7 | 2.22s | 105264B | 29155 | ✓ |

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

- Zinnia / HW prove-time ratio (on 108 both-verified cases): median **0.32×**, range 0.01× – 1.77×
- Zinnia slower: **6 (6%)**; Zinnia faster: **102 (94%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 70 | 65% |
| Zinnia 0.5-1× | 32 | 30% |
| Zinnia 1-1.5× | 3 | 3% |
| Zinnia 1.5-3× | 3 | 3% |
| Zinnia 3-10× | 0 | 0% |
| Zinnia >10× | 0 | 0% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0387 | 1.37s | 771ms | 1.77× |
| 0385 | 1.47s | 872ms | 1.69× |
| 0403 | 154ms | 100ms | 1.55× |
| 0402 | 140ms | 99ms | 1.42× |
| 0472 | 274ms | 199ms | 1.37× |
| 0508 | 1.18s | 929ms | 1.27× |
| 0441 | 72ms | 73ms | 0.98× |
| 0409 | 90ms | 93ms | 0.97× |
| 0317 | 146ms | 157ms | 0.93× |
| 0422 | 1.08s | 1.20s | 0.90× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 91ms | 15.34s | 0.006× | 168.8× |
| 0329 | 1.69s | 87.76s | 0.019× | 52.0× |
| 0334 | 125ms | 6.36s | 0.020× | 50.8× |
| 0330 | 1.85s | 90.25s | 0.020× | 48.9× |
| 0418 | 97ms | 3.31s | 0.029× | 34.0× |
| 0416 | 94ms | 3.19s | 0.029× | 34.0× |
| 0419 | 94ms | 3.12s | 0.030× | 33.2× |
| 0454 | 123ms | 2.82s | 0.044× | 23.0× |
| 0452 | 128ms | 2.89s | 0.044× | 22.6× |
| 0453 | 141ms | 3.18s | 0.044× | 22.6× |
