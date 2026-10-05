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
| 0291 | 4 | 10ms | 17ms | 992B | ✓ | 7 | 71ms | 1424B | 2 | ✓ |
| 0298 ⚠️ | — | n/a | n/a | n/a | — | 7 | 825ms | 34896B | 9269 | ✓ |
| 0300 | 13 | 875ms | 458ms | 2880B | — | 7 | 1.71s | 79984B | 22164 | ✓ |
| 0301 | 7 | 31ms | 68ms | 2496B | ✓ | 7 | 149ms | 4464B | 569 | ✓ |
| 0302 | 7 | 31ms | 67ms | 2496B | ✓ | 7 | 137ms | 4464B | 569 | ✓ |
| 0303 | 8 | 35ms | 85ms | 2560B | ✓ | 7 | 310ms | 12048B | 2640 | ✓ |
| 0304 | 9 | 54ms | 98ms | 2624B | ✓ | 7 | 296ms | 11088B | 2464 | ✓ |
| 0309 | 8 | 41ms | 82ms | 2560B | ✓ | 7 | 477ms | 18352B | 4655 | ✓ |
| 0310 | 8 | 47ms | 83ms | 2560B | ✓ | 7 | 462ms | 18352B | 4655 | ✓ |
| 0311 | 8 | 59ms | 85ms | 2560B | ✓ | 7 | 490ms | 19888B | 5172 | ✓ |
| 0312 | 8 | 56ms | 83ms | 2560B | ✓ | 7 | 502ms | 19888B | 5172 | ✓ |
| 0313 | 8 | 2ms | 82ms | 2560B | ✓ | 7 | 464ms | 18352B | 4655 | ✓ |
| 0314 | 10 | 98ms | 124ms | 2688B | ✓ | 7 | 1.25s | 55952B | 15798 | ✓ |
| 0317 | 10 | 0ms | 137ms | 1376B | ✓ | 10 | 157ms | 33552B | 1000 | ✓ |
| 0318 | — | n/a | n/a | n/a | — | 7 | 528ms | 24208B | 6588 | ✓ |
| 0319 | — | n/a | n/a | n/a | — | 9 | 645ms | 33328B | 27408 | ✓ |
| 0320 | 8 | 115ms | 129ms | 2560B | ✓ | 7 | 345ms | 13648B | 3390 | ✓ |
| 0321 | 8 | 64ms | 130ms | 2560B | ✓ | 7 | 373ms | 13648B | 3390 | ✓ |
| 0322 ⚠️ | 6 | 37ms | 103ms | 2432B | ✓ | 7 | 145ms | 4144B | 655 | ✓ |
| 0323 | 9 | 62ms | 149ms | 2624B | ✓ | 7 | 704ms | 27600B | 6999 | ✓ |
| 0324 | — | n/a | n/a | n/a | — | 7 | 666ms | 27376B | 6847 | ✓ |
| 0325 | — | n/a | n/a | n/a | — | 7 | 1.22s | 52592B | 13608 | ✓ |
| 0326 | 9 | 61ms | 139ms | 2624B | ✓ | 7 | 740ms | 29264B | 7369 | ✓ |
| 0329 | 15 | 2.27s | 1.98s | 3008B | ✓ | 7 | 87.76s | 3167760B | 967599 | ✓ |
| 0330 | 15 | 131ms | 1.89s | 3008B | ✓ | 7 | 90.25s | 3167760B | 967599 | ✓ |
| 0334 | 8 | 47ms | 89ms | 2560B | ✓ | 7 | 6.36s | 313072B | 82551 | ✓ |
| 0335 | 9 | 55ms | 100ms | 2624B | ✓ | 7 | 820ms | 36080B | 9150 | ✓ |
| 0336 | 4 | 3ms | 18ms | 992B | ✓ | 7 | 68ms | 1616B | 8 | ✓ |
| 0337 | 4 | 4ms | 18ms | 992B | ✓ | 7 | 74ms | 1616B | 8 | ✓ |
| 0338 | 5 | 8ms | 30ms | 1056B | ✓ | 7 | 75ms | 1872B | 16 | ✓ |
| 0339 | 5 | 6ms | 30ms | 1056B | ✓ | 7 | 71ms | 1872B | 16 | ✓ |
| 0340 | 6 | 9ms | 36ms | 1120B | ✓ | 7 | 77ms | 2320B | 30 | ✓ |
| 0341 | 6 | 0ms | 34ms | 1120B | ✓ | 7 | 68ms | 2320B | 30 | ✓ |
| 0342 | 6 | 1ms | 31ms | 1120B | ✓ | 7 | 66ms | 2320B | 30 | ✓ |
| 0343 | 6 | 6ms | 32ms | 1120B | ✓ | 7 | 75ms | 2320B | 30 | ✓ |
| 0344 | 4 | 3ms | 18ms | 992B | ✓ | 7 | 67ms | 1392B | 1 | ✓ |
| 0345 | 9 | 64ms | 105ms | 2624B | ✓ | 7 | 273ms | 9872B | 2497 | ✓ |
| 0346 | 9 | 43ms | 104ms | 2624B | ✓ | 7 | 246ms | 8560B | 2177 | ✓ |
| 0347 | 9 | 52ms | 98ms | 2624B | ✓ | 7 | 1.05s | 44912B | 11841 | ✗ |
| 0349 | 4 | 4ms | 17ms | 992B | ✓ | 7 | 68ms | 1392B | 1 | ✓ |
| 0353 | 9 | 76ms | 99ms | 2624B | ✓ | 7 | 315ms | 10992B | 2633 | ✓ |
| 0355 | 6 | 28ms | 62ms | 2432B | ✓ | 7 | 395ms | 15888B | 4285 | ✓ |
| 0356 | 6 | 1ms | 61ms | 2432B | ✓ | 7 | 398ms | 15888B | 4285 | ✓ |
| 0357 | 6 | 27ms | 60ms | 2432B | ✓ | 7 | 391ms | 15888B | 4285 | ✓ |
| 0358 | 8 | 36ms | 85ms | 2560B | ✓ | 7 | 190ms | 7024B | 1629 | ✓ |
| 0359 | 4 | 3ms | 18ms | 992B | ✓ | 7 | 65ms | 1552B | 6 | ✓ |
| 0360 | 5 | 4ms | 29ms | 1056B | ✓ | 7 | 68ms | 1744B | 12 | ✓ |
| 0361 | 5 | 10ms | 31ms | 1056B | ✓ | 7 | 65ms | 2128B | 24 | ✓ |
| 0363 | 10 | 122ms | 179ms | 2688B | ✓ | 8 | 579ms | 23088B | 11470 | ✓ |
| 0364 | 12 | 212ms | 370ms | 2816B | ✓ | 10 | 451ms | 42672B | 26781 | ✓ |
| 0365 | 10 | 36ms | 177ms | 2688B | ✓ | 8 | 592ms | 23088B | 11470 | ✓ |
| 0367 | 4 | 5ms | 22ms | 992B | ✓ | 7 | 68ms | 1456B | 3 | ✓ |
| 0368 | 9 | 79ms | 140ms | 2624B | ✓ | 7 | 451ms | 17392B | 4322 | ✓ |
| 0369 | 9 | 59ms | 143ms | 2624B | ✓ | 7 | 443ms | 17392B | 4322 | ✓ |
| 0370 | 9 | 2ms | 138ms | 2624B | ✓ | 7 | 462ms | 17392B | 4322 | ✓ |
| 0373 | 6 | 55ms | 97ms | 2432B | ✓ | 7 | 15.34s | 738544B | 202151 | ✓ |
| 0374 | 14 | 645ms | 1.16s | 2944B | ✗ | 7 | 18.10s | 848272B | 235263 | ✓ |
| 0375 | 14 | 561ms | 1.18s | 2944B | ✓ | 7 | 15.76s | 758768B | 208433 | ✓ |
| 0381 | 16 | 4.23s | 3.79s | 3072B | ✓ | 8 | 21.86s | 891472B | 564500 | ✓ |
| 0382 | 16 | 1.63s | 3.84s | 3072B | ✓ | 8 | 21.56s | 891472B | 564500 | ✓ |
| 0383 ⚠️ | 16 | 1.68s | 3.65s | 3072B | ✓ | 8 | 21.31s | 888752B | 564513 | ✓ |
| 0385 | 15 | 408ms | 1.45s | 1696B | ✓ | 15 | 872ms | 641872B | 20000 | ✓ |
| 0387 | 15 | 23ms | 1.43s | 1696B | ✓ | 15 | 771ms | 641872B | 20000 | ✓ |
| 0388 ⚠️ | 18 | 13.57s | 11.27s | 3200B | ✓ | 15 | 990ms | 642064B | 20013 | ✓ |
| 0389 ⚠️ | 7 | 67ms | 110ms | 2496B | ✗ | 7 | 1.10s | 48752B | 12266 | ✓ |
| 0390 ⚠️ | 18 | 7.89s | 11.73s | 3200B | ✓ | 15 | 24.77s | 683728B | 3400013 | ✓ |
| 0391 ⚠️ | 12 | 198ms | 290ms | 2816B | ✓ | 7 | 1.20s | 54864B | 13828 | ✓ |
| 0392 ⚠️ | 12 | 294ms | 345ms | 2816B | ✓ | 7 | 1.46s | 67664B | 17208 | ✓ |
| 0393 ⚠️ | 12 | 62ms | 375ms | 2816B | ✗ | 8 | 1.16s | 46608B | 24070 | ✓ |
| 0398 ⚠️ | — | n/a | n/a | n/a | — | 7 | 426ms | 16560B | 3989 | ✓ |
| 0399 ⚠️ | — | n/a | n/a | n/a | — | 7 | 509ms | 20016B | 4957 | ✓ |
| 0400 | 4 | 5ms | 21ms | 992B | ✓ | 7 | 62ms | 944B | — | ✓ |
| 0401 | 4 | 0ms | 22ms | 992B | ✓ | 7 | 55ms | 944B | — | ✓ |
| 0402 | 6 | 33ms | 150ms | 2432B | ✓ | 7 | 99ms | 2256B | 211 | ✓ |
| 0403 | 6 | 15ms | 171ms | 2432B | ✓ | 7 | 100ms | 2256B | 190 | ✓ |
| 0409 | 7 | 67ms | 107ms | 2496B | ✓ | 7 | 93ms | 3792B | 288 | ✓ |
| 0412 | 7 | 40ms | 104ms | 2496B | ✓ | 7 | 174ms | 5264B | 951 | ✓ |
| 0413 | 4 | 5ms | 20ms | 992B | ✓ | 7 | 59ms | 944B | — | ✓ |
| 0414 | 7 | 48ms | 104ms | 2496B | ✓ | 7 | 668ms | 26128B | 6585 | ✓ |
| 0415 | 7 | 41ms | 105ms | 2496B | ✓ | 7 | 111ms | 2928B | 368 | ✓ |
| 0416 | 9 | 72ms | 148ms | 2624B | ✓ | 7 | 3.19s | 150576B | 39505 | ✓ |
| 0417 | 7 | 42ms | 108ms | 2496B | ✓ | 7 | 699ms | 26128B | 6585 | ✓ |
| 0418 | 9 | 83ms | 145ms | 2624B | ✓ | 7 | 3.31s | 150576B | 39505 | ✓ |
| 0419 | 9 | 73ms | 145ms | 2624B | ✓ | 7 | 3.12s | 150576B | 39505 | ✓ |
| 0422 | 14 | 758ms | 1.01s | 2944B | ✓ | 7 | 1.20s | 59024B | 17750 | ✓ |
| 0428 | 11 | 142ms | 218ms | 2752B | ✓ | 7 | 3.81s | 188560B | 54990 | ✓ |
| 0429 | 11 | 131ms | 222ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0430 | 11 | 18ms | 218ms | 2752B | ✓ | — | n/a | n/a | — | — |
| 0431 | 9 | 69ms | 141ms | 2624B | ✓ | 7 | 642ms | 25904B | 6491 | ✓ |
| 0433 | 6 | 33ms | 93ms | 2432B | ✓ | 7 | 115ms | 3472B | 535 | ✓ |
| 0434 | — | n/a | n/a | n/a | — | 7 | 147ms | 4880B | 937 | ✓ |
| 0435 | 4 | 13ms | 41ms | 2304B | ✓ | 7 | 78ms | 2064B | 17 | ✓ |
| 0436 | 8 | 69ms | 124ms | 2560B | ✓ | 7 | 1.08s | 47664B | 12250 | ✓ |
| 0437 | 8 | 57ms | 130ms | 2560B | ✓ | 7 | 1.05s | 47664B | 12250 | ✓ |
| 0438 | 9 | 74ms | 143ms | 2624B | ✓ | — | n/a | n/a | — | — |
| 0439 | 15 | 1.22s | 1.99s | 3008B | ✓ | 11 | 3.06s | 128848B | 312200 | ✓ |
| 0441 | 5 | 33ms | 70ms | 2368B | ✓ | 7 | 73ms | 1584B | 77 | ✓ |
| 0442 | 9 | 85ms | 140ms | 2624B | ✓ | 7 | 178ms | 6448B | 1385 | ✓ |
| 0443 | 9 | 64ms | 136ms | 2624B | ✓ | 7 | 784ms | 38128B | 11037 | ✓ |
| 0444 | 9 | 72ms | 143ms | 2624B | ✓ | 7 | 1.16s | 54160B | 16877 | ✓ |
| 0448 | 5 | 4ms | 35ms | 1056B | ✓ | 7 | 69ms | 2128B | 24 | ✓ |
| 0449 | 5 | 2ms | 39ms | 1056B | ✓ | 7 | 68ms | 2128B | 24 | ✓ |
| 0450 | 9 | 126ms | 158ms | 2624B | ✓ | 7 | 3.24s | 177744B | 53821 | ✓ |
| 0451 | 4 | 9ms | 24ms | 992B | ✓ | 7 | 64ms | 1392B | 1 | ✓ |
| 0452 | 8 | 102ms | 119ms | 2560B | ✓ | 7 | 2.89s | 135920B | 35641 | ✓ |
| 0453 | 9 | 99ms | 137ms | 2624B | ✓ | 7 | 3.18s | 147088B | 38506 | ✓ |
| 0454 | 8 | 56ms | 120ms | 2560B | ✓ | 7 | 2.82s | 140656B | 36931 | ✓ |
| 0459 | 5 | 46ms | 63ms | 2368B | ✓ | 7 | 244ms | 7888B | 1662 | ✓ |
| 0460 | 5 | 1ms | 66ms | 2368B | ✓ | 7 | 256ms | 7888B | 1662 | ✓ |
| 0462 | 9 | 67ms | 139ms | 2624B | ✓ | 7 | 772ms | 37072B | 10699 | ✓ |
| 0463 | 9 | 57ms | 142ms | 2624B | ✗ | 7 | 755ms | 37072B | 10699 | ✓ |
| 0467 ⚠️ | 5 | 8ms | 37ms | 1056B | ✓ | 7 | 188ms | 10224B | 1581 | ✓ |
| 0468 ⚠️ | 5 | 0ms | 54ms | 1056B | ✓ | 7 | 195ms | 10224B | 1581 | ✓ |
| 0469 | 4 | 11ms | 71ms | 2304B | ✓ | 7 | 370ms | 13776B | 3293 | ✓ |
| 0470 | 4 | 3ms | 64ms | 2304B | ✓ | 7 | 361ms | 13776B | 3293 | ✓ |
| 0472 | 11 | 28ms | 269ms | 1440B | ✓ | 11 | 199ms | 50000B | 1512 | ✓ |
| 0473 | 4 | 6ms | 43ms | 992B | ✓ | 7 | 77ms | 1584B | 1 | ✓ |
| 0478 | 4 | 19ms | 69ms | 2304B | ✓ | 7 | 79ms | 1680B | 17 | ✓ |
| 0480 | 7 | 96ms | 104ms | 2496B | ✓ | 7 | 159ms | 4624B | 942 | ✓ |
| 0481 | 7 | 37ms | 103ms | 2496B | ✓ | 7 | 237ms | 9616B | 2392 | ✓ |
| 0485 | 11 | 144ms | 225ms | 2752B | ✓ | 7 | 1.30s | 59792B | 14949 | ✓ |
| 0490 ⚠️ | — | n/a | n/a | n/a | — | 7 | 2.39s | 118096B | 31715 | ✓ |
| 0491 | 9 | 106ms | 97ms | 2624B | ✓ | 7 | 308ms | 11248B | 2639 | ✓ |
| 0492 | 9 | 0ms | 98ms | 2624B | ✓ | 7 | 320ms | 11248B | 2639 | ✓ |
| 0500 | 6 | 4ms | 31ms | 1120B | ✓ | 7 | 69ms | 3088B | 54 | ✓ |
| 0501 | 10 | 88ms | 124ms | 2688B | ✓ | 8 | 201ms | 10544B | 2535 | ✓ |
| 0502 | 10 | 27ms | 123ms | 2688B | ✓ | 8 | 204ms | 10544B | 2535 | ✓ |
| 0503 | 12 | 143ms | 308ms | 2816B | ✓ | 8 | 575ms | 26640B | 15717 | ✓ |
| 0504 | 12 | 236ms | 357ms | 2816B | ✓ | 7 | 954ms | 46096B | 16394 | ✓ |
| 0505 | 13 | 213ms | 586ms | 2880B | ✓ | 7 | 1.12s | 55216B | 19774 | ✓ |
| 0507 | 11 | 168ms | 224ms | 2752B | ✓ | 7 | 384ms | 15984B | 4357 | ✓ |
| 0508 | 13 | 365ms | 590ms | 2880B | ✓ | 7 | 929ms | 45392B | 13154 | ✓ |
| 0509 | 12 | 259ms | 354ms | 2816B | ✓ | 7 | 2.12s | 104112B | 28755 | ✓ |
| 0510 | 12 | 256ms | 351ms | 2816B | ✓ | 7 | 2.22s | 105264B | 29155 | ✓ |

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

- Zinnia / HW prove-time ratio (on 108 both-verified cases): median **0.31×**, range 0.01× – 1.86×
- Zinnia slower: **6 (6%)**; Zinnia faster: **102 (94%)**

### Distribution

| Bucket | Cases | % |
|---|---:|---:|
| Zinnia ≤0.5× (much faster) | 81 | 75% |
| Zinnia 0.5-1× | 21 | 19% |
| Zinnia 1-1.5× | 2 | 2% |
| Zinnia 1.5-3× | 4 | 4% |
| Zinnia 3-10× | 0 | 0% |
| Zinnia >10× | 0 | 0% |

### Residual hotspots (Zinnia slower by most)

| Case | Zin prove | HW prove | Ratio |
|------|----------:|---------:|------:|
| 0387 | 1.43s | 771ms | 1.86× |
| 0403 | 171ms | 100ms | 1.72× |
| 0385 | 1.45s | 872ms | 1.67× |
| 0402 | 150ms | 99ms | 1.52× |
| 0472 | 269ms | 199ms | 1.35× |
| 0409 | 107ms | 93ms | 1.15× |
| 0441 | 70ms | 73ms | 0.95× |
| 0415 | 105ms | 111ms | 0.94× |
| 0317 | 137ms | 157ms | 0.87× |
| 0478 | 69ms | 79ms | 0.87× |

### Biggest Zinnia wins (Zinnia faster by most)

| Case | Zin prove | HW prove | Ratio | HW/Zin |
|------|----------:|---------:|------:|------:|
| 0373 | 97ms | 15.34s | 0.006× | 157.7× |
| 0334 | 89ms | 6.36s | 0.014× | 71.8× |
| 0330 | 1.89s | 90.25s | 0.021× | 47.6× |
| 0329 | 1.98s | 87.76s | 0.023× | 44.3× |
| 0452 | 119ms | 2.89s | 0.041× | 24.3× |
| 0454 | 120ms | 2.82s | 0.043× | 23.4× |
| 0453 | 137ms | 3.18s | 0.043× | 23.1× |
| 0418 | 145ms | 3.31s | 0.044× | 22.9× |
| 0416 | 148ms | 3.19s | 0.046× | 21.6× |
| 0419 | 145ms | 3.12s | 0.046× | 21.6× |
