# Zinnia halo2 vs handwritten halo2-graph baseline

Side-by-side prove-time + circuit-size comparison for the 5 cases in batch 1.
Both backends ultimately produce a Halo2 SHPLONK proof, but the lowering
paths differ:

- **Zinnia (`backend="halo2"`)** runs through `src/prove/halo2/synthesizer.rs`
  — a hand-written halo2 circuit that consumes IR statements directly. Each
  IR op assigns its own advice cells; high-level ops like `CosFIR` /
  `ArgmaxFIR` are kept as single IR statements and expanded inside the
  synthesizer.
- **Handwritten halo2-graph baseline** is the `target.rs` file per case,
  written against halo2-graph's `FixedPointChip` / `GateChip` / `RangeChip`
  exactly as `halo2_builder.py` would emit. Each high-level op is
  explicitly spelled out in primitive `qmul` / `qdiv` / `qsub` / `qadd` /
  `is_less_than` / `select` / `is_equal` calls.

Both use lookup-bits = 8 for Zinnia (its synthesizer requires
`precision_bits % lookup_bits == 0`, default precision 32 → must be 8) and
6 for halo2-graph (its conventional setting). `k` is each circuit's
**minimum viable** value — measured by stepping `k` from 10 upward until
the prover succeeds.

## Results

| Case | Op                                | Zinnia IR | Zinnia *k* | Zinnia prove | Zinnia proof | HW advice | HW fixed | HW range-chk | HW *k*ₘᵢₙ | HW prove | HW snark |
|------|-----------------------------------|----------:|-----------:|---------------:|-------------:|----------:|---------:|-------------:|-----------:|---------:|---------:|
| 0309 | `argmax` on Float[8,6]            |       235 |         11 |          1.47 s |    2 720 B |     4 668 |       71 |        1 081 |         10 |    188 ms |  3 876 B |
| 0323 | `cos(deg2rad(x))`                  |         6 |         12 |          2.04 s |    2 784 B |     7 393 |       43 |        2 226 |         10 |    253 ms |  3 876 B |
| 0329 | `a ** 3` on Float[8,6]            |       341 |         16 |        20.62 s |    3 040 B |    34 848 |       41 |       10 464 |         10 |    820 ms |  5 380 B |
| 0334 | `mean([a,b,c], axis=0)` Float[50] |       355 |         12 |          1.30 s |    2 784 B |    22 800 |       42 |        6 600 |         10 |    554 ms |  5 444 B |
| 0336 | `diag(fliplr(a))` Float[8,8]      |        75 |         10 |          0.98 s |    2 656 B |     1 448 |       27 |          368 |         10 |    150 ms |  4 100 B |

(Zinnia auto-picked *k* for every case except 0323 / 0329 where its
estimator under-sized; for those two I bisected upward from the estimate.)

## Reading the table

**Two metrics are *not* directly comparable.** Zinnia's "IR stmts" is
**post-optimisation Zinnia IR** — `CosFIR` / `ArgmaxFIR` are single ops
even though they each expand into many halo2 gates inside Zinnia's
synthesizer. Halo2-graph's "advice cells" is the **post-lowering** count.
So Zinnia's IR is small *because the IR is high-level*, not because the
final circuit is small.

The comparable size metric is **k** (log₂ of halo2 row count). On *k*
alone:

- **Handwritten halo2-graph fits in *k* = 10 for every case.** halo2-graph's
  `BaseCircuitBuilder` auto-allocates additional advice columns when the
  cell count exceeds 2¹⁰ ≈ 1024 rows, so even the 34 848-cell 0329
  circuit packs into 10 rows × ~34 columns.
- **Zinnia uses higher *k* (10 → 16).** Its synthesizer uses a fixed
  column layout (~5 advice columns) and grows rows instead. The
  resulting *k* is closer to a direct row count of the lowered work.

Prove time is the cleaner end-to-end metric:

| Case | Zinnia prove | HW prove (k = 10) | HW prove (k = Zinnia *k*) | Zinnia / HW |
|------|--------------:|------------------:|---------------------------:|-------------:|
| 0309 | 1.47 s         |  188 ms            |  202 ms                     |       7.3× |
| 0323 | 2.04 s         |  253 ms            |  204 ms                     |       10×  |
| 0329 | 20.62 s        |  820 ms            | 1 276 ms                     |       16×  |
| 0334 | 1.30 s         |  554 ms            |  393 ms                     |       3.3× |
| 0336 | 0.98 s         |  150 ms            |  150 ms                     |       6.5× |

Zinnia is **3.3× – 16× slower** than the handwritten halo2-graph baseline
across the 5 cases — the gap widens on arithmetic-heavy kernels (0329
power, 0323 cos) and narrows on indexing-heavy ones (0336 diag+fliplr).

## Where the gap comes from

1. **halo2-graph's `FixedPointChip` is heavily optimised.** Its
   `qmul`/`qdiv`/`qcos`/`qpow` use multi-column gate-chip layouts and
   share a single range-check lookup table across all ops. Zinnia's
   synthesizer assigns 4–5 advice cells per IR statement using a more
   uniform column layout — fewer fused gates.

2. **Zinnia's `k` estimator can over-size for arithmetic kernels.** For
   0329 `a ** power`, Zinnia's estimate jumped to *k* = 16 (≈ 65 k rows)
   because `PowFIR` with a runtime exponent unrolls in the synthesizer.
   The handwritten baseline bakes `power = 3` compile-time and packs the
   two `qmul` ops into ~10 k advice cells at *k* = 10. So part of the
   0329 gap reflects Zinnia handling *more general* work (runtime power)
   than the baseline's compile-time-3 cube.

3. **0336 (pure gather) is the closest match.** No arithmetic, just
   load-witness + equality-check; both backends land at *k* = 10 and
   prove in ~150 ms (HW) vs ~980 ms (Zinnia). The remaining 6.5× is the
   per-op overhead of Zinnia's synthesizer vs halo2-graph's gate chip.

4. **Proof size is consistently *smaller* for Zinnia.** This is purely a
   *k*-driven artifact — at higher *k* you'd expect bigger proofs, but
   halo2-graph adds advice columns which inflate the SHPLONK
   commitments. So Zinnia's row-heavy / column-light layout wins on
   wire-size, even though it pays in prover time.

## Reproducing

- Handwritten halo2-graph numbers:
  `python3 benchmarking/run_halo2_baselines.py --repeats 3` against the
  per-case `target.rs` files (after copying into `halo2-graph/examples/`).
  Each case is built with `RUSTUP_TOOLCHAIN=nightly-2025-12-01`.

- Zinnia numbers: `program = ZKCircuit.from_method(kernel).compile();
  program.prove(*args, backend="halo2", params={"k": k, "precision_bits":
  32, "lookup_bits": 8})`. For 0323 and 0329 you must pass `k`
  explicitly because Zinnia's `estimate_params` undersizes.
