# Remote zkVM sandwich — zk20

Completed successfully: **20/20 programs; all 97 applicable program/system pairs**. All **457 measured proofs** and **137 warmup proofs** verified and their outputs matched NumPy; all **137 per-stage tamper checks** rejected the modified proof. Cairo uses the verification-format procedure described below. All timings were measured on the remote machine; local execute checks are not timing inputs.

Measured source and rebuilt Zinnia extension: `a65574b5d00271cbb71c533761cf8263dd63a9f5` (`all-merged`). All 60 epoch manifests record that source commit. Remote measurement window: **2026-10-10T06:58:20+0000 to 2026-10-10T09:19:30+0000**; final warmup verification completed **09:20:07 UTC**. The precise last measurement-stage timestamp is in `measurement.log`.

The local branch later advanced to `6fa27f8e58bd450a46670cb981cb7beb2fc9dfc9` to add unrelated cases. A final diff check found no changes to these 20 programs, the compiler, harness, or shared arithmetic; the local-check agent reports no fixes. No case required remeasurement. See `source-recheck.json` and `local-check-done.md` in the result directory. The remote tracked source remained unchanged; its final status lists only an untracked `benchmarking/results/` directory. I made no edits or commits in local `wt/literal`.

Protocol: per program, Zinnia halo2/Noir ×5, SP1 core and RISC Zero composite (plus Cairo Stwo for DS-1000) ×1, then Zinnia halo2/Noir ×5. The harness also makes one untimed warmup per system/phase. All jobs have 8 threads pinned to one of 0–7, 8–15, 16–23, 24–31; no timed job uses SMT cores 32–63. Memory cap: 30 GB, or 64 GB for ResNet; step timeout 14,400 s. Setup uses fixture A; proofs and NumPy checks use fixture B. Prove-core excludes setup and verification. Zinnia columns pool the ten measured samples; zkVM columns contain the single measured sample. Ratios are pooled-median Zinnia / zkVM (smaller is faster). Separate before/after medians and exact ratios are in summary.csv.

## Per-program results

| Program | Systems proved | Outputs | Halo2 s | Noir s | SP1 s | RISC0 s | Cairo s | H/SP1 | H/R0 | H/Cairo | N/SP1 | N/R0 | N/Cairo |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ds1000_case0313 | H, N, SP1, R0, Cairo | match | 0.141943 | 0.121461 | 23.2897 | 4.11737 | 21.8048 | 0.00609464 | 0.0344741 | 0.0065097 | 0.0052152 | 0.0294996 | 0.00557037 |
| ds1000_case0330 | H, N, SP1, R0, Cairo | match | 0.246648 | 0.175432 | 22.438 | 6.82806 | 20.2651 | 0.0109924 | 0.0361228 | 0.0121711 | 0.00781852 | 0.0256928 | 0.00865685 |
| ds1000_case0356 | H, N, SP1, R0, Cairo | match | 0.137838 | 0.110995 | 26.1051 | 3.99907 | 36.6455 | 0.00528011 | 0.0344674 | 0.00376137 | 0.00425184 | 0.0277551 | 0.00302887 |
| ds1000_case0357 | H, N, SP1, R0, Cairo | match | 0.153341 | 0.145777 | 23.6418 | 3.93497 | 38.9384 | 0.00648604 | 0.0389689 | 0.00393806 | 0.00616606 | 0.0370465 | 0.00374378 |
| ds1000_case0365 | H, N, SP1, R0, Cairo | match | 0.168175 | 0.129963 | 24.1732 | 7.28427 | 13.7095 | 0.00695707 | 0.0230874 | 0.012267 | 0.00537633 | 0.0178416 | 0.00947978 |
| ds1000_case0370 | H, N, SP1, R0, Cairo | match | 0.0879163 | 0.128026 | 22.7731 | 3.54624 | 13.9327 | 0.00386054 | 0.0247914 | 0.00631008 | 0.00562181 | 0.0361018 | 0.00918888 |
| ds1000_case0382 | H, N, SP1, R0, Cairo | match | 0.308636 | 0.176611 | 25.8363 | 7.37341 | 16.2051 | 0.0119459 | 0.041858 | 0.0190457 | 0.00683579 | 0.0239524 | 0.0108985 |
| ds1000_case0413 | H, N, SP1, R0, Cairo | match | 0.0941244 | 0.148815 | 22.2801 | 3.42225 | 16.6644 | 0.00422459 | 0.0275037 | 0.00564824 | 0.00667927 | 0.0434846 | 0.00893012 |
| ds1000_case0417 | H, N, SP1, R0, Cairo | match | 0.0713608 | 0.120364 | 24.831 | 3.62818 | 15.2282 | 0.00287386 | 0.0196685 | 0.0046861 | 0.00484731 | 0.0331746 | 0.00790401 |
| ds1000_case0418 | H, N, SP1, R0, Cairo | match | 0.136365 | 0.127539 | 22.8144 | 7.22346 | 14.3832 | 0.00597716 | 0.0188781 | 0.00948085 | 0.0055903 | 0.0176562 | 0.00886722 |
| ds1000_case0419 | H, N, SP1, R0, Cairo | match | 0.139287 | 0.131797 | 24.4812 | 7.17463 | 16.2854 | 0.00568955 | 0.0194139 | 0.00855292 | 0.0053836 | 0.0183699 | 0.00809299 |
| ds1000_case0430 | H, N, SP1, R0, Cairo | match | 0.104202 | 0.128935 | 23.3244 | 6.98907 | 15.7109 | 0.00446752 | 0.0149093 | 0.00663246 | 0.00552792 | 0.0184481 | 0.00820672 |
| ds1000_case0456 | H, N, SP1, R0, Cairo | match | 0.229461 | 0.171858 | 25.8446 | 13.5881 | 15.051 | 0.0088785 | 0.0168869 | 0.0152456 | 0.00664967 | 0.0126477 | 0.0114184 |
| ds1000_case0457 | H, N, SP1, R0, Cairo | match | 0.288205 | 0.184666 | 22.3491 | 13.3785 | 13.5021 | 0.0128956 | 0.0215425 | 0.0213453 | 0.00826282 | 0.0138032 | 0.0136769 |
| ds1000_case0458 | H, N, SP1, R0, Cairo | match | 0.23816 | 0.17581 | 24.0444 | 13.3047 | 16.9807 | 0.00990502 | 0.0179004 | 0.0140253 | 0.0073119 | 0.0132141 | 0.0103535 |
| ds1000_case0463 | H, N, SP1, R0, Cairo | match | 0.116744 | 0.122029 | 23.5641 | 3.74238 | 17.9447 | 0.00495432 | 0.0311952 | 0.00650578 | 0.00517857 | 0.0326072 | 0.00680025 |
| ds1000_case0502 | H, N, SP1, R0, Cairo | match | 0.298373 | 0.149662 | 21.658 | 6.7701 | 12.5377 | 0.0137766 | 0.0440721 | 0.023798 | 0.00691028 | 0.0221064 | 0.011937 |
| npbench_deep_learning_resnet_resnet__resnet_basicblock | H, N, SP1, R0 | match | 11.283 | 2.21995 | 100.377 | 2349.6 | — | 0.112407 | 0.00480211 | — | 0.0221162 | 0.000944822 | — |
| npbench_polybench_floyd_warshall_floyd_warshall__kernel | H, N, SP1, R0 | match | 4.46803 | 0.754935 | 24.0077 | 25.5807 | — | 0.186108 | 0.174664 | — | 0.0314455 | 0.0295119 | — |
| pythran_ccopy__CCOPY | H, N, SP1, R0 | match | 2.17216 | 0.847351 | 21.7414 | 3.44679 | — | 0.099909 | 0.630199 | — | 0.038974 | 0.245838 | — |

H = zinnia_halo2; N = zinnia_noir. Cairo does not apply to the three NPBench/Pythran programs.

## Build and execution evidence

The Zinnia release build, SP1 host/guests, RISC Zero host/guests, Cairo package, and kernels-native runner all built successfully on the remote with `CARGO_BUILD_JOBS=6`, using one shared build tree per system. The build ended with `ALL_BUILDS_SUCCEEDED` and exit 0 at 06:57:51 UTC. Build records and the native-runner SHA-256 are in `build-records/`.

During the long ResNet run, separate remote work used the other 8-core groups. Saved affinity observations show ResNet on 8–15 and other jobs on 0–7, 16–23, and 24–31, without core or SMT-sibling overlap. Whole-machine load was therefore not constant across the entire multi-hour sandwich.

## Evidence and commands

Results: `/home/zhantong/Projects/zinnia-project/eval-redo/literal-remote/results-tw/2026-10-10-zk20/zk20`. Raw records include each sample’s prove-core time, output comparison, proof verification, and tamper-check outcome. Manifests include source commits, toolchain versions, build records, binary hashes, and machine/configuration details. Warmup verification is in `warmup-verification.json`. Proofs remain on the remote under `$RUN/zk20/epochs/*/work/`; the local copy omits bulky work directories.

Build and transport commands: `remote-commands.md` beside this report. Launcher: `zk20-run.py`; exact per-shard harness commands are in `zk20-commands-s*.jsonl` in the result directory. Build log: `build.log` in the result directory.

Cairo follows the existing ff1 harness: prove-core times binary Stwo generation; Scarb 2.20.1 verifies a separately regenerated JSON proof. The binary serialization is not directly accepted by that verifier. Both are generated from the same execution. This is a verification-format limitation, not a wrapping step.

## Unverified or incomplete

No missing applicable system or unchecked output in the measured samples; every measured proof and warmup has a successful verification verdict, subject to Cairo’s serialization caveat above.

## Exact commands


Transport: `/home/zhantong/Projects/zinnia-project/eval-redo/tools/remote.sh`.
Remote `$RUN=/home/zhantong/zinnia-eval-run`; checkout `$RUN/zinnia-literal`.

Initial build (remote.sh bg zk20-build):
```bash
set -e
cd "$RUN/zinnia-literal"
export CARGO_BUILD_JOBS=6
for system in sp1 risc0 cairo; do
  cp -a --reflink=auto "$RUN/zinnia-src/benchmarking/baselines/$system/target" "benchmarking/baselines/$system/target"
done
.venv/bin/python -m benchmarking.harness build --jobs 6 --baselines sp1,risc0,cairo
cd benchmarking/baselines/zkvm_common/kernels
CARGO_TARGET_DIR=../../target cargo build --release --features std
echo ALL_BUILDS_SUCCEEDED
```

Shared targets are built once before timed work. Each proof uses the harness's isolated work directory. Initial commit: `a65574b5d00271cbb71c533761cf8263dd63a9f5`.

Launcher: `zk20-run.py` in this directory, copied to `$RUN/zk20-run.py`. Every exact harness command, CPU affinity and timestamp is written to remote `$RUN/zk20/zk20-commands-s{0,1,2,3}.jsonl`. Results are under `$RUN/zk20/epochs`.

After `ALL_BUILDS_SUCCEEDED` and exit 0 (2026-10-10 06:57:51 UTC), launch (`remote.sh bg zk20-measure`):
```bash
set -e
mkdir -p "$RUN/zk20"
cp "$RUN/logs/zk20-build.log" "$RUN/zk20/build.log"
cp "$RUN"/zk20-*.py "$RUN/zk20/"
export CARGO_BUILD_JOBS=6
"$RUN/zinnia-literal/.venv/bin/python" "$RUN/zk20-run.py"
cd "$RUN/zinnia-literal"
taskset -c 0-7 .venv/bin/python "$RUN/zk20-verify-warmups.py"
echo ZK20_MEASUREMENT_AND_VERIFICATION_FINISHED
```

Read-only execution diagnostic during the long ResNet RISC Zero warmup (not a measurement sample; unused cores 0–7):
```bash
cd "$RUN/zinnia-literal"
RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 HARDWARE_CONCURRENCY=8 taskset -c 0-7 \
  benchmarking/baselines/risc0/target/release/risc0-baseline \
  --case-dir benchmarking/datasets/npbench/npbench_deep_learning_resnet_resnet \
  --function resnet_basicblock \
  --fixture benchmarking/datasets/npbench/npbench_deep_learning_resnet_resnet/fixture.json \
  --input B --mode execute --proof-mode composite \
  --work-dir "$RUN/zk20/diagnostic-resnet" --out "$RUN/zk20/resnet-risc0-execute.json"
```
It confirms 21,031,674 user cycles, 22,151,168 total cycles, 22 segments, matching the local execute-check workload. No proof timing is taken from this diagnostic.

The warmup-verification script sets its affinity to cores 8–15 on entry, reusing the last active measurement shard after ResNet completes. This overrides the launch wrapper's initial 0–7 affinity and avoids occupying cores that other remote work may have taken after our earlier shards finished. It does not alter any timed proof.

Final collection (all timed work and warmup verification had finished):
```bash
remote.sh run 'set -e; taskset -pc 8-15 $$ >/dev/null; cp logs/zk20-measure.log zk20/measurement.log; cp zk20-*.py zk20/; mkdir -p zk20/build-records; cp zinnia-literal/target/zinnia-harness-*.json zk20/build-records/; test -x zinnia-literal/benchmarking/baselines/target/release/kernels-native; sha256sum zinnia-literal/benchmarking/baselines/target/release/kernels-native > zk20/build-records/kernels-native.sha256; git -C zinnia-literal status --porcelain > zk20/final-remote-git-status.txt; git -C zinnia-literal rev-parse HEAD > zk20/final-remote-commit.txt; tar --exclude="*/work" -czf zk20-results.tar.gz zk20'
remote.sh pull /home/zhantong/zinnia-eval-run/zk20-results.tar.gz /home/zhantong/Projects/zinnia-project/eval-redo/literal-remote/results-tw/2026-10-10-zk20
tar -xzf /home/zhantong/Projects/zinnia-project/eval-redo/literal-remote/results-tw/2026-10-10-zk20/zk20-results.tar.gz -C /home/zhantong/Projects/zinnia-project/eval-redo/literal-remote/results-tw/2026-10-10-zk20
python3 /home/zhantong/Projects/zinnia-project/eval-redo/all-work/zkvm20/summarize-remote.py /home/zhantong/Projects/zinnia-project/eval-redo/literal-remote/results-tw/2026-10-10-zk20/zk20 /home/zhantong/Projects/zinnia-project/eval-redo/all-work/zkvm20/remote-measure-done.md
```
Here `remote.sh` is the absolute transport path at the top of this document. The local report adds the final source recheck and verification counts to the generated table.

ALL_PROOFS_VERIFIED
