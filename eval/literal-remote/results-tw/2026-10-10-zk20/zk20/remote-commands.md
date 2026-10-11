# zk20 remote commands

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
