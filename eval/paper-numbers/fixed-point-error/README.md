# Fixed-point error of the compiled circuits, final compiler

Metric (as `zinnia-paper/oopsla-2027-r1/fig/data/fixed-point-error.py`): for every float output y of
NumPy (float64, the epoch's `raw/_reference/<case>.json`) and y' of the circuit (the public outputs
of the first proof sample on fixture B, `raw/<system>/<case>.json`), error = |y' - y| / max(1, |y|),
i.e. absolute below 1 and relative above. The maximum is over all float outputs of all programs.

The paper script takes the newest record per program over all epochs by file mtime, which took
42 programs from the older zk42 runs (compilers a65574b5/6fa27f8e). `fixed_point_error.py` takes
every record from the epoch-fz proving runs only (`2026-10-11-fz-{ds,npb}-s{0,1,2}`), whose manifests
it checks to be commit 2569c9ed, clean: all 214 programs of `rq2-all.txt` (copy), both backends.

Run on the remote host shuaicpu6 (container zhantong-dev), 2026-10-12 08:06 UTC, over the remote
epoch directory:

```sh
source $RUN/env.sh; cd $RUN/paper-numbers-final/fixed-point-error
$RUN/zinnia-sizefix/.venv/bin/python fixed_point_error.py $RUN/fz/epochs fixed-point-error.json
```

Result (`run.log`, per-program values in `fixed-point-error.json`), identical for halo2-base and Noir:
214 programs, 151 with float outputs; maximum **1.674e-06** (NPBench `resnet_basicblock`), median of
the per-program maxima 1.83e-10; only that one program reaches 1e-6. The largest plain absolute
difference |y' - y| is 4.218e-05 (Pythran `rosen`, an output above 1 in magnitude, relative error
below the maximum). Same maximum as the mixed-epoch value (1.67e-06).
