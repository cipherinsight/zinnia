# NumPy coverage probe on the final compiler

Commit `2569c9edb8a2bf0b88e9351ca5752f164d2fad24` (checkout `/home/zhantong/zinnia-eval-run/zinnia-sizefix`,
clean, extension SHA-256 `637a5c9a…1324`), remote host shuaicpu6 (container zhantong-dev),
2026-10-12 08:05 UTC. numpy 2.2.2, Python 3.14.7 (the remote venv).

`np-probe.py` is a verbatim copy of `zinnia-paper/oopsla-2027-r1/fig/data/np-probe.py`. Command:

```sh
source $RUN/env.sh; cd $RUN/paper-numbers-final/np-probe && ./run.sh > run.log 2>&1
```

Candidates (dir() of the remote numpy): 392 numpy, 32 numpy.linalg, 57 math, 74 ndarray methods.
Supported (`np-probe.json`): 155 numpy + 1 numpy.linalg (`norm`) = **156 NumPy functions**,
15 math functions, 33 ndarray methods. The supported sets are identical, name by name, to the
3ebe0558 run (`fig/data/np-probe.json`, also `../np_probe.json`). `run.log` holds the compiler's
rejection messages for the unsupported candidates and the summary at the end.
