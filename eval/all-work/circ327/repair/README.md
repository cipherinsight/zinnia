# CirC repair artifacts

`merged.json` is keyed logically by `(program, function)` and follows the order of `../programs.json`. `REPORT.md` records the previous and repaired classifications. Only this directory was written during the repair.

Each entry directory contains its source identity in `entry.json`, the original assessment in `old-result.json`, executable Z# attempts (where expressible), compiler logs and exact commands, and its final `result.json`. Multi-entry programs have separate `program__function` directories. NumPy oracles are loaded from the original kernel with Zinnia imports, annotations and decorators removed; no Zinnia IR, circuits or sizes are used.

Compile an attempt using its `*.command.json`; all compilations use `MEMRUN_MAX=8G`, `memrun.sh`, `timeout 300`, and `TMPDIR` inside `repair/tmp`. For a final candidate, the following re-runs compilation and the A/B checks:

```
/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python repair/run.py <entry-directory-name>
```

Run this command from the parent of `repair/`. `repair/recheck.py` rechecks accepted compilations without recompiling. `repair/check_unaccepted.py` checks candidates with failed compilations without upgrading their classification. The exact translated source and imported helper named in that source are the authoritative artifacts; the builder scripts retain implementation history and should not be rerun over final candidates.

Real inputs use scale `2^32`. Native unsigned representations use signed two's-complement decoding. Complex values are pairs or named structs; the checker accounts for CirC's alphabetical struct-field order. Wide products use either 32-bit parts or a native field product plus native bit unpacking before rescaling. Native `/` and `%` compute quotient digits without overflowing a pre-scaled numerator. There are no custom witness inputs or manually asserted arithmetic constraints. Integer Fibonacci outputs use twelve 64-bit limbs to cover the entire declared `n <= 1024` domain. Output metadata specifies logical dimensions or optional-return presence where needed.

A successful `zxi` evaluation never substitutes for successful R1CS compilation. A compile failure, timeout or memory kill remains a failure even when A/B evaluation succeeds. The correctness tolerance is `1e-6` relative plus `1e-6` absolute after decoding. `before-*` and `field-loop.*` artifacts preserve superseded attempts and diagnostics.

Progress logs include superseded classifications. Per-entry `result.json` and `merged.json` are the final classifications.
