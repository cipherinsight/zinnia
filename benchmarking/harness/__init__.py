"""The benchmark harness: one measurement protocol for every proving system.

Entry point: ``python -m benchmarking.harness {check-env,build,run,analyze,accept}``
(see ``benchmarking/README.md`` for usage, the protocol and the per-system
program contract).

Module map:

- ``paths``        where the repository, datasets, baselines and results live
- ``toolchains``   the pinned toolchain versions and the ``check-env`` probe
- ``environment``  machine facts, per-case snapshots, CPU-only child env, indexer pause
- ``build``        the release build of the extension and its build record
- ``manifest``     the epoch manifest (what produced the numbers)
- ``fixtures``     ``fixture.json`` loading (inputs A and B)
- ``case_sets``    ``datasets/case_sets/*.json`` loading (cases, tolerances)
- ``kernels``      loading and compiling ``kernel.py`` files; the scope rule
- ``reference``    the NumPy reference (the kernel body run as plain NumPy)
- ``outputs``      decoding public values and the ordered output check
- ``isolation``    running one step in a child process under the memory cap
- ``protocol``     the per-case protocol and the failure taxonomy
- ``step``         the child-process entry point that runs one adapter step
- ``adapters/``    one adapter per proving system
- ``analysis/``    summaries computed only from an epoch directory
- ``acceptance/``  compile + mock-run + reference compare for the acceptance sets
"""
