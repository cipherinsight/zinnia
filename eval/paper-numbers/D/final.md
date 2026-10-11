Completed [REPORT.md](/home/zhantong/Projects/zinnia-project/eval-redo/paper-numbers/D/REPORT.md).

| Metric | Default | Local |
|---|---:|---:|
| Halo2-base cells | **5,494 (X)** | **6,046 (Y)** |
| Noir gates | 3,997 | 3,974 |

Both inputs match NumPy in all four configurations.

Sort verification and scatter one-hot sharing hold. The scan claims do not: no scans are selected, both nonempty assertions remain, and the window builds fresh equality tests.

Reproduction scripts and complete evidence are saved in D/.