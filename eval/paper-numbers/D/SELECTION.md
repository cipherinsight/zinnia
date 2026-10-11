# Selection evidence

Every group in every lowering round is listed below. Group IDs restart in each round. `selection.jsonl` also lists every selected helper node and its estimated cost; these estimates are not measured cell counts.

## default_halo2

| Round | Group | Family | Lanes | Chosen |
|---|---|---|---|---|
| 1 | 0 | sort | 6 | sort: permutation check |
| 1 | 1 | permutation scatter | 6 | gather: permutation scatter by hits |
| 1 | 2 | element | 6 | gather: element by a recorded one-hot |
| 1 | 3 | element | 6 | gather: element by a recorded one-hot |
| 1 | 4 | element | 6 | gather: element by a recorded one-hot |
| 1 | 5 | element | 6 | gather: element by a recorded one-hot |
| 1 | 6 | element | 6 | gather: element by a recorded one-hot |
| 1 | 7 | element | 1 | memory: element by index bits |
| 1 | 8 | element | 1 | memory: element by index bits |
| 1 | 9 | window | 30 | gather: window shift |
| 2 | 0 | element | 6 | memory: element by index bits |

## default_noir

| Round | Group | Family | Lanes | Chosen |
|---|---|---|---|---|
| 1 | 0 | sort | 6 | sort: permutation check |
| 1 | 1 | permutation scatter | 6 | gather: permutation scatter by hits |
| 1 | 2 | element | 6 | gather: element by a recorded one-hot |
| 1 | 3 | element | 6 | gather: element by a recorded one-hot |
| 1 | 4 | element | 6 | gather: element by a recorded one-hot |
| 1 | 5 | element | 6 | gather: element by a recorded one-hot |
| 1 | 6 | element | 6 | gather: element by a recorded one-hot |
| 1 | 7 | element | 1 | memory: element by index bits |
| 1 | 8 | element | 1 | memory: element by index bits |
| 1 | 9 | window | 30 | gather: window shift |
| 2 | 0 | element | 6 | memory: element by table read |

## local_halo2

| Round | Group | Family | Lanes | Chosen |
|---|---|---|---|---|
| 1 | 0 | sort | 6 | sort: permutation check |
| 1 | 1 | permutation scatter | 6 | gather: permutation scatter by hits |
| 1 | 2 | element | 6 | memory: element by index bits |
| 1 | 3 | element | 6 | memory: element by index bits |
| 1 | 4 | element | 6 | memory: element by index bits |
| 1 | 5 | element | 6 | memory: element by index bits |
| 1 | 6 | element | 6 | memory: element by index bits |
| 1 | 7 | element | 1 | memory: element by index bits |
| 1 | 8 | element | 1 | memory: element by index bits |
| 1 | 9 | window | 30 | gather: window shift |
| 2 | 0 | element | 6 | memory: element by index bits |

## local_noir

| Round | Group | Family | Lanes | Chosen |
|---|---|---|---|---|
| 1 | 0 | sort | 6 | sort: permutation check |
| 1 | 1 | permutation scatter | 6 | gather: permutation scatter by hits |
| 1 | 2 | element | 6 | memory: element by table read |
| 1 | 3 | element | 6 | memory: element by table read |
| 1 | 4 | element | 6 | memory: element by table read |
| 1 | 5 | element | 6 | memory: element by table read |
| 1 | 6 | element | 6 | memory: element by table read |
| 1 | 7 | element | 1 | memory: element by index bits |
| 1 | 8 | element | 1 | memory: element by index bits |
| 1 | 9 | window | 30 | gather: window shift |
| 2 | 0 | element | 6 | memory: element by table read |
