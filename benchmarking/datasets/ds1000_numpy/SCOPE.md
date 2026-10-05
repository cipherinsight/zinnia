# DS-1000 NumPy scope inventory

Written by `benchmarking/datasets/tools/ds1000_scope.py`; the rules are stated in
`benchmarking/datasets/FILTERING.md`. Do not edit by hand.

- Source: `https://raw.githubusercontent.com/xlang-ai/DS-1000/main/data/ds1000.jsonl.gz`
- NumPy programs: 220
- In scope: 163
- Out of scope: 57

| Rule | Programs it fires on |
| --- | ---: |
| `imports` | 32 |
| `exception_control_flow` | 1 |
| `nondeterminism` | 3 |
| `non_numeric_data` | 10 |
| `non_finite_values` | 18 |

## Out of scope

| Problem | Rules and evidence |
| --- | --- |
| 0292 | `non_finite_values`: np.isnan, test input x: test input contains NaN or infinity |
| 0293 | `non_finite_values`: np.isnan, np.inf, test input x: test input contains NaN or infinity |
| 0294 | `non_finite_values`: np.isnan, test input x: test input contains NaN or infinity |
| 0305 | `non_finite_values`: np.nan |
| 0306 | `non_finite_values`: np.nan |
| 0307 | `non_finite_values`: np.nan |
| 0308 | `nondeterminism`: np.random.seed(), np.random.randint() |
| 0315 | `non_finite_values`: np.isnan, test input a: test input contains NaN or infinity |
| 0316 | `non_finite_values`: np.isnan, test input a: test input contains NaN or infinity |
| 0333 | `non_finite_values`: np.nan |
| 0344 | `non_numeric_data`: test input mystr: test input is a str |
| 0350 | `imports`: solution uses `scipy` from the exec_context import of scipy.stats |
| 0351 | `imports`: solution uses `scipy` from the exec_context import of scipy.stats; `non_finite_values`: test input a: test input contains NaN or infinity |
| 0352 | `imports`: solution uses `scipy` from the exec_context import of scipy.stats |
| 0367 | `imports`: import copy; `non_numeric_data`: test input array_of_arrays: object array (e.g. arrays of different shapes) |
| 0371 | `imports`: import scipy.integrate |
| 0372 | `imports`: import scipy.integrate |
| 0376 | `nondeterminism`: np.random.shuffle() |
| 0377 | `imports`: a test input is a torch object (test input needs a library: name 'torch' is not defined) |
| 0378 | `imports`: solution uses `torch` from the exec_context import of torch |
| 0379 | `imports`: a test input is a tensorflow object (test input needs a library: name 'tf' is not defined) |
| 0380 | `imports`: solution uses `tf` from the exec_context import of tensorflow |
| 0394 | `non_numeric_data`: .replace() string method, test input string: test input is a str |
| 0395 | `imports`: import scipy.stats |
| 0396 | `imports`: import scipy.stats |
| 0397 | `imports`: import scipy.stats |
| 0404 | `imports`: solution uses `pd` from the exec_context import of pandas; `non_numeric_data`: test input index: array of strings |
| 0410 | `nondeterminism`: np.random.seed(), np.random.choice() |
| 0421 | `imports`: import scipy.special |
| 0423 | `imports`: a test input is a pandas object (test input needs pandas.Timestamp) |
| 0424 | `imports`: a test input is a pandas object (test input needs pandas.Timestamp) |
| 0432 | `non_finite_values`: np.nan, np.nanpercentile |
| 0445 | `imports`: solution uses `rankdata` from the exec_context import of scipy.stats.rankdata |
| 0446 | `imports`: solution uses `rankdata` from the exec_context import of scipy.stats.rankdata |
| 0447 | `imports`: solution uses `rankdata` from the exec_context import of scipy.stats.rankdata |
| 0455 | `imports`: a test input is a pandas object (test input needs pandas.DataFrame); `non_finite_values`: np.nan |
| 0459 | `non_numeric_data`: test input A: array of strings |
| 0460 | `non_numeric_data`: test input A: array of strings |
| 0461 | `non_numeric_data`: test input A: array of strings |
| 0464 | `imports`: solution uses `pd` from the exec_context import of pandas |
| 0465 | `imports`: solution uses `pd` from the exec_context import of pandas |
| 0466 | `imports`: solution uses `pd` from the exec_context import of pandas |
| 0469 | `non_finite_values`: test input a: test input contains NaN or infinity |
| 0470 | `non_finite_values`: test input a: test input contains NaN or infinity |
| 0473 | `non_numeric_data`: test input CNTS: ragged sequence (setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (4,) + inhomogeneous part.) |
| 0474 | `non_finite_values`: np.isnan, test input c: test input contains NaN or infinity |
| 0475 | `imports`: solution uses `intp` from the exec_context import of scipy.interpolate |
| 0476 | `imports`: solution uses `pd` from the exec_context import of pandas; `non_numeric_data`: test input data: test input is a dict |
| 0479 | `imports`: solution uses `pd` from the exec_context import of pandas; `non_numeric_data`: test input start: test input is a str |
| 0484 | `imports`: solution uses `pd` from the exec_context import of pandas, a test input is a pandas object (test input needs pandas.DataFrame) |
| 0486 | `imports`: solution uses `MinMaxScaler` from the exec_context import of sklearn.preprocessing.MinMaxScaler |
| 0487 | `imports`: import sklearn.preprocessing |
| 0488 | `imports`: solution uses `MinMaxScaler` from the exec_context import of sklearn.preprocessing.MinMaxScaler |
| 0492 | `non_finite_values`: test input s1: test input contains NaN or infinity |
| 0493 | `exception_control_flow`: try statement |
| 0494 | `non_finite_values`: np.isnan, test input a: test input contains NaN or infinity |
| 0506 | `imports`: a test input is a pandas object (test input needs pandas.DataFrame); `non_finite_values`: np.nan |
