# Epoch 2026-10-09-aa1-npb-s0

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (1 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (1 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (1 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 1 | 1 | - |
| zinnia_noir | 1 | 1 | - |
| halo2_hw | 1 | 1 | - |
| noir_hw | 1 | 1 | - |
| halo2_lit | 1 | 1 | - |
| noir_lit | 1 | 1 | - |

## Headline: all-supported subset without copy-only cases

1 cases completed by every system that do more than copy inputs: npbench_deep_learning_resnet_resnet__resnet_basicblock

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.328 | [0.328, 0.328] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.339 | [0.339, 0.339] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.597 | [0.597, 0.597] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.391 | [0.391, 0.391] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.402 | [0.402, 0.402] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.686 | [0.686, 0.686] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.911 | [0.911, 0.911] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 2.233 | [2.233, 2.233] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 1.800 | [1.800, 1.800] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 1.640 | [1.640, 1.640] | 0 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 2.426 | [2.426, 2.426] | 0 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.279 | [0.279, 0.279] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.288 | [0.288, 0.288] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 0.517 | [0.517, 0.517] | 1 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 2.234 | [2.234, 2.234] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 1.788 | [1.788, 1.788] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 1.620 | [1.620, 1.620] | 0 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 2.737 | [2.737, 2.737] | 0 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 0.057 | [0.057, 0.057] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 0.076 | [0.076, 0.076] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 0.142 | [0.142, 0.142] | 1 / 0 |
| zinnia_noir | halo2_hw | verify_s | 1 | 0.224 | [0.224, 0.224] | 1 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 0.049 | [0.049, 0.049] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 0.064 | [0.064, 0.064] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 0.121 | [0.121, 0.121] | 1 / 0 |
| zinnia_noir | halo2_lit | verify_s | 1 | 0.194 | [0.194, 0.194] | 1 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.391 | [0.391, 0.391] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.399 | [0.399, 0.399] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.678 | [0.678, 0.678] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.028 | [1.028, 1.028] | 0 / 1 |

## Copy-only cases (reported apart)

0 all-supported cases that only copy inputs: (none)

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 0 | - | - | - |
| zinnia_halo2 | halo2_hw | end_to_end_s | 0 | - | - | - |
| zinnia_halo2 | halo2_hw | verify_s | 0 | - | - | - |
| zinnia_noir | noir_hw | prove_core_s | 0 | - | - | - |
| zinnia_noir | noir_hw | end_to_end_s | 0 | - | - | - |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_noir | noir_hw | verify_s | 0 | - | - | - |
| zinnia_halo2 | noir_hw | prove_core_s | 0 | - | - | - |
| zinnia_halo2 | noir_hw | end_to_end_s | 0 | - | - | - |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_halo2 | noir_hw | verify_s | 0 | - | - | - |
| zinnia_halo2 | halo2_lit | prove_core_s | 0 | - | - | - |
| zinnia_halo2 | halo2_lit | end_to_end_s | 0 | - | - | - |
| zinnia_halo2 | halo2_lit | verify_s | 0 | - | - | - |
| zinnia_halo2 | noir_lit | prove_core_s | 0 | - | - | - |
| zinnia_halo2 | noir_lit | end_to_end_s | 0 | - | - | - |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_halo2 | noir_lit | verify_s | 0 | - | - | - |
| zinnia_noir | halo2_hw | prove_core_s | 0 | - | - | - |
| zinnia_noir | halo2_hw | end_to_end_s | 0 | - | - | - |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_noir | halo2_hw | verify_s | 0 | - | - | - |
| zinnia_noir | halo2_lit | prove_core_s | 0 | - | - | - |
| zinnia_noir | halo2_lit | end_to_end_s | 0 | - | - | - |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_noir | halo2_lit | verify_s | 0 | - | - | - |
| zinnia_noir | noir_lit | prove_core_s | 0 | - | - | - |
| zinnia_noir | noir_lit | end_to_end_s | 0 | - | - | - |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_noir | noir_lit | verify_s | 0 | - | - | - |

## Group npbench: headline cases

1 headline cases of group npbench: npbench_deep_learning_resnet_resnet__resnet_basicblock

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.328 | [0.328, 0.328] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.339 | [0.339, 0.339] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.597 | [0.597, 0.597] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.391 | [0.391, 0.391] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.402 | [0.402, 0.402] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.686 | [0.686, 0.686] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.911 | [0.911, 0.911] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 2.233 | [2.233, 2.233] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 1.800 | [1.800, 1.800] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 1.640 | [1.640, 1.640] | 0 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 2.426 | [2.426, 2.426] | 0 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.279 | [0.279, 0.279] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.288 | [0.288, 0.288] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 0.517 | [0.517, 0.517] | 1 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 2.234 | [2.234, 2.234] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 1.788 | [1.788, 1.788] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 1.620 | [1.620, 1.620] | 0 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 2.737 | [2.737, 2.737] | 0 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 0.057 | [0.057, 0.057] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 0.076 | [0.076, 0.076] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 0.142 | [0.142, 0.142] | 1 / 0 |
| zinnia_noir | halo2_hw | verify_s | 1 | 0.224 | [0.224, 0.224] | 1 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 0.049 | [0.049, 0.049] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 0.064 | [0.064, 0.064] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 0.121 | [0.121, 0.121] | 1 / 0 |
| zinnia_noir | halo2_lit | verify_s | 1 | 0.194 | [0.194, 0.194] | 1 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.391 | [0.391, 0.391] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.399 | [0.399, 0.399] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.678 | [0.678, 0.678] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.028 | [1.028, 1.028] | 0 / 1 |

## All-supported subset

1 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_deep_learning_resnet_resnet__resnet_basicblock

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.328 | [0.328, 0.328] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.339 | [0.339, 0.339] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.597 | [0.597, 0.597] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.391 | [0.391, 0.391] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.402 | [0.402, 0.402] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.686 | [0.686, 0.686] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.911 | [0.911, 0.911] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 2.233 | [2.233, 2.233] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 1.800 | [1.800, 1.800] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 1.640 | [1.640, 1.640] | 0 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 2.426 | [2.426, 2.426] | 0 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.279 | [0.279, 0.279] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.288 | [0.288, 0.288] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 0.517 | [0.517, 0.517] | 1 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 2.234 | [2.234, 2.234] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 1.788 | [1.788, 1.788] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 1.620 | [1.620, 1.620] | 0 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 2.737 | [2.737, 2.737] | 0 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 0.057 | [0.057, 0.057] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 0.076 | [0.076, 0.076] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 0.142 | [0.142, 0.142] | 1 / 0 |
| zinnia_noir | halo2_hw | verify_s | 1 | 0.224 | [0.224, 0.224] | 1 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 0.049 | [0.049, 0.049] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 0.064 | [0.064, 0.064] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 0.121 | [0.121, 0.121] | 1 / 0 |
| zinnia_noir | halo2_lit | verify_s | 1 | 0.194 | [0.194, 0.194] | 1 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.391 | [0.391, 0.391] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.399 | [0.399, 0.399] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.678 | [0.678, 0.678] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.028 | [1.028, 1.028] | 0 / 1 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.328 | [0.328, 0.328] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.339 | [0.339, 0.339] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.597 | [0.597, 0.597] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.391 | [0.391, 0.391] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.402 | [0.402, 0.402] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.686 | [0.686, 0.686] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.911 | [0.911, 0.911] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 2.233 | [2.233, 2.233] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 1.800 | [1.800, 1.800] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 1.640 | [1.640, 1.640] | 0 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 2.426 | [2.426, 2.426] | 0 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.279 | [0.279, 0.279] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.288 | [0.288, 0.288] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 0.517 | [0.517, 0.517] | 1 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 2.234 | [2.234, 2.234] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 1.788 | [1.788, 1.788] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 1.620 | [1.620, 1.620] | 0 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 2.737 | [2.737, 2.737] | 0 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 0.057 | [0.057, 0.057] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 0.076 | [0.076, 0.076] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 0.142 | [0.142, 0.142] | 1 / 0 |
| zinnia_noir | halo2_hw | verify_s | 1 | 0.224 | [0.224, 0.224] | 1 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 0.049 | [0.049, 0.049] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 0.064 | [0.064, 0.064] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 0.121 | [0.121, 0.121] | 1 / 0 |
| zinnia_noir | halo2_lit | verify_s | 1 | 0.194 | [0.194, 0.194] | 1 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.391 | [0.391, 0.391] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.399 | [0.399, 0.399] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.678 | [0.678, 0.678] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.028 | [1.028, 1.028] | 0 / 1 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

