# Epoch 2026-10-09-tw1-npb-s7

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (9 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (9 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (9 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 9 | 9 | - |
| zinnia_noir | 9 | 9 | - |
| halo2_hw | 9 | 9 | - |
| noir_hw | 9 | 9 | - |
| halo2_lit | 9 | 9 | - |
| noir_lit | 9 | 9 | - |

## Headline: all-supported subset without copy-only cases

9 cases completed by every system that do more than copy inputs: npbench_polybench_bicg_bicg__kernel, npbench_polybench_gramschmidt_gramschmidt__kernel, npbench_polybench_syrk_syrk__kernel, pythran_allpairs__sqr_dists_loops, pythran_deuxd_convolution__conv, pythran_ising__ising_step, pythran_make_decision__md, pythran_periodic_dist__dist, pythran_spectraldns__add_pressure_diffusion

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 9 | 0.896 | [0.836, 0.958] | 8 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 9 | 0.909 | [0.848, 0.973] | 7 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 9 | 0.904 | [0.810, 0.997] | 7 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 9 | 1.012 | [0.977, 1.046] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 9 | 1.018 | [0.985, 1.050] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 9 | 1.121 | [1.056, 1.201] | 1 / 8 |
| zinnia_noir | noir_hw | verify_s | 9 | 1.015 | [0.932, 1.116] | 4 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 9 | 2.966 | [2.048, 4.130] | 0 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 9 | 2.726 | [1.938, 3.693] | 0 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 9 | 1.731 | [1.124, 2.547] | 2 / 7 |
| zinnia_halo2 | noir_hw | verify_s | 9 | 1.266 | [0.893, 1.795] | 2 / 7 |
| zinnia_halo2 | halo2_lit | prove_core_s | 9 | 0.411 | [0.176, 0.787] | 8 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 9 | 0.416 | [0.179, 0.798] | 8 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 9 | 0.531 | [0.230, 0.947] | 6 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 9 | 1.460 | [0.590, 3.218] | 2 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 9 | 1.313 | [0.514, 2.936] | 3 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 9 | 0.891 | [0.374, 1.926] | 4 / 5 |
| zinnia_halo2 | noir_lit | verify_s | 9 | 1.212 | [0.905, 1.602] | 2 / 7 |
| zinnia_noir | halo2_hw | prove_core_s | 9 | 0.306 | [0.224, 0.437] | 9 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 9 | 0.339 | [0.256, 0.472] | 9 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 9 | 0.588 | [0.431, 0.852] | 7 / 2 |
| zinnia_noir | halo2_hw | verify_s | 9 | 0.724 | [0.580, 0.897] | 7 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 9 | 0.140 | [0.050, 0.326] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 9 | 0.156 | [0.056, 0.353] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 9 | 0.270 | [0.098, 0.630] | 7 / 2 |
| zinnia_noir | halo2_lit | verify_s | 9 | 0.425 | [0.181, 0.806] | 7 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 9 | 0.498 | [0.187, 1.017] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 9 | 0.490 | [0.178, 1.026] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 9 | 0.576 | [0.237, 1.116] | 4 / 5 |
| zinnia_noir | noir_lit | verify_s | 9 | 0.972 | [0.919, 1.023] | 6 / 3 |

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

3 headline cases of group npbench: npbench_polybench_bicg_bicg__kernel, npbench_polybench_gramschmidt_gramschmidt__kernel, npbench_polybench_syrk_syrk__kernel

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 3 | 0.906 | [0.768, 0.987] | 3 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 3 | 0.920 | [0.779, 1.005] | 2 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 3 | 0.937 | [0.870, 0.989] | 3 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 3 | 1.053 | [1.035, 1.086] | 0 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 3 | 1.058 | [1.045, 1.078] | 0 / 3 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 3 | 1.100 | [1.053, 1.130] | 0 / 3 |
| zinnia_noir | noir_hw | verify_s | 3 | 0.971 | [0.944, 1.007] | 2 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 3 | 2.821 | [1.399, 4.660] | 0 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 3 | 2.627 | [1.370, 4.176] | 0 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 3 | 1.625 | [0.753, 2.537] | 1 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 3 | 1.296 | [0.846, 1.702] | 1 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 3 | 0.836 | [0.645, 1.015] | 2 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 3 | 0.849 | [0.655, 1.031] | 2 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 3 | 0.993 | [0.764, 1.298] | 2 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 3 | 2.646 | [1.172, 4.666] | 0 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 3 | 2.476 | [1.161, 4.192] | 0 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 3 | 1.547 | [0.660, 2.552] | 1 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 3 | 1.242 | [0.764, 1.585] | 1 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 3 | 0.338 | [0.217, 0.766] | 3 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 3 | 0.371 | [0.250, 0.792] | 3 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 3 | 0.623 | [0.388, 1.405] | 2 / 1 |
| zinnia_noir | halo2_hw | verify_s | 3 | 0.702 | [0.555, 1.064] | 2 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 3 | 0.312 | [0.195, 0.691] | 3 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 3 | 0.342 | [0.216, 0.714] | 3 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 3 | 0.575 | [0.326, 1.268] | 2 / 1 |
| zinnia_noir | halo2_lit | verify_s | 3 | 0.744 | [0.486, 1.447] | 2 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 3 | 0.988 | [0.910, 1.036] | 1 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 3 | 0.997 | [0.914, 1.054] | 1 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 3 | 1.047 | [0.923, 1.137] | 1 / 2 |
| zinnia_noir | noir_lit | verify_s | 3 | 0.930 | [0.851, 1.009] | 2 / 1 |

## Group pythran: headline cases

6 headline cases of group pythran: pythran_allpairs__sqr_dists_loops, pythran_deuxd_convolution__conv, pythran_ising__ising_step, pythran_make_decision__md, pythran_periodic_dist__dist, pythran_spectraldns__add_pressure_diffusion

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 6 | 0.891 | [0.825, 0.962] | 5 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 6 | 0.903 | [0.836, 0.977] | 5 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 6 | 0.887 | [0.758, 1.029] | 4 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.992 | [0.953, 1.036] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.999 | [0.962, 1.040] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 1.131 | [1.034, 1.243] | 1 / 5 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.037 | [0.914, 1.183] | 2 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 3.041 | [1.926, 4.536] | 0 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 2.777 | [1.804, 3.999] | 0 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 1.787 | [1.018, 2.915] | 1 / 5 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 1.251 | [0.746, 2.092] | 1 / 5 |
| zinnia_halo2 | halo2_lit | prove_core_s | 6 | 0.288 | [0.095, 0.687] | 6 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 6 | 0.292 | [0.096, 0.695] | 6 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 6 | 0.388 | [0.125, 0.860] | 4 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 1.084 | [0.328, 3.231] | 2 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 0.956 | [0.270, 2.947] | 3 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.676 | [0.221, 1.933] | 3 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 1.198 | [0.795, 1.761] | 1 / 5 |
| zinnia_noir | halo2_hw | prove_core_s | 6 | 0.290 | [0.209, 0.433] | 6 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 6 | 0.325 | [0.243, 0.468] | 6 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 6 | 0.572 | [0.414, 0.880] | 5 / 1 |
| zinnia_noir | halo2_hw | verify_s | 6 | 0.736 | [0.548, 0.954] | 5 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 6 | 0.094 | [0.025, 0.290] | 6 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 6 | 0.105 | [0.030, 0.314] | 6 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 6 | 0.185 | [0.051, 0.579] | 5 / 1 |
| zinnia_noir | halo2_lit | verify_s | 6 | 0.322 | [0.101, 0.756] | 5 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 0.354 | [0.095, 1.028] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 0.344 | [0.089, 1.034] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 0.428 | [0.127, 1.145] | 3 / 3 |
| zinnia_noir | noir_lit | verify_s | 6 | 0.993 | [0.929, 1.050] | 4 / 2 |

## All-supported subset

9 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_polybench_bicg_bicg__kernel, npbench_polybench_gramschmidt_gramschmidt__kernel, npbench_polybench_syrk_syrk__kernel, pythran_allpairs__sqr_dists_loops, pythran_deuxd_convolution__conv, pythran_ising__ising_step, pythran_make_decision__md, pythran_periodic_dist__dist, pythran_spectraldns__add_pressure_diffusion

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 9 | 0.896 | [0.836, 0.958] | 8 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 9 | 0.909 | [0.848, 0.973] | 7 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 9 | 0.904 | [0.810, 0.997] | 7 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 9 | 1.012 | [0.977, 1.046] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 9 | 1.018 | [0.985, 1.050] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 9 | 1.121 | [1.056, 1.201] | 1 / 8 |
| zinnia_noir | noir_hw | verify_s | 9 | 1.015 | [0.932, 1.116] | 4 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 9 | 2.966 | [2.048, 4.130] | 0 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 9 | 2.726 | [1.938, 3.693] | 0 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 9 | 1.731 | [1.124, 2.547] | 2 / 7 |
| zinnia_halo2 | noir_hw | verify_s | 9 | 1.266 | [0.893, 1.795] | 2 / 7 |
| zinnia_halo2 | halo2_lit | prove_core_s | 9 | 0.411 | [0.176, 0.787] | 8 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 9 | 0.416 | [0.179, 0.798] | 8 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 9 | 0.531 | [0.230, 0.947] | 6 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 9 | 1.460 | [0.590, 3.218] | 2 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 9 | 1.313 | [0.514, 2.936] | 3 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 9 | 0.891 | [0.374, 1.926] | 4 / 5 |
| zinnia_halo2 | noir_lit | verify_s | 9 | 1.212 | [0.905, 1.602] | 2 / 7 |
| zinnia_noir | halo2_hw | prove_core_s | 9 | 0.306 | [0.224, 0.437] | 9 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 9 | 0.339 | [0.256, 0.472] | 9 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 9 | 0.588 | [0.431, 0.852] | 7 / 2 |
| zinnia_noir | halo2_hw | verify_s | 9 | 0.724 | [0.580, 0.897] | 7 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 9 | 0.140 | [0.050, 0.326] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 9 | 0.156 | [0.056, 0.353] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 9 | 0.270 | [0.098, 0.630] | 7 / 2 |
| zinnia_noir | halo2_lit | verify_s | 9 | 0.425 | [0.181, 0.806] | 7 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 9 | 0.498 | [0.187, 1.017] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 9 | 0.490 | [0.178, 1.026] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 9 | 0.576 | [0.237, 1.116] | 4 / 5 |
| zinnia_noir | noir_lit | verify_s | 9 | 0.972 | [0.919, 1.023] | 6 / 3 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 9 | 0.896 | [0.836, 0.958] | 8 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 9 | 0.909 | [0.848, 0.973] | 7 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 9 | 0.904 | [0.810, 0.997] | 7 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 9 | 1.012 | [0.977, 1.046] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 9 | 1.018 | [0.985, 1.050] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 9 | 1.121 | [1.056, 1.201] | 1 / 8 |
| zinnia_noir | noir_hw | verify_s | 9 | 1.015 | [0.932, 1.116] | 4 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 9 | 2.966 | [2.048, 4.130] | 0 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 9 | 2.726 | [1.938, 3.693] | 0 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 9 | 1.731 | [1.124, 2.547] | 2 / 7 |
| zinnia_halo2 | noir_hw | verify_s | 9 | 1.266 | [0.893, 1.795] | 2 / 7 |
| zinnia_halo2 | halo2_lit | prove_core_s | 9 | 0.411 | [0.176, 0.787] | 8 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 9 | 0.416 | [0.179, 0.798] | 8 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 9 | 0.531 | [0.230, 0.947] | 6 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 9 | 1.460 | [0.590, 3.218] | 2 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 9 | 1.313 | [0.514, 2.936] | 3 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 9 | 0.891 | [0.374, 1.926] | 4 / 5 |
| zinnia_halo2 | noir_lit | verify_s | 9 | 1.212 | [0.905, 1.602] | 2 / 7 |
| zinnia_noir | halo2_hw | prove_core_s | 9 | 0.306 | [0.224, 0.437] | 9 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 9 | 0.339 | [0.256, 0.472] | 9 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 9 | 0.588 | [0.431, 0.852] | 7 / 2 |
| zinnia_noir | halo2_hw | verify_s | 9 | 0.724 | [0.580, 0.897] | 7 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 9 | 0.140 | [0.050, 0.326] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 9 | 0.156 | [0.056, 0.353] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 9 | 0.270 | [0.098, 0.630] | 7 / 2 |
| zinnia_noir | halo2_lit | verify_s | 9 | 0.425 | [0.181, 0.806] | 7 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 9 | 0.498 | [0.187, 1.017] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 9 | 0.490 | [0.178, 1.026] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 9 | 0.576 | [0.237, 1.116] | 4 / 5 |
| zinnia_noir | noir_lit | verify_s | 9 | 0.972 | [0.919, 1.023] | 6 / 3 |

## Excluded cases

- npbench_azimint_hist_azimint_hist__azimint_hist: rejected_compiler_limitation (np.histogram of runtime data): ZinniaException: IR generation failed: histogram: first slice supports compile-time-data histogram only; runtime/private values are not supported
- npbench_azimint_naive_azimint_naive__azimint_naive: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_cavity_flow_cavity_flow__cavity_flow: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_channel_flow_channel_flow__channel_flow: rejected_resource_limit (compile time over 300 s): timeout after 300s
- npbench_contour_integral_contour_integral__contour_integral: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_deep_learning_lenet_lenet__lenet5: rejected_compiler_limitation (operation on a runtime-shaped array): ZinniaException: IR generation failed: this operation is not supported on an array with a runtime shape (the result of a boolean mask, nonzero/argwhere/unique, or a slice with runtime bounds)
- npbench_deep_learning_resnet_resnet__resnet_basicblock: rejected_resource_limit (compile time over 300 s): timeout after 300s
- npbench_mandelbrot1_mandelbrot1__mandelbrot: rejected_no_provable_bound (np.linspace num from a runtime value): ZinniaException: IR generation failed: linspace `num` must be a compile-time constant int
- npbench_mandelbrot2_mandelbrot2__mandelbrot: rejected_unsupported_construct (assignment to an Attribute): ZinniaException:   In method "mandelbrot", line 10
        Xi.shape = Yi.shape = C.shape = xn * yn
        ^^^^^^^^
InvalidAssignStatementException: Unsupported assignment target `Attribute`; only nam
- npbench_nbody_nbody__nbody: rejected_compiler_limitation (precondition of `pow_f` neither proved nor checkable in the circuit): ZinniaException: IR generation failed: op `pow_f` requires precondition `BoolComb { op: Or, operands: [Cmp { op: Ne, lhs: Var(Value(ValueId(48545))), rhs: LitFloat(ContractFloat(0.0)) }, Cmp { op: Ge,
- npbench_polybench_adi_adi__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_cholesky2_cholesky2__kernel: rejected_compiler_limitation (np.linalg.cholesky (result-returning linear algebra)): ZinniaException: IR generation failed: np.linalg.cholesky: implicit result-returning linear algebra is deferred because @zk_external currently returns one scalar, not an array witness; supply an expli
- npbench_polybench_correlation_correlation__kernel: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_polybench_covariance_covariance__kernel: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_polybench_doitgen_doitgen__kernel: rejected_compiler_limitation (operation on a runtime-shaped array): ZinniaException: IR generation failed: this operation is not supported on an array with a runtime shape (the result of a boolean mask, nonzero/argwhere/unique, or a slice with runtime bounds)
- npbench_polybench_fdtd_2d_fdtd_2d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_floyd_warshall_floyd_warshall__kernel: rejected_resource_limit (compile time over 300 s): timeout after 300s
- npbench_polybench_heat_3d_heat_3d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_jacobi_1d_jacobi_1d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_jacobi_2d_jacobi_2d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_nussinov_nussinov__kernel: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_polybench_seidel_2d_seidel_2d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_spmv_spmv__spmv: rejected_compiler_limitation (matmul of a runtime-shaped array): ZinniaException: IR generation failed: DynamicNDArray matmul is not yet supported for bounded-dynamic arrays; use static arrays or keep matmul outside the bounded-dynamic surface
- npbench_stockham_fft_stockham_fft__stockham_fft: rejected_unsupported_construct (np.mgrid): ZinniaException: IR generation failed: np.mgrid is not implemented
- pythran_allpairs_distances__allpairs_distances: rejected_no_provable_bound (np.arange length from a runtime value): ZinniaException: IR generation failed: range stop must be a compile-time constant int
- pythran_allpairs_distances_loops__allpairs_distances_loops: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 1 must be a compile-time constant int
- pythran_another_quicksort__QuickSort: rejected_no_provable_bound (while-loop trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: the `while` condition was not proven false within loop_limit=1000 unrolled iterations. Bound the 
- pythran_approximated_callgraph__approximated_callgraph: rejected_compiler_limitation (list() of a runtime-shaped array): ZinniaException: IR generation failed: list() of an array with a runtime shape is not supported: its length is not a compile-time number
- pythran_babylonian__is_square: rejected_unsupported_construct (set): ZinniaException:   In method "is_square", line 5
        seen = {x}
               ^^^
UnsupportedLangFeatureException: Set literals are not supported inside @zk_circuit because circuits have no set t
- pythran_blacksholes__BlackScholes: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_calculate_u__timeloop: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_caxpy__CAXPY: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_ccopy__CCOPY: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_cdotc__CDOTC: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_cdotu__CDOTU: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_check_mask__check_mask: rejected_unsupported_construct (np.bitwise_and): ZinniaException: IR generation failed: np.bitwise_and is not implemented
- pythran_clip2__clip: rejected_compiler_limitation (shape element from a bounded runtime value where a constant is needed): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int (it is a runtime value in [0, 2]; a runtime size is not supported here)
- pythran_collatz__collatz: rejected_compiler_limitation (operation on a runtime-shaped array): ZinniaException: IR generation failed: this operation is not supported on an array with a runtime shape (the result of a boolean mask, nonzero/argwhere/unique, or a slice with runtime bounds)
- pythran_collatz_modified__collatz_modified: rejected_no_provable_bound (while-loop trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: the `while` condition was not proven false within loop_limit=1000 unrolled iterations. Bound the 
- pythran_compute_subpix_2d_gaussian2__compute_subpix_2d_gaussian2: rejected_compiler_limitation (np.nonzero of a runtime-shaped array): ZinniaException: IR generation failed: nonzero: DynamicNDArray input is deferred; use static numeric arrays for now
- pythran_convnet__convnet: rejected_compiler_limitation (precondition of `div_f` neither proved nor checkable in the circuit): ZinniaException: IR generation failed: op `div_f` requires precondition `Cmp { op: Ne, lhs: Var(Value(ValueId(2013))), rhs: LitFloat(ContractFloat(0.0)) }` could not be lowered to a witness constraint
- pythran_deriv__deriv: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_diffusion__diffuseNumpy: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_emin__run: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_euler_challenge14__euler14: rejected_unsupported_construct (dict): ZinniaException:   In method "euler14", line 5
        lengths = {1: 0}
                  ^^^^^^
UnsupportedLangFeatureException: Dict literals are not supported inside @zk_circuit because mapping typ
- pythran_factorize_naive__factorize_naive: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_fannkuch__fannkuch2: rejected_compiler_limitation (range() with a runtime bound used as a value): ZinniaException: IR generation failed: range() with a runtime argument is only supported as the iterable of a `for` loop or comprehension (got a Integer that is not a compile-time constant)
- pythran_fdtd__fdtd: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_fft__fft: rejected_unsupported_construct (call of `fft` (no builtin, chip or numpy function of that name in scope)): ZinniaException: IR generation failed: Unknown function `fft` in @zk_circuit. If `fft` is a helper you defined, decorate it with @zk_chip (or @zk_external for off-circuit witness functions). no @zk_ch
- pythran_fibo__test: rejected_compiler_limitation (recursion deeper than the inlining limit (16)): ZinniaException: IR generation failed: RecursionLimitExceededError: recursion limit (16) exceeded while inlining @zk_chip `rfibo`. Each recursive call into a chip is unrolled at compile time; if your 
- pythran_gauss__gauss: rejected_compiler_limitation (chip that returns on some paths only): ZinniaException: IR generation failed: Chip return inference does not support partial returns without an explicit return annotation
- pythran_grouping__grouping: rejected_compiler_limitation (np.concatenate/np.stack of nested lists): ZinniaException: IR generation failed: concatenate/stack: unsupported element type List { elements: [Integer] }
- pythran_growcut__test: rejected_resource_limit (compile memory over 8G): out of memory after 56s
- pythran_guerre__guerre: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_histogram__histogram: rejected_no_provable_bound (list repetition with a runtime count): ZinniaException: IR generation failed: sequence repetition `seq * n` with a runtime count `n` is not supported: a list's length must be known at compile time (use np.zeros/np.full with a bounded shape
- pythran_hyantes_core__run: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_lapl2d__lapl2d: rejected_compiler_limitation (array name bound to different arrays on runtime paths, then written): ZinniaException: IR generation failed: Reading `In` is not supported here: a name that refers to this array on some paths and to another array on others (bound under a runtime condition) was used to m
- pythran_laplace__calc: rejected_no_provable_bound (list repetition with a runtime count): ZinniaException: IR generation failed: sequence repetition `seq * n` with a runtime count `n` is not supported: a list's length must be known at compile time (use np.zeros/np.full with a bounded shape
- pythran_ldpc_decoder__Decoding_logBP: rejected_unsupported_construct (method or attribute `.dot`): ZinniaException: IR generation failed: Named attribute `X.dot` not yet implemented in Rust IR generator
- pythran_lttb_pt__downsample: rejected_compiler_limitation (split sections from a bounded runtime value where a constant is needed): ZinniaException: IR generation failed: split sections must be a compile-time constant int (it is a runtime value in [-2, 1022]; a runtime size is not supported here)
- pythran_make_circle_two_points___make_circle_one_point: rejected_unsupported_construct (`is` / `is not`): ZinniaException:   In method "make_circle_two_points", line 13
            if c is None:
               ^^^^^^^^^
UnsupportedOperatorException: `is` / `is not` compare object identity, which circuit v
- pythran_mandel__mandel: rejected_resource_limit (compile memory over 8G): out of memory after 24s
- pythran_matrix_class_distance__matrix_class_distance: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_monte_carlo__montecarlo_integration: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_morphology__dilate_decompose_loops: rejected_no_provable_bound (range() trip count (proven bound too large to unroll)): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: the proven bound on the number of iterations of range() is 9223372036854775806, more than the 167
- pythran_mulmod__gf2mulmod: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_nd_local_maxima__local_maxima: rejected_unsupported_construct (unannotated entry parameter): ZinniaException:   In method "local_maxima", line 2
    def local_maxima(data: Float[Array, '8 6 4 2'], mode=wrap):
                                                    ^^^^
InvalidCircuitInputExceptio
- pythran_nqueens__n_queens: rejected_unsupported_construct (`is` / `is not`): ZinniaException:   In method "permutations", line 6
        if r is None:
           ^^^^^^^^^
UnsupportedOperatorException: `is` / `is not` compare object identity, which circuit values do not have; 
- pythran_pivot__pivot: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_primes_sieve2__get_primes7: rejected_compiler_limitation (range() with a runtime bound used as a value): ZinniaException: IR generation failed: range() with a runtime argument is only supported as the iterable of a `for` loop or comprehension (got a Integer that is not a compile-time constant)
- pythran_primes_sieve__primes_sieve: rejected_no_provable_bound (list repetition with a runtime count): ZinniaException: IR generation failed: sequence repetition `seq * n` with a runtime count `n` is not supported: a list's length must be known at compile time (use np.zeros/np.full with a bounded shape
- pythran_pselect__pselect: rejected_unsupported_construct (function used as a value): ZinniaException: IR generation failed: functions as values are not supported: `sel0` is used without being called (circuits have no function values; call it directly)
- pythran_queens_numba__solve: rejected_unsupported_construct (list mutated through a second name (aliasing)): ZinniaException: IR generation failed: mutating this list through `queens_x` is not supported: the same list is also reachable through `queens_x` (Python shares a list on assignment, argument passing 
- pythran_quicksort__quicksort: rejected_no_provable_bound (while-loop trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: the `while` condition was not proven false within loop_limit=1000 unrolled iterations. Bound the 
- pythran_repeating__repeating: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_sexy_primes__primes_below: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_shallow_water__run: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_slowparts__slowparts: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_sobelfilter__sobelFilter: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_stone__whetstone: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_sum_primes__sum_primes: rejected_unsupported_construct (np.ceil): ZinniaException: IR generation failed: np.ceil is not implemented
- pythran_train_eq__train_eq: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_train_eq__train_eq2: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_wave_simulation__test: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 1 must be a compile-time constant int
- pythran_zero__zero: rejected_resource_limit (compile memory over 8G): out of memory after 22s

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

