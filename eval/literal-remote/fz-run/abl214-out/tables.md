214 programs (172 compared in RQ2). Ratio = variant / final (size); geomeans over programs where both compile.

## halo2: final measured 214, failed 0 []
| variant | gmean 214 (n) | gmean 172 (n) | changed (172) | up / down | IR-only | variant fails | largest increases |
|---|---|---|---|---|---|---|---|
| fix_delegate | 1.0099 (208) | 1.0119 (172) | 15 (15) | 10 / 5 | 0 | none | pythran_insertion_sort ×1.89, pythran_lu ×1.51, pythran_lu ×1.45, 0347 ×1.26, 0346 ×1.24, 0314 ×1.20, 0433 ×1.14, 0434 ×1.10 |
| abl_layer_facts | 1.0139 (208) | 1.0169 (172) | 30 (30) | 30 / 0 | 8 | none | pythran_multitype ×4.82, 0298 ×1.94, 0434 ×1.11, 0318 ×1.07, 0501 ×1.06, 0502 ×1.06, 0326 ×1.05, 0381 ×1.05 |
| abl_combined | 1.0869 (207) | 1.1055 (172) | 51 (51) | 51 / 0 | 35 | 0389 | 0381 ×5.66, 0382 ×5.66, pythran_multitype ×4.82, 0383 ×4.69, 0375 ×4.35, 0319 ×2.23, 0450 ×2.01, npbench_scattering_self_energies_scattering_self_energies ×1.98 |
| abl_check | 1.0001 (208) | 1.0001 (172) | 2 (2) | 2 / 0 | 34 | none | 0507 ×1.01, 0510 ×1.00 |
| fix_selects | 1.0733 (208) | 1.0893 (172) | 38 (38) | 38 / 0 | 4 | none | 0381 ×5.66, 0382 ×5.66, 0383 ×4.69, 0375 ×4.35, 0319 ×2.20, 0450 ×2.01, npbench_scattering_self_energies_scattering_self_energies ×1.98, 0374 ×1.97 |
| abl_joint | 1.0117 (208) | 1.0142 (172) | 13 (13) | 13 / 0 | 0 | none | 0319 ×1.93, 0450 ×1.68, 0503 ×1.27, 0504 ×1.26, 0318 ×1.25, 0505 ×1.24, pythran_lu ×1.10, 0346 ×1.06 |
| abl_smt | 1.0013 (207) | 1.0015 (172) | 3 (3) | 3 / 0 | 1 | 0389 | 0353 ×1.28, 0507 ×1.01, 0510 ×1.00 |
| abl_array_facts | 1.0093 (208) | 1.0112 (172) | 16 (16) | 16 / 0 | 3 | none | pythran_multitype ×4.82, 0318 ×1.22, 0326 ×1.05, 0319 ×1.03, 0298 ×1.02, pythran_make_decision ×1.01, 0450 ×1.01, pythran_pairwise ×1.01 |

## noir: final measured 214, failed 0 []
| variant | gmean 214 (n) | gmean 172 (n) | changed (172) | up / down | IR-only | variant fails | largest increases |
|---|---|---|---|---|---|---|---|
| fix_delegate | 1.0280 (214) | 1.0349 (172) | 17 (17) | 17 / 0 | 0 | none | pythran_lu ×4.25, pythran_lu ×3.88, pythran_insertion_sort ×2.04, 0434 ×1.60, pythran_ccopy ×1.48, 0501 ×1.36, 0502 ×1.36, 0433 ×1.31 |
| abl_layer_facts | 1.0106 (214) | 1.0132 (172) | 31 (31) | 23 / 8 | 6 | none | pythran_multitype ×7.33, pythran_fibo_seq ×1.17, 0298 ×1.11, 0501 ×1.03, 0502 ×1.03, 0504 ×1.02, 0505 ×1.02, 0503 ×1.02 |
| abl_combined | 1.0501 (213) | 1.0625 (172) | 52 (52) | 50 / 2 | 36 | 0389 | pythran_multitype ×7.33, 0381 ×3.27, 0382 ×3.27, 0383 ×2.99, npbench_scattering_self_energies_scattering_self_energies ×2.52, 0375 ×2.08, 0450 ×1.64, 0319 ×1.51 |
| abl_check | 1.0000 (214) | 1.0000 (172) | 2 (2) | 2 / 0 | 35 | none | 0507 ×1.00, 0510 ×1.00 |
| fix_selects | 1.0386 (214) | 1.0483 (172) | 42 (42) | 42 / 0 | 4 | none | 0381 ×3.27, 0382 ×3.27, 0383 ×2.99, npbench_scattering_self_energies_scattering_self_energies ×2.52, 0375 ×2.08, 0450 ×1.64, 0319 ×1.50, 0300 ×1.26 |
| abl_joint | 1.0016 (214) | 1.0019 (172) | 13 (13) | 13 / 0 | 0 | none | pythran_ccopy ×1.07, 0510 ×1.06, 0508 ×1.06, 0509 ×1.04, 0507 ×1.03, pythran_lu ×1.03, 0345 ×1.01, pythran_lu ×1.01 |
| abl_smt | 1.0003 (213) | 1.0003 (172) | 4 (4) | 4 / 0 | 1 | 0389 | 0353 ×1.06, 0507 ×1.00, 0510 ×1.00, 0375 ×1.00 |
| abl_array_facts | 1.0102 (214) | 1.0128 (172) | 10 (10) | 10 / 0 | 3 | none | pythran_multitype ×7.33, pythran_fibo_seq ×1.17, 0318 ×1.01, 0319 ×1.01, 0298 ×1.01, 0450 ×1.00, 0326 ×1.00, npbench_polybench_covariance2_covariance2 ×1.00 |

