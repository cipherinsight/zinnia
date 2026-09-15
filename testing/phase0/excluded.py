"""Parity circuits phase 0 is expected to reject, and why.

`revision/phase0/coverage.md` §1 defines the parity set as the programs the
baseline compiles; §9 lists the baseline defects phase 0 does not replicate
and names the programs that depend on them. This module is the
machine-readable form of that list, so that `test_coverage.py` can mark those
circuits as expected rejections instead of failures. Every entry is a
`program::function` label with the reason it is excluded; the reasons are the
§9 rows and the two design limits of `definition.md` that the corpus reaches.

The list is checked strictly: a circuit listed here that phase 0 *accepts*
fails the test, so an entry cannot outlive its cause.
"""

from __future__ import annotations

from enum import Enum


class Exclusion(str, Enum):
    """Why a parity circuit is expected to be rejected (`coverage.md` §9)."""

    #: The baseline compiles a `range` with a runtime bound to zero iterations
    #: (§9, first row). Phase 0 unrolls to the declared maximum, so it rejects
    #: when no maximum is declared, when the bound is a `Float`, when the
    #: unrolled program exceeds the work budget, or when a comprehension or a
    #: list length would depend on the trip count.
    RANGE_ZERO_ITERATIONS = "baseline compiles the runtime-bounded range to zero iterations"

    #: Two branches of different rank merge in the baseline through
    #: `select_value`; `definition.md` §6.1 requires equal rank.
    RANK_MISMATCH_MERGE = "branches of different rank (definition.md §6.1)"

    #: The baseline compiles the construct to nothing or to a silent no-op
    #: (§9: comprehension over a scalar, subscript of a scalar, an index that
    #: NumPy rejects).
    SILENT_MISCOMPILE = "baseline silently miscompiles the construct"

    #: `compile_validation.json` records `compile_ok`, but the current baseline
    #: no longer compiles the program (§9, "entries the baseline no longer
    #: compiles").
    STALE_RECORD = "the current baseline no longer compiles it"


_RANGE = Exclusion.RANGE_ZERO_ITERATIONS

#: `program::function` -> why it is expected to be rejected.
EXCLUDED: dict[str, Exclusion] = {
    # `range` bound is a `Float` parameter (carries no term).
    "npbench_polybench_fdtd_2d_fdtd_2d::kernel": _RANGE,
    "npbench_polybench_jacobi_1d_jacobi_1d::kernel": _RANGE,
    "npbench_polybench_jacobi_2d_jacobi_2d::kernel": _RANGE,
    "npbench_polybench_seidel_2d_seidel_2d::kernel": _RANGE,
    "pythran_diffusion::diffuseNumpy": _RANGE,
    "pythran_ramsurf::matrc": _RANGE,
    # Trip count has no declared maximum, or one past the work budget.
    "pythran_hyantes_core::run": _RANGE,
    "pythran_matrix_class_distance::matrix_class_distance": _RANGE,
    "pythran_monte_carlo::montecarlo_integration": _RANGE,
    "pythran_primes_sieve2::get_primes7": _RANGE,
    "pythran_sexy_primes::primes_below": _RANGE,
    "pythran_stone::whetstone": _RANGE,
    # A comprehension or a list length would depend on the trip count.
    "pythran_laplace::calc": _RANGE,
    "pythran_mandel::mandel": _RANGE,
    "pythran_matmul::matrix_multiply": _RANGE,
    "pythran_sobelfilter::sobelFilter": _RANGE,
    "pythran_sum_primes::sum_primes": _RANGE,
    "pythran_zero::zero": _RANGE,
    "pythran_blacksholes::BlackScholes": _RANGE,
    # Loop nests over N <= 1024 exceed the work budget once unrolled.
    "npbench_polybench_lu_lu::kernel": _RANGE,
    "npbench_polybench_syr2k_syr2k::kernel": _RANGE,
    "npbench_polybench_syrk_syrk::kernel": _RANGE,
    "npbench_polybench_trmm_trmm::kernel": _RANGE,
    "pythran_allpairs::sqr_dists": _RANGE,
    "pythran_bubble_sort::bubble_sort": _RANGE,
    "pythran_growcut::test": _RANGE,
    "pythran_insertion_sort::insertion_sort": _RANGE,
    "pythran_lu::factorMatrix0": _RANGE,
    "pythran_lu::factorMatrix1": _RANGE,
    # `if rows.shape[0] == 0: result = np.array([]) else: result = im[...]`.
    "ds1000_case0507::kernel": Exclusion.RANK_MISMATCH_MERGE,
    "ds1000_case0509::kernel": Exclusion.RANK_MISMATCH_MERGE,
    "ds1000_case0510::kernel": Exclusion.RANK_MISMATCH_MERGE,
    # Comprehension over a `str` input compiles to zero IR statements;
    # a rank-1 array indexed on two axes; an index tuple of `Float` values.
    "pythran_scrabble::scrabble_fun_score": Exclusion.SILENT_MISCOMPILE,
    "pythran_lapl2d::lapl2d": Exclusion.SILENT_MISCOMPILE,
    "pythran_deuxd_convolution::conv": Exclusion.SILENT_MISCOMPILE,
    # Recorded `compile_ok`; the current baseline raises during IR generation.
    "pythran_primes_sieve::primes_sieve": Exclusion.STALE_RECORD,
    "pythran_spectraldns::cross2": Exclusion.STALE_RECORD,
}
