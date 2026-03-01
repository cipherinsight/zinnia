"""
Private KPI Benchmarking Example
--------------------------------

This example proves a private portfolio's KPI performance against public benchmark
thresholds (NPL ratio and flagged-transaction ratio). The circuit reveals only
benchmark-level conclusions (`outperform_benchmark` and a disclosed tier), while
keeping the underlying business volumes private.
"""


from zinnia import *

from typing import Any

from examples.utils import create_circuit, create_prove_verify_keys, prove, verify


def generate_private_kpi_input(
    total_loans: float,
    nonperforming_loans: float,
    total_transactions: float,
    flagged_transactions: float,
    benchmark_npl_bps: float = 350.0,
    benchmark_fraud_bps: float = 45.0,
) -> dict[str, Any]:
    # Guard rails for ratio computations.
    if total_loans <= 0:
        raise ValueError("total_loans must be positive.")
    if total_transactions <= 0:
        raise ValueError("total_transactions must be positive.")

    # Basis-point arithmetic avoids floating-point division inside circuit logic.
    npl_ok = nonperforming_loans * 10_000 <= total_loans * benchmark_npl_bps
    fraud_ok = flagged_transactions * 10_000 <= total_transactions * benchmark_fraud_bps
    outperform_benchmark = npl_ok and fraud_ok

    # Map binary checks into a coarse, publicly disclosed tier.
    if npl_ok and fraud_ok:
        disclosed_performance_tier = 2
    elif npl_ok or fraud_ok:
        disclosed_performance_tier = 1
    else:
        disclosed_performance_tier = 0

    # Emit complete payload expected by the circuit interface.
    return {
        "total_loans": total_loans,
        "nonperforming_loans": nonperforming_loans,
        "total_transactions": total_transactions,
        "flagged_transactions": flagged_transactions,
        "benchmark_npl_bps": benchmark_npl_bps,
        "benchmark_fraud_bps": benchmark_fraud_bps,
        "outperform_benchmark": outperform_benchmark,
        "disclosed_performance_tier": disclosed_performance_tier,
    }


EXAMPLE_INPUT = generate_private_kpi_input(
    total_loans=250_000.0,
    nonperforming_loans=7_200.0,
    total_transactions=4_600_000.0,
    flagged_transactions=14_500.0,
)


@zk_circuit
def private_kpi_benchmarking(
    total_loans: Private[Float],
    nonperforming_loans: Private[Float],
    total_transactions: Private[Float],
    flagged_transactions: Private[Float],
    benchmark_npl_bps: Public[Float],
    benchmark_fraud_bps: Public[Float],
    outperform_benchmark: Public[Boolean],
    disclosed_performance_tier: Public[Integer],
):
    # Safety constraints ensure denominator-like terms remain valid.
    assert total_loans > 0
    assert total_transactions > 0

    # Recompute KPI checks in-circuit from private amounts + public thresholds.
    npl_ok = nonperforming_loans * 10_000 <= total_loans * benchmark_npl_bps
    fraud_ok = flagged_transactions * 10_000 <= total_transactions * benchmark_fraud_bps

    # Explicit control-flow demonstrates Boolean branching support in Zinnia.
    expected_outperform = False
    if npl_ok:
        if fraud_ok:
            expected_outperform = True

    # Public claim must match constrained internal decision.
    assert outperform_benchmark == expected_outperform

    # Tier disclosure is constrained to policy-consistent coarse categories.
    if npl_ok and fraud_ok:
        assert disclosed_performance_tier == 2
    else:
        if npl_ok or fraud_ok:
            assert disclosed_performance_tier == 1
        else:
            assert disclosed_performance_tier == 0


def run_private_kpi_benchmarking(
    data: dict[str, Any] | None = None,
    circuit_name: str = "private_kpi_benchmarking",
    k: int = 16,
):
    # Convenience default keeps the file directly runnable as an end-to-end demo.
    if data is None:
        data = EXAMPLE_INPUT

    # Keep caller data immutable through pipeline stages.
    proving_data = dict(data)

    # Standard pipeline helpers: circuit creation, key generation, proving, verification.
    circuit = create_circuit(private_kpi_benchmarking, chips=[])
    keygen_result = create_prove_verify_keys(
        circuit,
        proving_data,
        circuit_name=circuit_name,
        k=k,
    )
    prove_result = prove(circuit, keygen_result, proving_data)
    verify_result = verify(keygen_result)

    return {
        "keygen": keygen_result,
        "prove": prove_result,
        "verify": verify_result,
    }


if __name__ == '__main__':
    run_private_kpi_benchmarking(
        data=EXAMPLE_INPUT,
        circuit_name="private_kpi_benchmarking",
        k=16,
    )
