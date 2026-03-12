"""
AML Structuring Detection Proof Example
---------------------------------------

This example models an AML (anti-money-laundering) structuring detector where raw
transaction amounts remain private, while the prover publicly discloses only
policy outcomes (alert flag + peak near-threshold count). The circuit proves that
those disclosed values are consistent with private transaction history under an
explicit threshold policy.
"""


from zinnia import *

from typing import Any
import numpy as onp

from utils import create_circuit, create_prove_verify_keys, prove, verify


TX_COUNT = 12
WINDOW_SIZE = 4


def generate_aml_structuring_input(
    cash_transactions: list[int],
    reporting_threshold: int = 10_000,
    proximity_margin: int = 1_000,
    max_near_in_any_window: int = 2,
) -> dict[str, Any]:
    # Pre-processing mirrors the circuit logic in plain Python so we can generate
    # witness data and expected public claims for a full prove/verify flow.
    if len(cash_transactions) != TX_COUNT:
        raise ValueError(f"cash_transactions must have length {TX_COUNT}.")

    # Convert once to an ndarray for vectorized reference computation.
    tx_array = onp.asarray(cash_transactions, dtype=int)
    # Define the "near-threshold" interval used to detect structuring behavior.
    lower_bound = reporting_threshold - proximity_margin
    near_mask = (tx_array >= lower_bound) & (tx_array < reporting_threshold)
    # Sliding-window aggregation: count near-threshold transactions per window.
    window_kernel = onp.ones(WINDOW_SIZE, dtype=int)
    window_near_counts = onp.convolve(near_mask.astype(int), window_kernel, mode="valid")
    peak_near_count = int(window_near_counts.max()) if window_near_counts.size > 0 else 0

    # Business rule: trigger alert when any window exceeds allowed density.
    structuring_alert = peak_near_count > max_near_in_any_window

    # Return both private witness and public claims consumed by the circuit.
    return {
        "cash_transactions": cash_transactions,
        "reporting_threshold": reporting_threshold,
        "proximity_margin": proximity_margin,
        "max_near_in_any_window": max_near_in_any_window,
        "structuring_alert": structuring_alert,
        "disclosed_peak_near_count": peak_near_count,
    }


EXAMPLE_INPUT = generate_aml_structuring_input(
    cash_transactions=[
        2400,
        8700,
        9200,
        9800,
        4300,
        9500,
        9600,
        9700,
        1800,
        2100,
        9900,
        6100,
    ],
)


@zk_circuit
def aml_structuring_detection_proof(
    cash_transactions: Private[NDArray[Integer, 12]],
    reporting_threshold: Public[Integer],
    proximity_margin: Public[Integer],
    max_near_in_any_window: Public[Integer],
    structuring_alert: Public[Boolean],
    disclosed_peak_near_count: Public[Integer],
):
    # Compute the lower bound once inside the circuit so all subsequent comparisons
    # are constrained against the same public policy parameterization.
    lower_bound = reporting_threshold - proximity_margin
    # Zinnia feature: fixed-size NDArray allocation inside circuit logic.
    window_near_counts = np.zeros((9,), dtype=int)

    # Iterate over each 4-transaction window and count "near" transactions.
    # This demonstrates loop lowering into arithmetic constraints.
    for start in range(0, 9):
        window = cash_transactions[start:start + 4]
        # Chained comparison over NDArrays produces elementwise Boolean flags.
        near_flags = lower_bound <= window < reporting_threshold
        window_near_counts[start] = near_flags.sum()

    # Aggregate to the peak suspicious density.
    peak_near_count = window_near_counts.max()

    # Public disclosure integrity: reported peak must equal constrained result.
    assert disclosed_peak_near_count == peak_near_count
    # Public alert integrity: alert bit must match policy threshold decision.
    assert structuring_alert == (peak_near_count > max_near_in_any_window)


def run_aml_structuring_detection_proof(
    data: dict[str, Any] | None = None,
    circuit_name: str = "aml_structuring_detection_proof",
    k: int = 16,
):
    # Keep runnable defaults for docs and CI-style local demos.
    if data is None:
        data = EXAMPLE_INPUT

    # Copy input to avoid accidental mutation across proving phases.
    proving_data = dict(data)

    # Typical Zinnia execution pipeline: build circuit -> keygen -> prove -> verify.
    circuit = create_circuit(aml_structuring_detection_proof, chips=[])
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
    run_aml_structuring_detection_proof(
        data=EXAMPLE_INPUT,
        circuit_name="aml_structuring_detection_proof",
        k=16,
    )
