"""
Proof of Solvency Example
-------------------------

This example proves aggregate solvency from private asset/liability buckets. The
circuit discloses only totals plus a coarse reserve band and pass/fail status,
allowing external stakeholders to validate capital adequacy without seeing each
underlying bucket value. 
"""


from zinnia import *

from typing import Any

from examples.utils import create_circuit, create_prove_verify_keys, prove, verify


NUM_BUCKETS = 8


def generate_solvency_input(
    asset_buckets: list[int],
    liability_buckets: list[int],
    min_reserve_bps: int = 10_500,
) -> dict[str, Any]:
    # Fixed-size vectors keep circuit shape static and predictable.
    if len(asset_buckets) != NUM_BUCKETS:
        raise ValueError(f"asset_buckets must have length {NUM_BUCKETS}.")
    if len(liability_buckets) != NUM_BUCKETS:
        raise ValueError(f"liability_buckets must have length {NUM_BUCKETS}.")

    # Compute disclosed aggregates from private bucket-level data.
    total_assets = sum(asset_buckets)
    total_liabilities = sum(liability_buckets)

    # Solvency rule in basis points for integer-safe comparisons.
    solvency_pass = total_assets * 10_000 >= total_liabilities * min_reserve_bps

    # Coarse disclosure band communicates cushion strength with limited leakage.
    if total_assets * 10_000 < total_liabilities * 10_000:
        disclosed_buffer_band = 0
    elif total_assets * 10_000 < total_liabilities * 11_000:
        disclosed_buffer_band = 1
    elif total_assets * 10_000 < total_liabilities * 13_000:
        disclosed_buffer_band = 2
    else:
        disclosed_buffer_band = 3

    # Return full proving payload: private vectors + public commitments.
    return {
        "asset_buckets": asset_buckets,
        "liability_buckets": liability_buckets,
        "min_reserve_bps": min_reserve_bps,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "solvency_pass": solvency_pass,
        "disclosed_buffer_band": disclosed_buffer_band,
    }


EXAMPLE_INPUT = generate_solvency_input(
    asset_buckets=[45000, 18000, 23000, 12000, 9000, 7000, 5000, 3000],
    liability_buckets=[36000, 16000, 17000, 11000, 8000, 6500, 4200, 2500],
)


@zk_circuit
def proof_of_solvency(
    asset_buckets: Private[NDArray[Integer, 8]],
    liability_buckets: Private[NDArray[Integer, 8]],
    min_reserve_bps: Public[Integer],
    total_assets: Public[Integer],
    total_liabilities: Public[Integer],
    solvency_pass: Public[Boolean],
    disclosed_buffer_band: Public[Integer],
):
    # Rebuild aggregate totals in-circuit so published totals are cryptographically bound.
    computed_assets = 0
    computed_liabilities = 0

    # Accumulation over NDArrays in circuits
    computed_assets = asset_buckets.sum()
    computed_liabilities = liability_buckets.sum()

    # Public aggregates must exactly match constrained computations.
    assert computed_assets == total_assets
    assert computed_liabilities == total_liabilities

    # Recompute solvency decision from disclosed totals and reserve requirement.
    expected_solvency_pass = total_assets * 10_000 >= total_liabilities * min_reserve_bps
    assert solvency_pass == expected_solvency_pass

    # Enforce categorical disclosure consistency for reserve buffer strength.
    if total_assets * 10_000 < total_liabilities * 10_000:
        assert disclosed_buffer_band == 0
    else:
        if total_assets * 10_000 < total_liabilities * 11_000:
            assert disclosed_buffer_band == 1
        else:
            if total_assets * 10_000 < total_liabilities * 13_000:
                assert disclosed_buffer_band == 2
            else:
                assert disclosed_buffer_band == 3


def run_proof_of_solvency(
    data: dict[str, Any] | None = None,
    circuit_name: str = "proof_of_solvency",
    k: int = 16,
):
    # Default sample data makes this file executable out of the box.
    if data is None:
        data = EXAMPLE_INPUT

    # Defensive copy avoids unintended side effects.
    proving_data = dict(data)

    # Execute the canonical Zinnia proof lifecycle.
    circuit = create_circuit(proof_of_solvency, chips=[])
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
    run_proof_of_solvency(
        data=EXAMPLE_INPUT,
        circuit_name="proof_of_solvency",
        k=16,
    )
