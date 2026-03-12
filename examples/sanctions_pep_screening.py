"""
Sanctions and PEP Screening Example
-----------------------------------

This circuit models compliance screening against sanctions and PEP watchlists.
It keeps the customer identifier private while proving that disclosed risk and
clearance outcomes are consistent with deterministic matching rules.
"""


from zinnia import *

from typing import Any

from utils import create_circuit, create_prove_verify_keys, prove, verify


WATCHLIST_SIZE = 8


def generate_sanctions_pep_input(
    sanctions_ids: list[int],
    pep_ids: list[int],
    customer_id: int,
) -> dict[str, Any]:
    # Enforce bounded watchlists so circuit dimensions remain fixed.
    if len(sanctions_ids) > WATCHLIST_SIZE:
        raise ValueError(f"sanctions_ids exceeds watchlist size {WATCHLIST_SIZE}.")
    if len(pep_ids) > WATCHLIST_SIZE:
        raise ValueError(f"pep_ids exceeds watchlist size {WATCHLIST_SIZE}.")

    # Pad with sentinel values to produce fixed-length NDArrays for public inputs.
    sanctions_watchlist = sanctions_ids + [-1] * (WATCHLIST_SIZE - len(sanctions_ids))
    pep_watchlist = pep_ids + [-1] * (WATCHLIST_SIZE - len(pep_ids))

    # Off-circuit reference matching used to prepare expected public claims.
    sanctions_match = customer_id in sanctions_ids
    pep_match = customer_id in pep_ids

    # Policy semantics:
    # - sanctions hit => highest risk, not cleared
    # - PEP-only hit => medium risk, cleared
    # - no hit => low risk, cleared
    if sanctions_match:
        disclosed_risk_tier = 2
        is_cleared = False
    elif pep_match:
        disclosed_risk_tier = 1
        is_cleared = True
    else:
        disclosed_risk_tier = 0
        is_cleared = True

    # Return private witness plus all public claims checked in-circuit.
    return {
        "customer_id": customer_id,
        "sanctions_watchlist": sanctions_watchlist,
        "sanctions_match": sanctions_match,
        "pep_watchlist": pep_watchlist,
        "pep_match": pep_match,
        "disclosed_risk_tier": disclosed_risk_tier,
        "is_cleared": is_cleared,
    }


EXAMPLE_INPUT = generate_sanctions_pep_input(
    sanctions_ids=[101, 303, 505, 777],
    pep_ids=[202, 404, 606, 808],
    customer_id=202,
)


@zk_circuit
def sanctions_pep_screening(
    customer_id: Private[Integer],
    sanctions_watchlist: Public[NDArray[Integer, 8]],
    sanctions_match: Public[Boolean],
    pep_watchlist: Public[NDArray[Integer, 8]],
    pep_match: Public[Boolean],
    disclosed_risk_tier: Public[Integer],
    is_cleared: Public[Boolean],
):
    # Count matches explicitly to demonstrate loop + conditional constraints.
    sanctions_hits = 0
    pep_hits = 0

    for i in range(0, 8):
        if sanctions_watchlist[i] == customer_id:
            sanctions_hits += 1
        if pep_watchlist[i] == customer_id:
            pep_hits += 1

    # This example assumes no duplicate identifiers per list.
    assert sanctions_hits <= 1
    assert pep_hits <= 1

    # Derive boolean match flags from constrained hit counters.
    computed_sanctions_match = sanctions_hits == 1
    computed_pep_match = pep_hits == 1

    # Public match disclosures must be honest.
    assert sanctions_match == computed_sanctions_match
    assert pep_match == computed_pep_match

    # Enforce risk/clearance policy consistency from match outcomes.
    if sanctions_match:
        assert disclosed_risk_tier == 2
        assert is_cleared == False
    else:
        assert is_cleared == True
        if pep_match:
            assert disclosed_risk_tier == 1
        else:
            assert disclosed_risk_tier == 0


def run_sanctions_pep_screening(
    data: dict[str, Any] | None = None,
    circuit_name: str = "sanctions_pep_screening",
    k: int = 16,
):
    # Use bundled fixture when no external data is provided.
    if data is None:
        data = EXAMPLE_INPUT

    # Duplicate payload for safety across helper calls.
    proving_data = dict(data)

    # Standard compile/keygen/prove/verify path shared across examples.
    circuit = create_circuit(sanctions_pep_screening, chips=[])
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
    run_sanctions_pep_screening(
        data=EXAMPLE_INPUT,
        circuit_name="sanctions_pep_screening",
        k=16,
    )
