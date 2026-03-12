"""
Private Credit Policy Check Example
-----------------------------------

This circuit proves that a private borrower profile satisfies a lending policy
(income floor, debt-to-income bound, and credit-score floor) while revealing only
policy-level outcomes (`policy_pass` and a coarse risk band), not sensitive raw
inputs.
"""


from zinnia import *

from typing import Any

from utils import create_circuit, create_prove_verify_keys, prove, verify


def generate_credit_policy_input(
    monthly_income: int,
    monthly_debt_obligation: int,
    credit_score: int,
    min_income: int = 5000,
    max_dti_bps: int = 4000,
    min_credit_score: int = 680,
) -> dict[str, Any]:
    # Input validity checks that are mirrored conceptually in-circuit.
    if monthly_income <= 0:
        raise ValueError("monthly_income must be positive.")

    # Convert policy checks into deterministic Boolean outcomes.
    income_ok = monthly_income >= min_income
    dti_ok = monthly_debt_obligation * 10_000 <= monthly_income * max_dti_bps
    score_ok = credit_score >= min_credit_score

    # Final pass/fail gate combines all policy dimensions.
    policy_pass = income_ok and dti_ok and score_ok

    # Disclose only a coarse risk tier to limit information leakage.
    if credit_score >= 740:
        disclosed_risk_band = 2
    elif credit_score >= 680:
        disclosed_risk_band = 1
    else:
        disclosed_risk_band = 0

    # Return private witness values plus public claims expected by the circuit.
    return {
        "monthly_income": monthly_income,
        "monthly_debt_obligation": monthly_debt_obligation,
        "credit_score": credit_score,
        "min_income": min_income,
        "max_dti_bps": max_dti_bps,
        "min_credit_score": min_credit_score,
        "policy_pass": policy_pass,
        "disclosed_risk_band": disclosed_risk_band,
    }


EXAMPLE_INPUT = generate_credit_policy_input(
    monthly_income=7800,
    monthly_debt_obligation=2200,
    credit_score=712,
)


@zk_circuit
def private_credit_policy_check(
    monthly_income: Private[Integer],
    monthly_debt_obligation: Private[Integer],
    credit_score: Private[Integer],
    min_income: Public[Integer],
    max_dti_bps: Public[Integer],
    min_credit_score: Public[Integer],
    policy_pass: Public[Boolean],
    disclosed_risk_band: Public[Integer],
):
    # Sanity guard keeps division-like basis-point checks well-defined.
    assert monthly_income > 0

    # Recompute each policy primitive directly in-circuit.
    income_ok = monthly_income >= min_income
    dti_ok = monthly_debt_obligation * 10_000 <= monthly_income * max_dti_bps
    score_ok = credit_score >= min_credit_score

    # Control-flow feature demo: nested branching over booleans in Zinnia circuits.
    expected_policy_pass = False
    if income_ok:
        if dti_ok:
            if score_ok:
                expected_policy_pass = True

    # Public result must match constrained internal decision.
    assert policy_pass == expected_policy_pass

    # Enforce disclosed risk bucket consistency without revealing exact score.
    if credit_score >= 740:
        assert disclosed_risk_band == 2
    else:
        if credit_score >= 680:
            assert disclosed_risk_band == 1
        else:
            assert disclosed_risk_band == 0


def run_private_credit_policy_check(
    data: dict[str, Any] | None = None,
    circuit_name: str = "private_credit_policy_check",
    k: int = 16,
):
    # Use built-in sample payload when caller does not provide data.
    if data is None:
        data = EXAMPLE_INPUT

    # Copy for safety across keygen/prove stages.
    proving_data = dict(data)

    # Full proof lifecycle through reusable helper utilities.
    circuit = create_circuit(private_credit_policy_check, chips=[])
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
    run_private_credit_policy_check(
        data=EXAMPLE_INPUT,
        circuit_name="private_credit_policy_check",
        k=16,
    )
