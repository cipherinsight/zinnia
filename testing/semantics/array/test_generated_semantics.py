from __future__ import annotations

import pytest

from zinnia import ZKCircuit

from .generator import GeneratedCase, generate_cases, oracle, render_source
from .manifest import ExpectedStatus
from .reporting import ReportRow, record


CASES = generate_cases()


def _param(case: GeneratedCase):
    marks = []
    if case.effective_expected == ExpectedStatus.XFAIL:
        marks.append(pytest.mark.xfail(reason=case.shape.reason or case.operator.reason or "manifest xfail", strict=False))
    return pytest.param(case, id=case.id, marks=marks)


@pytest.mark.parametrize("case", [_param(case) for case in CASES])
def test_array_semantics(case: GeneratedCase):
    if case.effective_expected == ExpectedStatus.UNCLEAR:
        _record(case, "unclear", None)
        pytest.skip(case.operator.reason or "manifest marks this contract as unclear")

    try:
        if case.effective_expected == ExpectedStatus.REJECT:
            _assert_rejected(case)
            _record(case, "reject", None)
            return

        expected = oracle(case)
        source = render_source(case, expected)
        proof = ZKCircuit.from_source("semantic_case", source).prove(
            *_case_prove_args(case),
            backend="mock",
        )
        assert proof.proof_bytes_hex == "mock_satisfied"
    except BaseException as exc:
        if case.effective_expected == ExpectedStatus.XFAIL:
            _record(case, "xfail", type(exc).__name__)
        else:
            _record(case, "unexpected_failure", type(exc).__name__)
        raise

    _record(case, "pass", None)


def _assert_rejected(case: GeneratedCase) -> None:
    source = "\n".join(
        [
            "def semantic_case():",
            *[f"    {line}" for line in _reject_operation_lines(case)],
            "    assert out.sum() == 0",
        ]
    )
    try:
        proof = ZKCircuit.from_source("semantic_case", source).prove(backend="mock")
    except Exception:
        return
    assert proof.proof_bytes_hex != "mock_satisfied"


def _reject_operation_lines(case: GeneratedCase) -> list[str]:
    from .generator import _render_operation

    return _render_operation(case)


def _case_prove_args(case: GeneratedCase) -> list[object]:
    return list(case.shape.data.get("prove_args", []))


def _record(case: GeneratedCase, actual_status: str, failure_class: str | None) -> None:
    record(
        ReportRow(
            case_id=case.id,
            operator=case.operator.name,
            kind=case.operator.kind,
            spelling=case.spelling,
            mode=case.mode.value,
            dtype=case.dtype,
            rank=case.rank,
            shape_pattern=case.shape.pattern,
            expected_status=case.effective_expected.value,
            actual_status=actual_status,
            failure_class=failure_class,
            manifest_reason=case.shape.reason or case.operator.reason,
            replaces=case.shape.replaces,
        )
    )
