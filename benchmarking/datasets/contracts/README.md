# Superseded contract drafts

These files are the 2026-05 machine-drafted contracts (`"review_status": "draft"`)
for the NPBench and Pythran kernels. Their `proposed_annotation` fields were
never reviewed and 43 programs had annotations that contradicted upstream
(audit DB-10). They are no longer used: the kernels' annotations are derived
from upstream by `tools/acceptance_case_set.py`, and
`case_sets/acceptance.json` records the derivation, the previous annotation
and the preconditions per entry function. The files stay only because
`benchmarking/render_contract_review.py` still reads them; delete them when
that script is retired.
