# Array Semantics Tests

This suite generates deterministic pytest cases from `operators.yaml`.
Each manifest entry names an operator family, input domains, array modes, and
the expected status:

- `pass`: compile and prove with the mock backend.
- `xfail`: known bug or unsupported-but-desired behavior.
- `reject`: intentional compile-time rejection.
- `unclear`: contract is not settled; reported but not executed as correctness.

To add coverage, add a shape entry or a new operator family to
`operators.yaml`. Prefer small, distinctive shapes and values that expose
ordering and broadcasting mistakes. Avoid random inputs; this suite is meant to
be deterministic and dashboard-friendly.

Run with:

```bash
pytest testing/semantics/array
```

The test session writes a JSON summary to pytest's cache directory at
`.pytest_cache/array_semantics_report.json`.

## Migrating Manual Tests

Manifest shape entries may include a `replaces` field pointing at a manual
pytest case. Treat this as migration bookkeeping, not automatic permission to
delete the manual test. A manual test is ready to remove only when every
replacement cell for it is `expected: pass`, the generated assertions are at
least as strong as the handwritten assertions, and any old error-message or bug
pin has a matching generated `reject` or `xfail` cell.
