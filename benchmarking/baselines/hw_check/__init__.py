"""Tools for the handwritten circuit baselines (halo2-base and Noir).

- `entries`: the cases (DS-1000 cases, NPBench/Pythran entry functions) and
  where each one's fixture and baseline files live.
- `reference`: runs a case's `kernel.py` as plain NumPy on a fixture input.
- `prover_toml`: writes a Noir `Prover.toml` from `fixture.json`.
- `noir_driver`: the Noir baseline program (setup / prove / verify, JSON out).
- `check`: checks every baseline against the NumPy reference, with keys from
  input A and proofs from input B, and that a tampered public is rejected.
- `domain_edges`: runs every baseline (circuits, zkVMs, Cairo) on inputs at
  the edge of the int64 / Q32 domain (`domain_edges.json`) and requires
  Zinnia's outcome: no proof, or a proof of NumPy's outputs.
- `lint`: static checks of the baseline files.
- `sizes`: the size constants of the NPBench/Pythran baselines (checked
  against the fixture; instantiation at another fixture's shapes).
"""
