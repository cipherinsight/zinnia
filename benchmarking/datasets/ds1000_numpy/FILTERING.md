# DS-1000 NumPy Scope Inventory

Generated from the original DS-1000 JSONL metadata and reference solutions.

- Source: `https://raw.githubusercontent.com/xlang-ai/DS-1000/main/data/ds1000.jsonl.gz`
- Original NumPy slice: `220` programs
- Acceptance-study candidates: `162`
- Out-of-scope before acceptance study: `58`

## Filtering Policy

The acceptance study should start from all DS-1000 programs labeled `Numpy`.
A program is filtered out only when it is structurally outside Zinnia's
intended benchmark claim: deterministic, in-memory, numeric NumPy-style
array computation that can be considered for static or bounded circuit
synthesis.

Do **not** filter a program merely because Zinnia currently lacks an
operator, shape rule, dynamic-output path, or solver fact. Those cases must
remain in the acceptance study and be counted as rejection/coverage failures.

Out-of-scope categories:

- `exception_control_flow`: try/except/finally control flow.
- `external_io`: file, path, network, stdin/stdout, or NumPy file-I/O dependence.
- `nondeterminism`: randomness, current time, or process/environment dependence in the submitted solution.
- `nonfinite_float_semantics`: NaN or Inf values / predicates / propagation semantics.
- `non_numeric_data_model`: string/bytes/object/datetime/structured-record data model.
- `string_operations`: string parsing or string-method operations.
- `excluded_numpy_family`: np.linalg, np.lib, np.polyfit, and their NumPy-qualified variants.
- `external_library_runtime`: runtime dependence on non-NumPy data-science libraries.
- `opaque_python_runtime`: eval/exec/reflection/import machinery or dynamic code loading.
- `interactive_or_plotting`: interactive display, plotting, widgets, or visualization output.
- `unbounded_runtime_behavior`: data-dependent unbounded while/generator behavior not statically bounded by input shape.

## Counts

| Status | Count |
| --- | ---: |
| `acceptance_candidate` | `162` |
| `out_of_scope` | `58` |

| Out-of-scope reason | Count |
| --- | ---: |
| `external_library_runtime` | `29` |
| `nonfinite_float_semantics` | `14` |
| `excluded_numpy_family` | `9` |
| `nondeterminism` | `3` |
| `string_operations` | `3` |
| `exception_control_flow` | `1` |

## Acceptance Candidate IDs

`291`, `295`, `296`, `297`, `298`, `299`, `300`, `301`, `302`, `303`, `304`, `309`, `310`, `311`, `312`, `313`, `314`, `317`, `318`, `319`, `320`, `321`, `322`, `323`, `324`, `325`, `326`, `327`, `328`, `329`, `330`, `331`, `332`, `334`, `335`, `336`, `337`, `338`, `339`, `340`, `341`, `342`, `343`, `344`, `345`, `346`, `347`, `349`, `353`, `354`, `355`, `356`, `357`, `358`, `359`, `360`, `361`, `362`, `363`, `364`, `365`, `366`, `367`, `368`, `369`, `370`, `373`, `374`, `375`, `381`, `382`, `383`, `385`, `387`, `388`, `389`, `390`, `391`, `392`, `393`, `398`, `399`, `400`, `401`, `402`, `403`, `405`, `406`, `407`, `408`, `409`, `411`, `412`, `413`, `414`, `415`, `416`, `417`, `418`, `419`, `420`, `422`, `425`, `426`, `427`, `428`, `429`, `430`, `431`, `433`, `434`, `435`, `436`, `437`, `438`, `439`, `440`, `441`, `442`, `443`, `444`, `448`, `449`, `450`, `451`, `452`, `453`, `454`, `459`, `460`, `462`, `463`, `467`, `468`, `469`, `470`, `471`, `472`, `473`, `478`, `480`, `481`, `485`, `489`, `490`, `491`, `492`, `495`, `496`, `497`, `498`, `499`, `500`, `501`, `502`, `503`, `504`, `505`, `507`, `508`, `509`, `510`

## Out-of-Scope IDs

- `case0292`: `nonfinite_float_semantics`
- `case0293`: `nonfinite_float_semantics`
- `case0294`: `nonfinite_float_semantics`
- `case0305`: `nonfinite_float_semantics`
- `case0306`: `nonfinite_float_semantics`
- `case0307`: `nonfinite_float_semantics`
- `case0308`: `nondeterminism`
- `case0315`: `nonfinite_float_semantics`
- `case0316`: `nonfinite_float_semantics`
- `case0333`: `nonfinite_float_semantics`
- `case0348`: `excluded_numpy_family`
- `case0350`: `external_library_runtime`
- `case0351`: `external_library_runtime`
- `case0352`: `external_library_runtime`
- `case0371`: `external_library_runtime`
- `case0372`: `external_library_runtime`
- `case0376`: `nondeterminism`
- `case0377`: `external_library_runtime`
- `case0378`: `external_library_runtime`
- `case0379`: `external_library_runtime`
- `case0380`: `external_library_runtime`
- `case0384`: `excluded_numpy_family`
- `case0386`: `excluded_numpy_family`
- `case0394`: `string_operations`
- `case0395`: `external_library_runtime`
- `case0396`: `external_library_runtime`
- `case0397`: `external_library_runtime`
- `case0404`: `external_library_runtime`
- `case0410`: `nondeterminism`
- `case0421`: `external_library_runtime`
- `case0423`: `external_library_runtime`
- `case0424`: `external_library_runtime`
- `case0432`: `nonfinite_float_semantics`
- `case0445`: `external_library_runtime`
- `case0446`: `external_library_runtime`
- `case0447`: `external_library_runtime`
- `case0455`: `nonfinite_float_semantics`, `string_operations`
- `case0456`: `excluded_numpy_family`
- `case0457`: `excluded_numpy_family`
- `case0458`: `excluded_numpy_family`
- `case0461`: `string_operations`
- `case0464`: `external_library_runtime`
- `case0465`: `external_library_runtime`
- `case0466`: `external_library_runtime`
- `case0474`: `nonfinite_float_semantics`
- `case0475`: `external_library_runtime`
- `case0476`: `external_library_runtime`
- `case0477`: `excluded_numpy_family`
- `case0479`: `external_library_runtime`
- `case0482`: `excluded_numpy_family`
- `case0483`: `excluded_numpy_family`
- `case0484`: `external_library_runtime`
- `case0486`: `external_library_runtime`
- `case0487`: `external_library_runtime`
- `case0488`: `external_library_runtime`
- `case0493`: `exception_control_flow`
- `case0494`: `nonfinite_float_semantics`
- `case0506`: `nonfinite_float_semantics`

## Files

- `manifest.json`: full metadata, original prompt/context, reference code, status, and reasons.
- `caseNNNN/reference.py`: original DS-1000 reference solution text only.
- `caseNNNN/record.json`: per-case metadata copied from the manifest.
