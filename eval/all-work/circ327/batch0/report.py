import json,pathlib,collections
R=pathlib.Path(__file__).resolve().parent
B=json.load(open(R.parent/'batch0.json'))
results=[]
for p in B:
 d=json.load(open(R/p['program']/'result.json'))
 if d['category'].startswith('accepted'):
  assert all(d['correctness'][k]['status']=='pass' for k in ['A','B']),p['program']
 results.append(d)
counts=collections.Counter(d['category'] for d in results)
rows=[]
for d in results:
 attempt=d['as_written'] if d['category']=='accepted_as_written' else d.get('rewrite') or d['as_written']
 count=attempt.get('constraints')
 if d['category'].startswith('accepted'):
  reason='A/B pass. '+ ('; '.join(attempt.get('rewrites',[])) or d['notes'] or 'Static-shape translation using native field operations.')
 else:reason=attempt.get('error') or d['notes']
 reason=' '.join(reason.split()).replace('|','\\|')
 rows.append(f"| {d['program']} | {d['category']} | {count if count is not None else '—'} | {reason} |")
accepted=sum(v for k,v in counts.items() if k.startswith('accepted'))
text=f'''# CirC acceptance study — batch0

Completed {len(results)} programs: **{accepted} accepted**, all checked against both NumPy fixtures at relative + absolute tolerance 1e-6. No accepted program remains unchecked or mismatching.

## Method

Read BRIEF.md before any program work. Used only kernel source, signatures, preconditions and fixtures to choose translations; did not inspect Zinnia IR, circuits, sizes or use its outcomes. All work and temporary files are inside this batch directory.

Compiler: `/home/zhantong/.cache/zkcc-reconfirm/circ-stock/target/release/examples/circ`, language `zsharp`, backend `r1cs --action count`. Every executable attempt ran through the requested `memrun.sh` with `MEMRUN_MAX=8G`, `MemorySwapMax=0`, and `timeout 300`. `ZSHARP_STDLIB_PATH` selects the installed ZoKrates stdlib; `TMPDIR` is this batch's `tmp/`. Commands, exit statuses, durations and compiler diagnostics are in each result/log. No build or remote machine was used.

Inputs are private and return values public. Reals are rounded to integer multiples of 2^-32. Copies, indexing, linear operations and comparisons use centered signed field encoding; signed comparisons shift both operands by half the field modulus. Native `integer` arithmetic was tried for signed Q32 products and quotients. Those full translations reached CirC's R1CS backend and failed on integer operators. Modular field inversion was not substituted for fixed-point numerical division, nor were values narrowed to unsigned or fixture-only ranges. The missing signed fixed-point quotient/rescaling route is recorded as `needs_gadget: division` under the brief's precedence. No handwritten sort, search, division, square-root, transcendental or arithmetic circuit gadget was added.

For non-expressible shapes/loops and missing gadgets, `.zok` files are explicit explanatory records, **not executable stubs**. These cases have no compilation count; the result names the source line and why no permitted full translation can be emitted. Bounded rewrites that still need a forbidden gadget are documented but not presented as compiled programs. In particular, the study does not claim a compiler run for those exceptions.

Padded outputs include public lengths/dimensions (and zero padding); correctness decodes that representation back to the original output before comparison. Constant-output programs retain their original private input signatures. In-place kernels preserve their mutated arrays as observable outputs. The empty 3x0 array is represented by padding and public shape after CirC rejected a native empty array. For `case0385`, the static view permutation is expanded directly into an array constructor, avoiding the interpreter's memory blowup on sequential array stores; all 20,000 elements and the 5000x2x2 result shape are retained. For `pythran_emin`, the original million iterations are retained; the rewrite only uses the built-in integer-to-field cast on constant outputs.

## Counts

| Category | Programs |
| --- | ---: |
'''+''.join(f'| {k} | {v} |\n' for k,v in sorted(counts.items()))+f'| **Total** | **{len(results)}** |\n'+'''
## Per-program results

| Program | Category | Constraints | Reason |
| --- | --- | ---: | --- |
'''+ '\n'.join(rows)+'''

## Verification and reproduction

Each accepted program has fixture copies, A/B `.pin` inputs, interpreter logs, NumPy expected outputs, decoded actual outputs and explicit comparison statuses in `result.json`. NumPy references run the original function bodies after stripping only Zinnia decorators and annotations; no Zinnia compilation occurs. `np.in1d` uses `np.isin` on NumPy versions that removed the former. `compare.py` checks structure and values, not just flattened numbers. Equal NaNs are allowed by the comparator, though no accepted fixture depends on a NaN route.

From this directory, use `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python reproduce.py` to rerun the final executable translations and both attempts where executable, regenerate the exception records, compare accepted results, and rebuild this report. Run scripts do not rebuild or edit CirC. `build.py`, `extend.py` and `fixes.py` retain translation construction history; the final per-program `.zok` files are authoritative and reproduction does not overwrite them. Some `.loop_expansion.zok` files and `original_attempt.json` files document earlier translation/resource diagnostics.

No numerical correctness claim is made for programs without an accepted executable attempt. Missing-gadget conclusions rely on the documented native frontend/stdlib capabilities and, for native signed-integer arithmetic, the saved actual backend rejection. `not_expressible` conclusions concern the declared domain, not the two fixture values. A finite input array does not bound arbitrary requested padding or time-step counts. Resource failures are measured under the specified per-process limits; a SIGKILL/137 under the memory-capped scope is classified as out of memory.

BATCH_DONE
'''
(R/'REPORT.md').write_text(text)
print('Wrote REPORT.md:',len(results),'programs,',accepted,'accepted')
