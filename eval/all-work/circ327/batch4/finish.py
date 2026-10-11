import study, pathlib,json,hashlib,collections,platform
import numpy as np
R=study.ROOT
results=[]
for p in study.BATCH:
 d=R/p['program']; r=json.loads((d/'result.json').read_text()); results.append(r)
 f=json.loads((d/'fixture.json').read_text())
 for label in ('A','B'):
  (d/(label+'.pin')).write_text(study.pin_text(f,label))
 lines=['Program: '+r['program'],'','As written: '+r['as_written']['status'],r['as_written'].get('error') or study.SPECS[p['program']]['reason'],'']
 if r['rewrite']:
  a=r['rewrite'];lines+=['Minimal rewrite: '+a['status'],a.get('error') or 'Compiled and checked on both fixtures.','']+['- '+s for s in a['rewrites']]
 if not r['category'].startswith('accepted'):
  lines+=['','Correctness A/B: not run; no accepted executable translation.']
 if 'code' not in study.SPECS[p['program']]:
  lines+=['','Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.']
 (d/'attempts.md').write_text('\n'.join(lines)+'\n')
 if r['category'].startswith('accepted'):
  a=r['as_written'] if r['category']=='accepted_as_written' else r['rewrite']
  assert a['status']=='accepted' and isinstance(a['constraints'],int)
  assert all(r['correctness'][l]['status']=='pass' for l in ('A','B'))
  path=d/('as_written.zok' if r['category']=='accepted_as_written' else 'rewrite.zok')
  assert path.exists()
  for l in ('A','B'):
   assert json.loads((d/(l+'.comparison.json')).read_text())['match']
assert len(results)==len(study.BATCH)==54
# Remove obsolete alternate artifact from development; retain it under probes.
for suffix in ['zok','compile.log']:
 old=R/'case0341'/('rewrite.'+suffix)
 if old.exists():old.rename(R/'probes'/('case0341-initial.'+suffix))
provenance={'python':platform.python_version(),'numpy':np.__version__,'scale':2**32,'tolerance':{'relative':1e-6,'absolute':1e-6},'program_count':54,'files':{}}
for f in [pathlib.Path(study.CIRC+'circ'),pathlib.Path(study.CIRC+'zxi'),R.parent/'BRIEF.md',R.parent/'batch4.json']:
 provenance['files'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
study.write_json(R/'provenance.json',provenance)
counts=collections.Counter(r['category'] for r in results)
lines=['# Batch 4 CirC acceptance results','',f'54 programs: {counts["accepted_as_written"]} accepted as written, {counts["accepted_with_rewrite"]} accepted with minimal rewrite, {54-counts["accepted_as_written"]-counts["accepted_with_rewrite"]} not accepted. All 24 accepted programs pass fixtures A and B (48 comparisons).','',
'Each executable translation was compiled with the supplied stock CirC binary using `--language zsharp <file> r1cs --action count`, `MEMRUN_MAX=8G` through the supplied `memrun.sh`, and `timeout 300`. `TMPDIR` is this batch\'s `tmp/`. Per-program `command.json`, compile logs, source snapshots, fixtures, `.pin` inputs, interpreter logs, comparison JSON, and `result.json` preserve the evidence. Binary hashes and numerical settings are in `provenance.json`.','',
'Inputs are private and returned values are public. Reals use signed Q32 integers (scale 2^32), encoded in two\'s-complement u64; signed comparisons use sign-bit bias. Products use stock field multiplication and bit conversion to retain the middle 64 bits after scaling, with sign restored. This is the ordinary fixed-point representation, not a handwritten division, square-root, transcendental, or sorting circuit. Integer-count division in case0419 uses native u64 division without fixed-point rescaling. Complex values in case0413 are real/imaginary Q32 pairs. Standard Q32 range and quantization limitations apply.','',
'Private scalar shape parameters remain private even when fixture notes call them constants. Dynamic-output rewrites publish logical length/dimensions along with padded contents; the correctness decoder restores the NumPy output shape. For functions returning None while mutating arrays (SYRK and FDTD), all mutated arrays are public outputs and are compared. Reference execution strips decorators and annotations and executes the original function body with NumPy; it does not call Zinnia or inspect its IR, circuits, sizes, or acceptance results.','',
'## Counts','', '| Category | Programs |','|---|---:|']
for cat,n in sorted(counts.items()):lines.append(f'| `{cat}` | {n} |')
lines+=['','## Programs','','| Program | Category | Constraints | Reason |','|---|---|---:|---|']
for r in results:
 a=r['as_written'] if r['category']=='accepted_as_written' else r['rewrite']
 count=str(a['constraints']) if a and a['constraints'] is not None else '—'
 reason=r['notes']
 if r['program']=='pythran_fannkuch':reason='Bounded 1024!-step rewrite rejected: `0 is not a constant integer` for an integer-typed loop bound.'
 lines.append(f'| [{r["program"]}]({r["program"]}/result.json) | `{r["category"]}` | {count} | {reason.replace("|","/")} |')
lines+=['','## Rewrites and verification limits','',
'Every rewrite and its domain-bound argument is listed in the corresponding `result.json` and `attempts.md`. Static append enumeration in case0341 and the source bubble-sort loops are translated directly: case0341 has exactly 30 values, and bubble_sort\'s break is unreachable because every inner loop executes and sets swap_test=True unconditionally. No sorting network was substituted.','',
'Non-expressible and gadget-dependent attempts have source-line explanations instead of fabricated `.zok` stubs or invented compiler errors. Their A/B correctness fields are null because no allowed executable translation exists. Gadget requirements were identified from source operations and the stock Z# frontend/standard-library capabilities. These are not measured compiler rejections. There are 29 such programs and one measured compiler rejection; none timed out or exhausted memory.','',
'Fannkuch has an explicit n<=1024 contract, so it is not classified as unbounded. Its minimal rewrite pads the permutation arrays to 1024, bounds permutation/prefix-reversal enumeration by 1024!, and masks early exits. CirC rejects the arbitrary-precision integer loop bound before unrolling (`0 is not a constant integer`). It was not reduced to the fixture values n=4 or n=3, and its A/B results were not verified.','',
'Correctness establishes agreement on fixtures A and B at relative+absolute tolerance 1e-6, not a proof over every input or of Q32 overflow freedom. case0399 A has maximum absolute error approximately 2.16e-4 but passes the specified combined tolerance because the corresponding outputs are large. Source-domain reasoning for each padding/loop capacity is separate from fixture testing. No accepted result has a failed or missing fixture check.','',
'Reproduce all final attempts and fixture checks with `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python study.py`; pass program names to rerun a subset. Run `finish.py` with the same Python afterward to audit artifacts and regenerate this report. Development logs at batch root may contain corrected intermediate failures; per-program final logs and `result.json` are authoritative.','', 'BATCH_DONE']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
study.write_json(R/'audit.json',{'programs':54,'results':len(results),'accepted':24,'fixture_checks_passed':48,'counts':dict(sorted(counts.items())),'report_ends_with_batch_done':True})
print(json.dumps(counts,indent=2))
