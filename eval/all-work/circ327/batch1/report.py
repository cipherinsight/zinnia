import collections,json,pathlib,re
BASE=pathlib.Path(__file__).resolve().parent
M=json.load(open(BASE/'manifest.json'))
REASONS={
'case0301':'Twenty values plus public row/column metadata preserve the dynamic reshape.',
'case0311':'Scan for first argmax; unravel its flat index in Fortran order.',
'case0319':'Native private-index scatter and gather preserve the permutation.',
'case0338':'Static diagonal extraction and stacking.',
'case0354':'Pad symmetric row difference to nine rows and expose logical length.',
'case0360':'Static deletion of row 2.',
'case0386':'Static expansion of all 19,701 windows; 78,804 public values.',
'case0392':'Normalize private slice endpoints; pad to twenty rows with length.',
'case0402':'Fortran-order ravel via private dimension products.',
'case0409':'Elementwise fixed-point addition.',
'case0444':'Pad Boolean selection to twenty entries and expose length.',
'case0463':'Pad selected nonzero run starts to twenty entries and expose length.',
'case0478':'Preserve diagonal-view aliasing when zeroing the array.',
'case0504':'Native private gather along the last axis, followed by sum.',
'pythran_ccopy':'Guard 1,024 copy iterations from declared N bound; return mutated CY.',
'pythran_insertion_sort':'Bound source insertion-sort while by nine guarded shifts.',
'pythran_multitype':'Multiplication with declared 0..1,024 scalar domains.',
'pythran_perm':'Preserve cycles algorithm; exactly 4! outputs from a four-element input.',
'pythran_pythagorean_triples':'Static integer matrix multiplication, transpose and reshape.',
'npbench_crc16_crc16':'Native Integer bitwise rejected; unsigned narrowing was discarded.',
'case0489':'Masked 100x50 rewrite killed under the 8 GB cap.',
'pythran_collatz':'1022 slots and 178 proven rounds; killed under the 8 GB cap.',
'pythran_growcut':'1024x1024 capacity and masked neighborhoods; killed under the 8 GB cap.',
'pythran_laplace':'1024x1024 grid and 1024 time steps; killed under the 8 GB cap.'}
results=[]
for p,m in M.items():
 f=BASE/p/'result.json';r=json.load(open(f))
 assert r['program']==p
 assert r['category'] not in ['pending_correctness','wrong'],(p,r['category'])
 if r['category'].startswith('accepted'):
  assert all(isinstance(r['correctness'][k],dict) and r['correctness'][k]['status']=='pass' for k in ['A','B'])
  assert r[r['accepted_attempt']]['status']=='accepted'
 reason=REASONS.get(p,m.get('reason',''))
 if r['category']=='needs_gadget: fixed-point multiplication':reason='Native signed fixed-point arithmetic reaches R1CS, then crashes; a wide signed rescaling gadget is forbidden.'
 r['notes']=[x for x in r['notes'] if x]
 if reason not in r['notes']:r['notes'].append(reason)
 if r['category'].startswith('accepted'):
  note='A/B checked with zxi against undecorated original NumPy source; rtol=atol=1e-6. Reals use scale 2^32; signed field values use centered residues.'
  if note not in r['notes']:r['notes'].append(note)
 r['source']={'kernel':str(BASE/p/'kernel.py'),'function':next(x['function'] for x in json.load(open(BASE.parent/'batch1.json')) if x['program']==p)}
 if m.get('line'):r['source']['line']=m['line']
 f.write_text(json.dumps(r,indent=2)+'\n');results.append((p,r,reason))
counts=collections.Counter(r['category'] for _,r,_ in results)
compiled=sum(any('file' in m.get(a,{}) for a in ['as_written','rewrite']) for m in M.values())
lines=['# CirC acceptance study — batch1','',
'All 55 programs have a `result.json`. **19/55 accepted (34.5%)**: 11 as written and 8 with minimal rewrites. Every accepted program passed fixtures A and B after decoding.','',
'## Method and reproducibility','',
'Only original kernel source, signatures, requirements, and fixtures were used for translation decisions. No Zinnia IR, circuits, sizes, or acceptance outcomes were used. Source and fixtures are copied into each program directory. Multi-function fixtures use `fixture.<function>.json` from the source directory.','',
'All 32 executable translations were actually compiled with the supplied CirC binary using `--language zsharp <file> r1cs --action count`, wrapped by `memrun.sh`, `MEMRUN_MAX=8G`, and `timeout 300`. Each invocation has a `.command.json` and `.log`. `TMPDIR` is local to this batch. Binary hashes and numeric configuration are in `environment.json`. The remaining 23 programs have source-level non-expressibility or required-gadget assessments, not invented compiler failures; no executable stub was compiled for them.','',
'Reals use integer fixed point with scale 2^32. Simple copying, addition, selection, and integer polynomial arithmetic use field values with centered signed decoding. Signed comparisons explicitly handle the centered representation. Native Z# `integer` was tried for signed wide fixed-point multiplication/rescaling; the R1CS backend rejected it. Those eight cases are classified as requiring a fixed-point multiplication/rescaling gadget, which the brief forbids writing. They are not counted as accepted merely because the frontend or interpreter supports integers. Field division was not substituted for fixed-point quotient: the reshape metadata division is exact on valid reshape inputs.','',
'All input parameters remain private, including scalar shape parameters even where fixture notes call them constants. Returns are public. For procedures returning None, modified arrays are exposed and checked (CCOPY: CY; insertion sort: list2). Unsuccessful GEMVER/K2MM/spectralDNS translations likewise retain their observable modified arrays. Complex values are real/imaginary pairs. Padded outputs include logical length or dimensions; the decoder removes only padding and verifies reshape dimensions.','',
'Run from this directory with `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python run.py`, then the same Python with `check.py`, then `report.py`. The checked-in `.zok` files and `manifest.json` are the authoritative final translations; `build.py`, `build_more.py`, and `normalize.py` are development generators, not the replay path. `run.py <program>...` and `check.py <program>...` support individual replay.','',
'## Counts','',
'| Category | Programs |','|---|---:|']
for category,n in sorted(counts.items()):lines.append(f'| `{category}` | {n} |')
lines+=['| **Total** | **55** |','','## Per-program results','','Constraint counts are for the accepted attempt. A dash means no accepted R1CS; full diagnostics and rewrites are in the corresponding `result.json`.','','| Program | Category | Constraints | Reason |','|---|---|---:|---|']
for p,r,reason in results:
 a=r.get('accepted_attempt');c=r[a]['constraints'] if a and r['category'].startswith('accepted') else '—'
 lines.append(f'| [{p}]({p}/result.json) | `{r["category"]}` | {c} | {reason.replace(chr(10)," ").replace("|","/")} |')
lines+=['','## Limits and verification gaps','',
'- A/B checks establish fixture correctness, not an exhaustive proof over every numeric input; fixed-point quantization and the finite field representation remain numerical limits of the translation. No observed fixture range was used to invent array or loop bounds.',
'- The 36 programs without an accepted attempt have no completed A/B equivalence check. Their classifications distinguish source-level impossibility/gadget requirements from actual compilation failure. Four capped runs ended with signal 9: masked update, Collatz, GrowCut, and Laplace; the scope memory cap was 8 GB. None is counted as accepted.',
'- Collatz has a domain-derived bound: exhaustive integer simulation of every initial value 3..1024 gives at most 178 transitions and maximum intermediate value 250504. The proof data are in `pythran_collatz/bound.json`; this does not use fixture values.',
'- The large sliding-window interpreter initially exceeded the cap with repeated output-array mutation. Expanding the same static window expressions into an array literal preserved every output and allowed both fixtures to complete; the final source, compilation log, and checks correspond to that representation.',
'- CRC16 was not credited for an exploratory u64 translation: although it passed A/B, unconstrained signed Integer poly was narrowed. The final native-Integer attempt reports the actual bitwise-operand rejection. The discarded investigation is explicitly separated under `npbench_crc16_crc16/discarded_u64/`.',
'- Fixed-point division, transcendental functions, and library sorting were not replaced with handwritten gadgets. FFT twiddles were evaluated only as compile-time constants; its private fixed-point arithmetic still failed in R1CS. GrowCut’s zero color distance was folded only because image is initialized to zero and never modified for any input.',
'','BATCH_DONE']
(BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'programs':len(results),'compiled_programs':compiled,'categories':counts},indent=2))
