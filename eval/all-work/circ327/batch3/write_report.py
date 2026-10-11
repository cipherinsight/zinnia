from study import *
from collections import Counter
reasons={
'case0303':'Pad at 23 elements; public reshape dimensions, private ncol retained.',
'case0313':'Native signed Q32 maximum scan; first argmax preserved.',
'case0321':'Native maximum scan returns row and column.',
'case0340':'Flatten the fixed 5x6 input in row-major order.',
'case0362':'Six-column keep mask, padded compact output and column count.',
'case0369':'Native tolerance comparison and all reduction.',
'case0398':'Original 30-step linear recurrence, integer coefficients.',
'case0412':'Ten-slot masked compaction and public length.',
'case0434':'Native dynamic row/column writes, including negative indices.',
'case0449':'Native transpose/stack indexing; 3x4x2 output retained.',
'case0468':'Sixteen padded windows; radii saturate only where slices are identical.',
'case0481':'Nine-slot compaction of matching indices and public length.',
'case0491':'Native isclose comparisons and mismatch count.',
'case0500':'Static 54-element to 18x3 reshape.',
'case0507':'Guarded 5x6 bounding-box crop with public dimensions.',
'pythran_factorize_naive':'Declared n<=1024 implies 32 steps and ten factor slots; native u32 division.',
'pythran_make_decision':'Compare squared complex distances directly; retain original first argmin.',
'case0334':'Native field remainder crashes; alternate arbitrary-integer Q32 arithmetic also rejected.',
'pythran_harris':'Native Q32 multiplication rescaling crashes; alternate arbitrary integers also rejected.',
'pythran_ln':'Native constant rescaling crashes; alternate arbitrary integers also rejected.',
'npbench_polybench_syr2k_syr2k':'Native Q32 rescaling crashes; alternate arbitrary integers also rejected.',
'case0418':'Six-bin padded rewrite reaches native field remainder backend crash.',
'npbench_spmv_spmv':'Sixteen guarded sparse terms; native Q32 rescaling crashes.',
'pythran_fibo_seq':'1024 guarded steps preserve full integer result; R1CS rejects native integer output.',
'pythran_cdotu':'1024 guarded complex multiply-add steps; native Q32 rescaling crashes.',
'pythran_deriv':'Valid-array bounds permit three steps; native Q32 rescaling crashes.',
'npbench_deep_learning_lenet_lenet':'Full bounded-shape convolution/pooling/dense translation hits native remainder crash.',
'case0388':'20,000-slot padded patch extraction exhausted the 8 GB memory cap.',
'pythran_quicksort':'Original partition plus bounded stack/loops; see resource result; zxi matches A/B.'}
rs=[]
for x in BATCH:
 n=x['program'];d=ROOT/n;r=json.load(open(d/'result.json'))
 for attempt in ['as_written','rewrite']:
  if (d/(attempt+'.zok')).exists():
   assert (d/(attempt+'.command.json')).exists(),(n,attempt,'still pending')
   assert r[attempt]['status'] in ['accepted','rejected','timeout','out_of_memory','wrong'],(n,attempt,r[attempt]['status'])
 if r['category'].startswith('accepted'):
  assert all(c['status']=='pass' for c in r['correctness'].values()),n
  chosen='as_written' if r['category']=='accepted_as_written' else 'rewrite'
  assert r[chosen]['status']=='accepted' and isinstance(r[chosen]['constraints'],int)
 if not r['category'].startswith('accepted'):
  for l in ['A','B']:
   if r['correctness'][l]['status']=='not_run':r['correctness'][l]['reason']='No R1CS-accepted attempt; no circuit acceptance claimed.'
 r['notes']+= '\n'+reasons.get(n,r['notes']) if n in reasons and reasons[n] not in r['notes'] else ''
 r['numeric_encoding']={'modulus':str(P),'real_scale':S,'signed_encoding':'centered field residues; no unsigned-only interpretation of negative fixtures','validation_tolerance':{'rtol':1e-6,'atol':1e-6}}
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
 rs.append(r)
counts=Counter(r['category'] for r in rs)
accepted=sum(counts[c] for c in counts if c.startswith('accepted'))
lines=[f'**Batch3: CirC acceptance study — {len(rs)} programs**','',f'{accepted}/{len(rs)} programs accepted and passed both original fixtures: {counts["accepted_as_written"]} as written and {counts["accepted_with_rewrite"]} with minimal rewrites. No accepted result has an unchecked or mismatching A/B output.','',
'All materialized attempts were actually compiled with the supplied stock CirC binary, `--language zsharp ... r1cs --action count`, under `MEMRUN_MAX=8G memrun.sh timeout 300`. Exact argv, environment, return code, elapsed time and compiler output are next to each attempt. `environment.json` records binary hashes; `manifest.json` records source and fixture hashes. Source/fixture snapshots are retained per program. No Zinnia IR, circuits, sizes or acceptance outcomes were used in the classifications.','',
'All signature parameters remain private, including parameters that fixture prose calls constants. Static NumPy dimensions remain unchanged in direct translations. Dynamic rewrites expose lengths/shapes alongside padding; each result records the domain argument and every rewrite. Programs with no bounded representation or requiring a prohibited gadget have documentary attempts, not placeholder programs or invented compiler results.','',
'Reals are rounded to integers at scale 2^32 and decoded from centered field residues. Direct translations use native field operations; constant integer rescaling uses native remainder and division, not a hand-built division circuit. Private real division, sqrt, log, tanh and missing sorting primitives are recorded as needing gadgets. The native field remainder lowering crashes in this stock binary (`circ_fields/src/lib.rs:571:56`, `Option::unwrap()` on `None`); four static kernels were also tried with native arbitrary integers, which the R1CS backend rejects with `Unhandled variable operator return...`. Fibonacci retains arbitrary integers because its declared n<=1024 permits outputs wider than the default field.','',
'Validation executes the original Python/NumPy function after removing only Zinnia imports, decorators and annotations. No Zinnia compilation is involved. Mutated arrays are public outputs for in-place kernels. Every accepted attempt has A.pin/B.pin, interpreter logs, decoded/reference values and maximum error in result.json, compared with relative and absolute tolerance 1e-6. Zero constraints on reshapes and wiring-only kernels are actual CirC counts. Complex struct output is decoded from zxi’s alphabetical (im,re) presentation.','',
'| Category | Programs |','|---|---:|']
for c,k in sorted(counts.items()):lines.append(f'| {c} | {k} |')
lines+=['', '| Program | Category | Constraints | Reason |','|---|---|---:|---|']
for r in rs:
 n=r['program'];accepted_attempt=r['as_written'] if r['category']=='accepted_as_written' else r['rewrite'] if r['category']=='accepted_with_rewrite' else None
 c=accepted_attempt['constraints'] if accepted_attempt else '—'
 reason=reasons.get(n,r['notes'].split('\n')[0])
 if n=='pythran_quicksort':reason='Bounded original partition and explicit stack: '+r['rewrite']['status']+'; interpreter independently matches A/B.'
 lines.append(f'| [{n}]({n}/result.json) | {r["category"]} | {c} | {reason.replace("|","/")} |')
lines+=['',
'Limits of verification: A/B comparisons are fixture checks, not an exhaustive proof of equivalence or of absence of finite-field/fixed-point overflow on every mathematical real/integer. No fixture numeric range was used to invent a loop or allocation bound. Nonaccepted programs have no verified R1CS circuit; construct/gadget classifications are source-and-library assessments, whereas compiler errors and resource failures are measured. The original quicksort translation additionally passed both fixtures in zxi even though that alone does not establish R1CS acceptance. Factorization also passed interpreter checks at n=0,1,2,1021,1024 (edge_checks.json).','',
'Reproduce retained attempts with `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python reproduce.py` from this directory, or append program names for a subset. Run `write_report.py` after reproducing to refresh this report. Reproduction uses the supplied source/fixture locations, whose hashes and local snapshots are recorded.','', 'BATCH_DONE']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
(ROOT/'summary.json').write_text(json.dumps({'programs':len(rs),'accepted':accepted,'counts':dict(sorted(counts.items()))},indent=2)+'\n')
print(json.dumps({'programs':len(rs),'accepted':accepted,'counts':dict(counts)},indent=2))
