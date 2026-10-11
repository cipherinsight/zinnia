import study as s
import hashlib,json,collections,pathlib
# Refresh static blocker analyses (executable specs take precedence).
exec(compile((s.BASE/'blocked.py').read_text(),str(s.BASE/'blocked.py'),'exec'))
rs=[]
reasons={
'case0309':'Row-major argmax with first-tie selection.',
'case0317':'Copy the fixed 10x20x5 array.',
'case0336':'Static flip and diagonal gather.',
'case0342':'Exactly 30 statically determined list appends.',
'case0349':'The signature fixes the second dimension at six.',
'case0364':'Native dynamic gather for scalar insertion; result is 101x10.',
'case0373':'Remove unused sorting; ECDF ordinate output depends only on length 50.',
'case0390':'20,000-element padded patch scatter exceeded the 8 GB cap.',
'case0400':'CirC rejects empty arrays; use a padding slot with known zero logical length.',
'case0414':'Eight-bin capacity; guarded sums and native integer division by bin size.',
'case0436':'Row maxima and elementwise equality mask.',
'case0442':'Stable masked compaction into capacity 20, with logical length.',
'case0451':'Fixed all-zero tensor.',
'case0472':'Static slice retains the leading dimension of length one.',
'case0502':'Native dynamic gather with normalized negative indices.',
'case0509':'Bounded min/max and padded 10x10 crop with logical dimensions.',
'npbench_compute_compute':'Signed clipping, integer multiplication and addition.',
'npbench_polybench_floyd_warshall_floyd_warshall':'Static 16-step loop; snapshots preserve NumPy simultaneous updates.',
'pythran_create_grid':'Broadcast and reshape at fixed input size.',
'pythran_multitype':'Private boolean branch; preserve both tuple outputs in a struct.',
'pythran_periodic_bc':'Static boundary slice assignment.',
'pythran_reverse_cumsum':'Reverse, cumulative sum, then reverse.',
'pythran_lapl2d':'Guarded 1,024-step loop from the explicit precondition; retain both buffers.',
'pythran_sobelfilter':'Bounds follow from valid 16-element accesses; collapse repeated zero-column writes.',
}
for p in s.BATCH:
 n=p['program'];d=s.BASE/n;r=json.load(open(d/'result.json'));src=(s.ROOT/p['kernel']).read_text()
 (d/'kernel_source.py').write_text(src)
 r['source']={'kernel':p['kernel'],'function':p['function'],'sha256':hashlib.sha256(src.encode()).hexdigest()}
 if n in s.spec:
  r['encoding']='Private inputs; public return value. Real numbers are rounded to scale 2^32. Native u64 values use signed two-complement interpretation; integer arithmetic is used for candidate full-width Q32 products to avoid truncating them to 64 bits before rescaling.'
  r['fixture']=str((s.ROOT/p['kernel']).parent/('fixture.json' if ((s.ROOT/p['kernel']).parent/'fixture.json').exists() else 'fixture.'+p['function']+'.json'))
  if s.spec[n].get('fallback'):
   r['notes']='Full Q32 products require wider intermediates than native u64. The faithful native arbitrary-integer candidate fails R1CS lowering; no custom wide multiplication/rescaling/division gadget was introduced. See probes/native_integer_field_output.compile.log and probes/stdlib_inventory.json.'
 if r['category'].startswith('accepted'):
  assert all(r['correctness'][t]['status']=='pass' for t in ('A','B'))
  a=r['rewrite'] if r['category']=='accepted_with_rewrite' else r['as_written']
  assert isinstance(a['constraints'],int)
  assert all((d/(t+'.pin')).exists() and (d/(t+'.zxi.log')).exists() for t in ('A','B'))
  assert (d/('rewrite.zok' if r['category']=='accepted_with_rewrite' else 'as_written.zok')).exists()
 else:a=r['rewrite'] or r['as_written']
 r['reason']=reasons.get(n,r.get('notes') or r['as_written'].get('error'))
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
 text='# '+n+'\n\nCategory: `'+r['category']+'`\n\n'+r['reason']+'\n\n'
 for key in ('as_written','rewrite'):
  at=r[key]
  if at:
   text+=key+': '+at['status']+'\n\n'+(at.get('error') or 'Compilation accepted.')+'\n\n'
   for rw in at.get('rewrites',[]):text+='- '+rw+'\n'
 (d/'ATTEMPTS.md').write_text(text)
 rs.append(r)
counts=collections.Counter(r['category'] for r in rs)
accepted=sum(v for k,v in counts.items() if k.startswith('accepted'))
lines=['# CirC acceptance study — batch5','',f'{len(rs)} programs; **{accepted} accepted after both fixture checks** ({counts["accepted_as_written"]} as written, {counts["accepted_with_rewrite"]} with minimal rewrites).','',
'Every program has `result.json`, an unchanged `kernel_source.py` snapshot, and `ATTEMPTS.md`. Executable attempts have `.zok` files and compile logs. All accepted attempts include A/B `.pin`, interpreter logs, and decoded NumPy comparison JSON.','',
'## Method','',
'Used the supplied stock CirC and zxi binaries without rebuilding. All kernel compilations ran through `memrun.sh`, `MEMRUN_MAX=8G`, and `timeout 300`; TMPDIR was this batch’s `tmp/`. Inputs are private and return values public. The standard-library path is explicit in `study.py`. No Zinnia IR, circuits, sizes, or acceptance outcomes were used.','',
'Reals are rounded to fixed-point integers at scale 2^32. Native u64 values use two’s-complement signed interpretation; signed comparisons bias the sign bit. Static shapes remain multidimensional; rewritten dynamic outputs carry a logical length or dimensions and padding. NumPy is executed from the original function bodies with decorators and annotations removed, using the supplied virtual environment. Both fixtures use atol=rtol=1e-6; integer and boolean results are also checked exactly. For in-place kernels, all relevant mutated buffers are public outputs and checked.','',
'Wide Q32 multiplication needs an intermediate wider than u64 followed by rescaling. Six full native-`integer` kernel translations were compiled and rejected by the R1CS backend (`Unhandled variable operator return...`). An additional probe with a field output also rejects native integer inputs (`Unhandled variable operator a`). No existing fixed-point library was found in the supplied stdlib inventory. Those kernels are marked `needs_gadget: division` because repairing this requires the prohibited custom wide fixed-point arithmetic/rescaling, rather than silently overflowing products or replacing integer quotients with field inverses. Native integer division by a bin count, which does not require a widened Q32-by-Q32 quotient, is used successfully for case0414.','',
'For structurally unexpressible programs and missing gadgets, no placeholder or weakened kernel was compiled: the source construct, line, and rewrite analysis are recorded. Thus absence of a compile log for these cases is intentional. The patch-reshape rewrite was attempted and terminated under the 8 GB cap (exit 137, after IR optimization began); no constraint count or fixture correctness is claimed for it.','',
'## Counts','', '| Category | Programs |','|---|---:|']
for k,v in sorted(counts.items()):lines.append(f'| `{k}` | {v} |')
lines+=['', '## Programs','', '| Program | Category | Constraints | Reason |','|---|---|---:|---|']
for r in rs:
 a=r['rewrite'] if r['category']=='accepted_with_rewrite' else r['as_written'];c=a.get('constraints') if a else None
 lines.append(f'| [{r["program"]}]({r["program"]}/result.json) | `{r["category"]}` | {c if c is not None else "—"} | {r["reason"].replace("|","/")} |')
lines+=['','## Reproduction and verification limits','',
'Run `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python study.py <program>` from this directory to regenerate and run an executable candidate (omit the program to run all candidates). Run the same Python on `blocked.py` to regenerate structural/gadget analyses, and on `finalize.py` to audit files and regenerate this report. `run_more.py` is a convenience runner. All compiler commands are also recorded in the per-attempt JSON.','',
'All 23 finally accepted attempts passed A and B; these are fixture correctness checks, not exhaustive proofs over every numerical input. The numerical representation and rounding convention above apply throughout. No correctness is claimed for rejected, structurally unexpressible, gadget-blocked, or resource-limited attempts. Padding is decoded only according to the documented logical shape. No fixture value was used as a loop bound, array capacity, exponent, or private-data replacement.','',
'BATCH_DONE']
assert len(rs)==54
(s.BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('AUDIT OK',len(rs),accepted,dict(counts))
