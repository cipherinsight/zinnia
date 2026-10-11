from refinements import *
import hashlib,collections

def finalize():
 rows=[]
 for n,e in ENTRIES.items():
  d=BASE/n;p=d/'result.json';r=json.loads(p.read_text());s=SPECS[n]
  fp,f=fixture(n)
  r['source']=e['kernel'];r['function']=e['function'];r['fixture']=str(fp)
  r['source_sha256']=hashlib.sha256((ROOT/e['kernel']).read_bytes()).hexdigest()
  r['encoding']='Real and complex components rounded to nearest integer at scale 2^32; centered field residues for field translations; native arbitrary-precision integers for integer translations. Public return includes all logical output values; padded returns include shape/presence metadata.'
  r['limits']={'compile_seconds':300,'compile_memory':'8G','wrapper':MEM}
  for mode in ['as_written','rewrite']:
   file=d/(mode+'.zok')
   if file.exists():r.setdefault('translation_sha256',{})[mode]=hashlib.sha256(file.read_bytes()).hexdigest()
  if 'code' not in s:
   if r['category'].startswith('needs_gadget') and r['as_written']['status']=='not expressible':r['as_written']['status']='needs gadget'
   (d/'ATTEMPTS.md').write_text('# Translation assessment\n\n'+r['notes']+'\n\nAs written: '+r['as_written']['status']+'.\n\nMinimal rewrite: '+r['rewrite']['status']+'. '+r['rewrite']['error']+'\n\nNo invalid Z# placeholder was compiled or counted as a compiler rejection. See the batch stdlib inventory and numeric probes for available native operations.\n')
  elif r['as_written']['status']=='not expressible':
   (d/'AS_WRITTEN.md').write_text(r['as_written']['error']+'\n\nNo faithful statically typed Z# declaration of the private shape/loop is available. The bounded rewrite is in rewrite.zok.\n')
  if r['category'].startswith('accepted'):
   assert all(isinstance(v,dict) and v['status']=='pass' for v in r['correctness'].values()),n
   a=r['as_written'] if r['category']=='accepted_as_written' else r['rewrite']
   assert a['status']=='accepted' and a['constraints'] is not None,n
  p.write_text(json.dumps(r,indent=2)+'\n');rows.append(r)
 counts=collections.Counter(r['category'] for r in rows)
 lines=['# CirC batch2 acceptance study','',f'{len(rows)} programs; {sum(v for k,v in counts.items() if k.startswith("accepted"))} accepted with both fixture checks passing.','',
 'Each compilable translation was run locally using the supplied stock CirC executable, `--language zsharp ... r1cs --action count`, `timeout 300`, and the supplied `memrun.sh` with `MEMRUN_MAX=8G`. `TMPDIR` stayed inside this batch. The installed Z# stdlib was selected with `ZSHARP_STDLIB_PATH`. No Zinnia IR, circuits, sizes, or acceptance results were used to select a translation or category.','',
 'Real inputs use Q32 (scale 2^32), with complex values represented by real/imaginary pairs. All inputs remain private. Arrays preserve their declared dimensions; padded dynamic outputs include their logical dimensions or a presence tag. Accepted output values were decoded and compared against the original Python function, with decorators and annotations removed, using NumPy and `atol=rtol=1e-6`. These are A/B differential checks, not a proof over all inputs. Fixed-point quantization is mandated by the brief; IEEE nonfinite behavior is not represented.','',
 'The native `integer` type passes the Z# frontend but is unsupported in the R1CS backend. Arithmetic translations using it are retained with actual rejection logs. `probes/integer_field.log` also establishes that wrapping native integer arithmetic in field conversions does not fix the backend failure. Replacing these operations with finite-field division would change fixed-point semantics; custom rescaling/division gadgets were therefore not written. Missing sort/transcendental/search gadgets and truly unbounded shapes/loops are recorded as translation blockers, without compiling misleading stubs.','',
 'Data-movement-only programs can have zero constraints after CirC optimization. The 20,000-element reshape uses a statically expanded array literal: the equivalent repeated-assignment version compiled, but exhausted the interpreter memory cap. The final literal version compiles and both full-input zxi checks pass.','',
 'Reproduce all entries with `PYTHONDONTWRITEBYTECODE=1 /home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python run_batch.py`. The generator, original source copies, compile commands/logs, fixture `.pin` files, interpreter logs, expected outputs, decoded outputs, and per-program JSON records are retained here.','',
 '| Program | Category | Constraints | Reason |','|---|---|---:|---|']
 for r in rows:
  a=r['as_written'] if r['category']=='accepted_as_written' else r.get('rewrite')
  c=a.get('constraints') if a else None
  reason=r['notes'].replace('|','\\|').replace('\n',' ')
  lines.append(f"| {r['program']} | {r['category']} | {c if c is not None else '—'} | {reason} |")
 lines+=['','## Counts','','| Category | Programs |','|---|---:|']
 for k,v in sorted(counts.items()):lines.append(f'| {k} | {v} |')
 lines+=['', '## Verification limits','', 'All entries counted as accepted pass both A and B. Programs blocked at translation, rejected by CirC, or stopped by the resource cap have no verified circuit output and are not counted as accepted. Resource failures and errors are preserved in result.json and logs. No fixture-specific bound was used.','', 'BATCH_DONE']
 (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(dict(counts));print('BATCH_DONE')
if __name__=='__main__':finalize()
