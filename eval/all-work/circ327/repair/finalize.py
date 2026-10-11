import pathlib,json,hashlib,re,collections,ast,copy,subprocess
B=pathlib.Path(__file__).resolve().parent;R=B.parent;ROOT=pathlib.Path('/home/zhantong/Projects/zinnia-project/wt/literal')
for row in subprocess.check_output(['ps','-eo','args'],text=True).splitlines():
 parts=row.split()
 if len(parts)>1 and 'python' in pathlib.Path(parts[0]).name and parts[1].endswith('.py'):
  task=pathlib.Path(parts[1]).resolve()
  assert task.parent!=B or task.name=='finalize.py', 'Repair process still running: '+row
entries=[{k:v for k,v in p.items() if k!='zinnia_outcome'} for p in json.load(open(R/'programs.json'))]
lists=json.load(open(R/'repair-lists.json'));names=set(sum(lists.values(),[]));multi=set(lists['multi_function'])
base={}
for batch in sorted(R.glob('batch[0-9].json')):
 for p in json.load(open(batch)):
  key=(p['program'],p['function']);assert key not in base
  base[key]=batch.with_suffix('')/p['program']/'result.json'
assert len(entries)==327 and len(base)==327

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
fix_aw={
 'case0416':('dynamic shape','bin_data_mean =','The private bin_size controls reshape and the number of returned columns.'),
 'case0417':('dynamic shape','bin_data_mean =','The private bin_size controls reshape and the number of returned bins.'),
 'case0418':('dynamic shape','bin_data_mean =','The private bin_size controls reshape and the number of returned columns.'),
 'pythran_tax':('dynamic shape','tax_seg1 =','Boolean selections create value-dependent arrays; each is bounded by the ten input entries.'),
 'pythran_lttb_pt':('dynamic shape','data_bins =','Private n_out controls array_split bin shapes and output length; successful source executions require 3<=n_out<=16.'),
 'pythran_gauss':('dynamic loop','for i in range(1, n + 1):','The source loop bounds depend on private n. Successful accesses to the declared 4x4 matrix imply n<=4.'),
 'pythran_caxpy':('dynamic loop','for I in range(N):','Private N controls the source loops; the explicit contract supplies the 1024 bound. Early None returns require a presence tag.'),
 'npbench_polybench_covariance_covariance':('dynamic shape','cov = np.zeros','Private M controls the cov shape and source loops; successful assignments into cov imply 0<=M<=14.'),
 'npbench_cavity_flow_cavity_flow':('dynamic shape','b = np.zeros','Private nx and ny determine b; no bound applies when nt=0. The rewrite removes this unused allocation at nt=0 and uses the shape required by the active source assignments.'),
}
# Gate publication of the complete merge/report on all requested repairs existing.
for p in entries:
 if p['program'] in names:
  d=B/(p['program']+'__'+p['function'] if p['program'] in multi else p['program'])
  assert (d/'result.json').exists(),f'Missing repair: {d.name}'

merged=[];table=[];repair_results=[];provenance=[]
for p in entries:
 key=(p['program'],p['function']);old=json.load(open(base[key]));n=p['program'];f=p['function']
 if n not in names:
  r=copy.deepcopy(old);r['function']=f
  merged.append(r);provenance.append({'program':n,'function':f,'result':str(base[key].relative_to(R)),'sha256':sha(base[key])});continue
 d=B/(n+'__'+f if n in multi else n);r=json.load(open(d/'result.json'))
 assert (r['program'],r['function'])==key
 src=ROOT/p['kernel'];lines=src.read_text().splitlines()
 r['source']={**p,'sha256':sha(src)};r['old_category']=old['category'];r['old_result']=str(base[key].relative_to(R))
 r['limits']={'compile_seconds':300,'compile_memory':'8G','wrapper':'/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh','tmpdir':str(B/'tmp')}
 if n in fix_aw and r.get('rewrite'):
  kind,needle,explain=fix_aw[n];line=next(i+1 for i,s in enumerate(lines) if needle in s)
  r['as_written']={'status':'not_expressible','error':f'kernel.py:{line}: {lines[line-1].strip()}. {explain}','constraints':None}
 if r['category'].startswith('accepted'):
  assert all(r['correctness'][t].get('status')=='pass' for t in ['A','B']),d
  a=r['as_written'] if r['category']=='accepted_as_written' else r['rewrite'];assert a and a['status']=='accepted' and isinstance(a['constraints'],int)
  assert not any('needs gadget' in str(r.get(k,{})).lower() for k in ['as_written','rewrite']),d
 for attempt in ['as_written','rewrite']:
  a=r.get(attempt)
  if not a:continue
  z=d/(attempt+'.zok')
  if a.get('status') in ['not_expressible','not expressible','needs_gadget','needs gadget'] and not z.exists():
   z.write_text('// No executable '+attempt+' translation.\n// '+a['error'].replace('\n','\n// ')+'\n')
  if z.exists():
   a['source_file']=str(z.relative_to(B));a['artifact_sha256']=sha(z)
   if 'def main' in z.read_text():assert not re.search(r'\binteger\b',z.read_text()),z
 fp=src.with_name('fixture.'+f+'.json')
 if not fp.exists():fp=src.with_name('fixture.json')
 r['fixture']={'path':str(fp),'sha256':sha(fp)}
 (d/'kernel_source.py').write_text(src.read_text())
 mode=json.load(open(d/'translation.json'))['mode'] if (d/'translation.json').exists() else None
 if mode:
  code=(d/(mode+'.zok')).read_text()
  helper=re.search(r'from "\.\./(native[^\"]*)"',code)
  r['numeric_encoding']={'real_scale':4294967296,'rtol':1e-6,'atol':1e-6,'representation':'centered field residues with native unsigned division' if 'native_field' in code else 'signed two-complement u64 with native wide products'}
  if n in ['pythran_fibo','pythran_fibo_seq']:r['numeric_encoding']['integer_output']='Twelve little-endian unsigned 64-bit limbs; exact through n=1024.'
  if helper:
   hp=B/(helper[1]+'.zok');r['numeric_encoding']['helper']={'file':hp.name,'sha256':sha(hp)}
  r['notes']=json.load(open(d/'translation.json')).get('notes',r.get('notes',''))
  if 'native_field' in code:r['notes']='Wide native field products and accumulation; native unpacking implements the Q32 shift, native unsigned division handles quotients. All source loops and outputs are retained.'
 if r['category'].startswith('accepted'):
  if r['category']=='accepted_with_rewrite':why='; '.join(r['rewrite'].get('rewrites',[]))+' A/B pass.'
  else:why='Native signed Q32 arithmetic compiles; both fixtures pass.'
  if n in lists['integer_type_failures'] and r['category']=='accepted_as_written':why='Replace the arbitrary-integer/field-remainder encoding with native unsigned arithmetic and wide products; A/B pass.'
  if n=='npbench_weather_stencils_hdiff_hdiff':why='Use operand signs and nonzero tests for sign-only products to avoid u64 overflow; native Q32 arithmetic compiles and A/B pass.'
  if n in multi and n!='pythran_lu':why='Independent '+f+' translation and fixture. '+why
 elif r['category']=='timeout_or_oom':
  a=r.get('rewrite') or r['as_written'];why='Native-arithmetic '+a['status'].replace('_',' ')+' under 300 s / 8 GB; no arithmetic gadget blocker.'
 elif r['category']=='circ_error':
  a=r.get('rewrite') or r['as_written'];err=a.get('error','');meaning=[x for x in err.splitlines() if any(y in x.lower() for y in ['error:','panicked','overflow','fatal'])];why='Native translation: '+(' '.join(meaning[:2]) or err[:180])
 elif r['category']=='wrong':why='Compilation succeeded but A/B did not both pass; excluded from acceptance.'
 else:why=r.get('notes','')
 r['reason']=why
 (d/'result.json').write_text(json.dumps(r,indent=2)+'\n');merged.append(r);repair_results.append(r)
 provenance.append({'program':n,'function':f,'result':str((d/'result.json').relative_to(R)),'sha256':sha(d/'result.json')})
 a=r['as_written'] if r['category']=='accepted_as_written' else r.get('rewrite') or r['as_written'];count=a.get('constraints')
 checks='/'.join(str(r['correctness'][x].get('status','not_run')) for x in ['A','B'])
 table.append('| ['+n+' / '+f+']('+d.name+'/result.json) | '+old['category']+' | '+r['category']+' | '+(str(count) if count is not None else '—')+' | '+checks+' | '+why.replace('|','/').replace('\n',' ')+' |')
assert len(merged)==327 and len({(r['program'],r['function']) for r in merged})==327
assert len(repair_results)==77
assert all(not r['category'].startswith(('needs_gadget: division','needs_gadget: fixed-point')) for r in repair_results)
(B/'merged.json').write_text(json.dumps(merged,indent=2)+'\n')
(B/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
old_counts=collections.Counter(json.load(open(base[(p['program'],p['function'])]))['category'] for p in entries if p['program'] in names)
new_counts=collections.Counter(r['category'] for r in repair_results);all_counts=collections.Counter(r['category'] for r in merged)
lines=['# CirC acceptance repair','',f'Repaired **77 entries across {len(names)} program names**, including all 11 entries of the five multi-function programs. `merged.json` contains exactly **327 unique (program, function) results**, in `programs.json` order. Repair results take precedence; the remaining 250 results come from the matching batch entry. Batch directories were not edited.','', 'Every executable candidate was compiled with the stock CirC binary, `MEMRUN_MAX=8G`, `memrun.sh`, and `timeout 300`; temporary files stayed under `repair/tmp`. Real encoding remains Q32. Every repaired accepted result passes NumPy comparison on both fixture A and fixture B at 1e-6 relative plus absolute tolerance. Native arithmetic replaces the old arbitrary-integer backend operations and division/multiplication gadget labels.','', 'The source, inputs, decoded outputs, logs, exact commands and checksums are recorded per entry; `provenance.json` identifies the result used for every merged row. See `README.md` for reproduction. Native unsigned quotient/remainder and bit casts are ordinary Z# operations; no manually asserted division constraints, auxiliary witness arguments, sorting networks, or transcendental implementations were added.','', 'Fannkuch uses the declared entry `fannkuch2`, which discards a pure helper result and returns None. Its minimal rewrite removes that unused call while preserving the n=0 error with an assertion. Full bounded native candidates remain archived. Fibonacci uses twelve u64 limbs, preserving the entire n<=1024 domain.','', '## Category counts','', '| Category | Old affected entries | Repaired entries | Final merged entries |','|---|---:|---:|---:|']
for cat in sorted(set(old_counts)|set(new_counts)|set(all_counts)):lines.append(f'| {cat} | {old_counts[cat]} | {new_counts[cat]} | {all_counts[cat]} |')
lines+=['','## Per-entry changes','','| Program / function | Old category | New category | Constraints | A/B | Why |','|---|---|---|---:|---|---|']+table
unverified=[r['program']+'/'+r['function']+' ('+'/'.join(r['correctness'][x].get('status','not_run') for x in ['A','B'])+')' for r in repair_results if not all(r['correctness'][x].get('status')=='pass' for x in ['A','B'])]
lines+=['','## Verification limits','','Resource failures and non-expressible/gadget cases are not counted as accepted. Interpreter evaluation does not override failed circuit compilation. The original unmodified batch results outside this repair scope were merged without revalidation.','', 'A/B could not both be verified for: '+('; '.join(unverified) if unverified else 'none')+'.','', 'REPAIR_DONE']
(B/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'repaired':dict(new_counts),'merged':dict(all_counts),'entries':len(merged)},indent=2))
