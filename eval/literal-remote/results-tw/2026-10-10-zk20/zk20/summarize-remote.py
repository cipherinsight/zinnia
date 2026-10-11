import csv,json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]); report=pathlib.Path(sys.argv[2])
data={}; commits=set(); bad=[]
for ep in sorted((root/'epochs').glob('*')):
 m=json.loads((ep/'manifest.json').read_text());commits.add(m['repository']['commit'])
 if not m['finished_at']:bad.append(f'{ep.name}: unfinished')
 if m['repository']['dirty']:bad.append(f'{ep.name}: dirty source')
 phase=ep.name.rsplit('-',1)[-1]
 if m['config']['threads']!=8:bad.append(f'{ep.name}: wrong thread count')
 if m['config']['repeats']!=(1 if phase=='zkvm' else 5):bad.append(f'{ep.name}: wrong repeat count')
 for p in sorted((ep/'raw').glob('*/*.json')):
  if p.parent.name.startswith('_'):continue
  r=json.loads(p.read_text()); key=(r['case'],r['system'],phase)
  data[key]=r
  if r.get('tamper',{}).get('status')!='rejected':bad.append(f"{ep.name}/{r['system']}: tamper not rejected")
  if r['status']!='ok':bad.append(f"{ep.name}/{r['system']}: {r['status']} {r['failure']}")
rows=[]; cases=json.loads((root/'zk20-cases.json').read_text())['cases']
for e in cases:
 case=e['case']; row={'program':case,'proved':[],'outputs':'match'}
 systems=['zinnia_halo2','zinnia_noir','sp1','risc0']+(['cairo'] if case.startswith('ds1000') else [])
 for s in systems:
  phases=['zpre','zpost'] if s.startswith('zinnia') else ['zkvm']
  samples=[]; good=True
  for phase in phases:
   r=data.get((case,s,phase))
   if not r:bad.append(f'Missing {case}/{s}/{phase}');good=False;continue
   expected=5 if s.startswith('zinnia') else 1
   if len(r['samples'])!=expected:bad.append(f'Wrong sample count {case}/{s}/{phase}');good=False
   times=[]
   for sample in r['samples']:
    check=sample.get('output_check',{}).get('status')=='match'
    verified=sample.get('verify',{}).get('result',{}) or {}
    if not check or not verified.get('verified'):good=False;row['outputs']='incomplete';bad.append(f"Unchecked/unverified {case}/{s}/{phase}/{sample['tag']}")
    if sample.get('prove',{}).get('prove_core_s') is not None:times.append(sample['prove']['prove_core_s'])
   row[s+'_'+phase+'_median_s']=statistics.median(times) if times else None
   samples.extend(times)
   good=good and r['status']=='ok'
  row[s+'_median_s']=statistics.median(samples) if samples else None
  if good:row['proved'].append(s)
  else:row['outputs']='incomplete'
 for z in ['zinnia_halo2','zinnia_noir']:
  for v in systems[2:]:
   a=row.get(z+'_median_s');b=row.get(v+'_median_s');row[z+'_over_'+v]=a/b if a is not None and b else None
 rows.append(row)
warmups=json.loads((root/'warmup-verification.json').read_text()) if (root/'warmup-verification.json').exists() else []
if len(warmups)!=137:bad.append(f'Expected 137 warmup verifications, found {len(warmups)}')
for r in warmups:
 if not r['verify']['ok'] or not r['verify'].get('result',{}).get('verified') or r['output_check']['status']!='match':bad.append(f"Warmup not verified/matched: {r['epoch']}/{r['system']}")
if 'ALL_BUILDS_SUCCEEDED' not in (root/'build.log').read_text():bad.append('Build success marker missing')
(root/'summary.json').write_text(json.dumps({'epoch':'zk20','commits':sorted(commits),'rows':rows,'issues':bad},indent=2))
fields=list(dict.fromkeys(k for row in rows for k in row))
with (root/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def num(x):return '—' if x is None else f'{x:.6g}'
lines=['# Remote zkVM sandwich — zk20','', 'All timings below were measured on the remote machine. Local execute checks are not timing inputs.', '', 'Commits: '+', '.join('`'+c+'`' for c in sorted(commits))+'.', '', 'Protocol: per program, Zinnia halo2/Noir ×5, SP1 core and RISC Zero composite (plus Cairo Stwo for DS-1000) ×1, then Zinnia halo2/Noir ×5. The harness also makes one untimed warmup per system/phase. All jobs have 8 threads pinned to one of 0–7, 8–15, 16–23, 24–31; no timed job uses SMT cores 32–63. Memory cap: 30 GB, or 64 GB for ResNet; step timeout 14,400 s. Setup uses fixture A; proofs and NumPy checks use fixture B. Prove-core excludes setup and verification. Zinnia columns pool the ten measured samples; zkVM columns contain the single measured sample. Ratios are pooled-median Zinnia / zkVM (smaller is faster). Separate before/after medians and exact ratios are in summary.csv.', '', '## Per-program results', '', '| Program | Systems proved | Outputs | Halo2 s | Noir s | SP1 s | RISC0 s | Cairo s | H/SP1 | H/R0 | H/Cairo | N/SP1 | N/R0 | N/Cairo |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
short={'zinnia_halo2':'H','zinnia_noir':'N','sp1':'SP1','risc0':'R0','cairo':'Cairo'}
for r in rows:
 vals=[r['program'],', '.join(short[s] for s in r['proved']),r['outputs']]+[num(r.get(s+'_median_s')) for s in short]+[num(r.get(z+'_over_'+v)) for z in ('zinnia_halo2','zinnia_noir') for v in ('sp1','risc0','cairo')]
 lines.append('| '+' | '.join(vals)+' |')
lines+=['','H = zinnia_halo2; N = zinnia_noir. Cairo does not apply to the three NPBench/Pythran programs.','','## Evidence and commands','',f'Results: `{root}`. Raw records include each sample’s prove-core time, output comparison, proof verification, and tamper-check outcome. Manifests include source commits, toolchain versions, build records, binary hashes, and machine/configuration details. Warmup verification is in `warmup-verification.json`. Proofs remain on the remote under `$RUN/zk20/epochs/*/work/`; the local copy omits bulky work directories.','', 'Build and transport commands: `remote-commands.md` beside this report. Launcher: `zk20-run.py`; exact per-shard harness commands are in `zk20-commands-s*.jsonl` in the result directory. Build log: `build.log` in the result directory.','', 'Cairo follows the existing ff1 harness: prove-core times binary Stwo generation; Scarb 2.20.1 verifies a separately regenerated JSON proof. The binary serialization is not directly accepted by that verifier. Both are generated from the same execution. This is a verification-format limitation, not a wrapping step.','','## Unverified or incomplete','']
lines+=bad if bad else ['No missing applicable system or unchecked output in the measured samples; every measured proof and warmup has a successful verification verdict, subject to Cairo’s serialization caveat above.']
if not bad:lines+=['','ALL_PROOFS_VERIFIED']
report.write_text('\n'.join(lines)+'\n')
print('cases',len(rows),'issues',len(bad));print('\n'.join(bad))
