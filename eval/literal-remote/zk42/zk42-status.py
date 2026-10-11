import pathlib,json,collections,os,datetime,subprocess
root=pathlib.Path(os.environ['RUN'])/'zk42'
superseded=set(json.loads((root/'superseded-epochs.json').read_text())) if (root/'superseded-epochs.json').exists() else set()
records=[];bad=[]
for p in root.glob('epochs/*/raw/*/*.json'):
 if p.parent.name.startswith('_') or p.parents[2].name in superseded:continue
 r=json.loads(p.read_text());records.append(r)
 if r['status']!='ok':bad.append((r['case'],r['system'],r['status'],r.get('failure')))
print('finished records:',len(records),'/290',dict(collections.Counter(r['system'] for r in records)))
print('samples:',sum(len(r['samples']) for r in records),'/962; bad:',bad)
posts=[p for p in root.glob('epochs/*-zpost/manifest.json') if p.parent.name not in superseded]
print('finished sandwiches:',sum(bool(json.loads(p.read_text()).get('finished_at')) for p in posts),'/42')
active=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=(proc/'cmdline').read_bytes().decode().split('\0')
  if args and 'python' in pathlib.Path(args[0]).name and 'benchmarking.harness' in args and '--cases' in args:
   cpus=sorted(os.sched_getaffinity(int(proc.name)));label=args[args.index('--label')+1]
   print('active',proc.name,cpus,label)
   active.append({'pid':int(proc.name),'cpus':cpus,'label':label})
 except (OSError,UnicodeError):pass

parents={int(pid):int(ppid) for pid,ppid in (line.split() for line in subprocess.check_output(['ps','-e','-o','pid=,ppid='],text=True).splitlines())}
violations=[];thread_count=0
for harness in active:
 descendants={harness['pid']}
 while True:
  more={pid for pid,ppid in parents.items() if ppid in descendants}
  if more<=descendants:break
  descendants|=more
 for pid in descendants:
  try:
   for task in pathlib.Path(f'/proc/{pid}/task').iterdir():
    try:
     mask=os.sched_getaffinity(int(task.name));thread_count+=1
     if not mask<=set(harness['cpus']):violations.append({'label':harness['label'],'pid':pid,'tid':int(task.name),'cpus':sorted(mask)})
    except ProcessLookupError:pass
  except FileNotFoundError:pass
with (root/'affinity-observations.jsonl').open('a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'harnesses':active,'threads_checked':thread_count,'thread_affinity_violations':violations})+'\n')
assert all(a['cpus'] in [list(range(i,i+8)) for i in (0,8,16,24)] for a in active)
assert len({tuple(a['cpus']) for a in active})==len(active),'overlapping timed harness groups'
assert not violations,violations
print('thread affinities checked:',thread_count,'violations:',len(violations))
