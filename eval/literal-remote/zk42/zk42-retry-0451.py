import datetime,json,os,pathlib,subprocess,time,traceback
run=pathlib.Path(os.environ['RUN']);wt=run/'zinnia-zk42';out=run/'zk42'
case='ds1000_case0451'
env=dict(os.environ)
settings={}
plan={'case':case,'reason':'The original zkVM phase overlapped the old zk20 untimed warmup verification on 0-7, which ended at 09:20:07 UTC. Repeat the whole sandwich for isolated timings.','settings':settings,'source_commit':'6fa27f8e58bd450a46670cb981cb7beb2fc9dfc9'}
(out/'retry-0451-plan.json').write_text(json.dumps(plan,indent=2))
def occupied(j):
 for proc in pathlib.Path('/proc').glob('[0-9]*'):
  try:
   parts=(proc/'cmdline').read_bytes().decode().split('\0')
   if not parts or 'python' not in pathlib.Path(parts[0]).name:continue
   argv=' '.join(parts)
   if j==1 and 'zinnia-literal' in argv and 'zk20-run.py' in argv:return True
   if 'benchmarking.harness run ' in argv or 'zk20-verify-warmups.py' in argv:
    if os.sched_getaffinity(int(proc.name)) & set(range(j*8,j*8+8)):return True
  except (OSError,UnicodeError):pass
 return False
result=dict(plan,ok=False)
try:
 while True:
  log=(run/'logs/zk42-measure-queue.log').read_text()
  assignments=[line.split() for line in log.splitlines() if line.startswith('ASSIGN ')]
  last={a[2]:a[1] for a in assignments}
  available=[]
  retry_log=(run/'logs/zk42-retry-0386.log').read_text()
  reserved=[line.split()[1] for line in retry_log.splitlines() if line.startswith('RETRY_GROUP ')]
  original_superseded=json.loads((out/'superseded-epochs.json').read_text()) if (out/'superseded-epochs.json').exists() else []
  if len(assignments)==42 and reserved and any('ds1000_case0386' in e for e in original_superseded):
   for j in range(4):
    cpus=f'{j*8}-{j*8+7}'
    if cpus in reserved:continue
    if cpus in last and f'zk42-{last[cpus]}-zpost exit 0 ' not in log:continue
    if not occupied(j):available.append(j)
  if available:break
  time.sleep(5)
 j=available[0];cpus=f'{j*8}-{j*8+7}'
 print('RETRY_GROUP',cpus,datetime.datetime.now().isoformat(),flush=True)
 superseded=[p.name for p in sorted((out/'epochs').glob(f'*-zk42-{case}-*'))]
 assert len(superseded)==3,superseded
 (out/'superseded-epochs.json').write_text(json.dumps(original_superseded+superseded,indent=2))
 result.update({'cpus':cpus,'superseded_epochs':superseded,'started_at':datetime.datetime.now().isoformat(),'phases':[]})
 for phase,systems,reps in [('zpre','zinnia_halo2,zinnia_noir',5),('zkvm','sp1,risc0,cairo',1),('zpost','zinnia_halo2,zinnia_noir',5)]:
  label=f'zk42-retry1-{case}-{phase}'
  cmd=['taskset','-c',cpus,str(wt/'.venv/bin/python'),'-m','benchmarking.harness','run','--label',label,'--case-set',str(out/'zk42-cases.json'),'--cases',case,'--systems',systems,'--repeats',str(reps),'--threads','8','--memory-max','30G','--timeout','14400','--results-dir',str(out/'epochs')]
  with (out/'zk42-commands-retry-0451.jsonl').open('a') as f:f.write(json.dumps({'time':datetime.datetime.now().isoformat(),'cwd':str(wt),'argv':cmd,'env':settings})+'\n')
  with (out/(label+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=wt,env=env,stdout=f,stderr=subprocess.STDOUT)
  result['phases'].append({'label':label,'returncode':p.returncode})
  print(label,'exit',p.returncode,datetime.datetime.now().isoformat(),flush=True)
  if p.returncode:raise RuntimeError(f'{label}: {p.returncode}')
 result['ok']=True
except Exception:
 result['error']=traceback.format_exc();print(result['error'],flush=True)
finally:
 result['finished_at']=datetime.datetime.now().isoformat()
 (out/'retry-0451-finished.json').write_text(json.dumps(result,indent=2))
