import datetime,json,os,pathlib,subprocess,time,traceback
run=pathlib.Path(os.environ['RUN']);wt=run/'zinnia-zk42';out=run/'zk42'
case='ds1000_case0386'
env=dict(os.environ,SP1_WORKER_NUM_CORE_WORKERS='1',SP1_WORKER_CORE_BUFFER_SIZE='1',SP1_WORKER_NUM_SETUP_WORKERS='1',SP1_WORKER_SETUP_BUFFER_SIZE='1')
settings={k:v for k,v in env.items() if k.startswith('SP1_WORKER_')}
plan={'case':case,'reason':'Default SP1 warmup exceeded 30G RSS cap (32239599616 bytes); reduce in-flight workers while keeping eight computation threads, core proofs, and 30G cap.','settings':settings,'source_commit':'6fa27f8e58bd450a46670cb981cb7beb2fc9dfc9'}
(out/'retry-plan.json').write_text(json.dumps(plan,indent=2))
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
  if len(assignments)==42:
   for j in range(4):
    cpus=f'{j*8}-{j*8+7}'
    if cpus in last and f'zk42-{last[cpus]}-zpost exit 0 ' not in log:continue
    if not occupied(j):available.append(j)
  if available:break
  time.sleep(5)
 j=available[0];cpus=f'{j*8}-{j*8+7}'
 print('RETRY_GROUP',cpus,datetime.datetime.now().isoformat(),flush=True)
 superseded=[p.name for p in sorted((out/'epochs').glob(f'*-zk42-{case}-*'))]
 assert len(superseded)==3,superseded
 (out/'superseded-epochs.json').write_text(json.dumps(superseded,indent=2))
 result.update({'cpus':cpus,'superseded_epochs':superseded,'started_at':datetime.datetime.now().isoformat(),'phases':[]})
 for phase,systems,reps in [('zpre','zinnia_halo2,zinnia_noir',5),('zkvm','sp1,risc0,cairo',1),('zpost','zinnia_halo2,zinnia_noir',5)]:
  label=f'zk42-retry1-{case}-{phase}'
  cmd=['taskset','-c',cpus,str(wt/'.venv/bin/python'),'-m','benchmarking.harness','run','--label',label,'--case-set',str(out/'zk42-cases.json'),'--cases',case,'--systems',systems,'--repeats',str(reps),'--threads','8','--memory-max','30G','--timeout','14400','--results-dir',str(out/'epochs')]
  with (out/'zk42-commands-retry.jsonl').open('a') as f:f.write(json.dumps({'time':datetime.datetime.now().isoformat(),'cwd':str(wt),'argv':cmd,'env':settings})+'\n')
  with (out/(label+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=wt,env=env,stdout=f,stderr=subprocess.STDOUT)
  result['phases'].append({'label':label,'returncode':p.returncode})
  print(label,'exit',p.returncode,datetime.datetime.now().isoformat(),flush=True)
  if p.returncode:raise RuntimeError(f'{label}: {p.returncode}')
 result['ok']=True
except Exception:
 result['error']=traceback.format_exc();print(result['error'],flush=True)
finally:
 result['finished_at']=datetime.datetime.now().isoformat()
 (out/'retry-finished.json').write_text(json.dumps(result,indent=2))
