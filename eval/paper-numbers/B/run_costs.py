import measure,subprocess,concurrent.futures,sys,os
from bounds_cost import CONTROLS
out=measure.HERE/'costs';out.mkdir(exist_ok=True)
def run(case):
 with (out/(case+'.log')).open('w') as log:
  try:
   p=subprocess.run(['systemd-run','--user','--scope','-q','-p','MemoryMax=16G',sys.executable,'-B',str(measure.HERE/'bounds_cost.py'),case],stdout=log,stderr=log,timeout=900,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
   print(case,p.returncode,flush=True)
  except subprocess.TimeoutExpired:print(case,'timeout',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,CONTROLS))
