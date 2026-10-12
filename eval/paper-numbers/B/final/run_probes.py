import measure,concurrent.futures,subprocess,os,json,sys
out=measure.HERE/'probes';out.mkdir(exist_ok=True)
def run(p):
 name=measure.ident(p); target=out/(name+'.json')
 if target.exists():return
 with (out/(name+'.log')).open('w') as log:
  try:
   r=subprocess.run([sys.executable,'-B',str(measure.HERE/'probe.py'),name],stdout=log,stderr=log,timeout=600,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
   if not target.exists():target.write_text(json.dumps(dict(id=name,error=f'exit {r.returncode}')))
  except subprocess.TimeoutExpired:target.write_text(json.dumps(dict(id=name,error='resource timeout 600s')))
 print(name,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('B_JOBS','16'))) as pool:list(pool.map(run,measure.entries()))
