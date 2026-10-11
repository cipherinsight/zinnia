import run,json,sys,concurrent.futures,traceback

def one(name):
 d=run.BASE/name;r=json.load(open(d/'result.json'));m=json.load(open(d/'translation.json'));mode=m['mode'];p=json.load(open(d/'entry.json'))
 assert not r['category'].startswith('accepted')
 checks=run.check(d,p,mode,(d/(mode+'.zok')).read_text())
 # Fixture evaluation never upgrades a failed circuit compilation.
 r['correctness']=checks
 (d/'result.json').write_text(json.dumps(r,indent=2));print(name,r['category'],{k:v['status'] for k,v in checks.items()},flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(one,n) for n in sys.argv[1:]]
 for f in concurrent.futures.as_completed(fs):
  try:f.result()
  except Exception:print(traceback.format_exc(),flush=True)
