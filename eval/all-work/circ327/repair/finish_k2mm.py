import pathlib,json,time,subprocess,sys,run
B=pathlib.Path(__file__).resolve().parent;d=B/'npbench_polybench_k2mm_k2mm'
while any('/examples/circ --language zsharp '+str(d/'as_written.zok') in row for row in subprocess.check_output(['ps','-eo','args'],text=True).splitlines()):time.sleep(1)
time.sleep(1)
r=json.load(open(d/'result.json'))
if r['as_written']['status']=='accepted':
 p=json.load(open(d/'entry.json'));r['correctness']=run.check(d,p,'as_written',(d/'as_written.zok').read_text())
 r['category']='accepted_as_written' if all(v['status']=='pass' for v in r['correctness'].values()) else 'wrong'
 (d/'result.json').write_text(json.dumps(r,indent=2));print('as_written',r['category'],flush=True)
else:
 subprocess.run([sys.executable,str(B/'retry.py'),d.name],check=True)
 r=run.process(d)
 if not r['category'].startswith('accepted'):
  p=json.load(open(d/'entry.json'));r['correctness']=run.check(d,p,'rewrite',(d/'rewrite.zok').read_text())
  (d/'result.json').write_text(json.dumps(r,indent=2));print('final',r['category'],{k:v['status'] for k,v in r['correctness'].items()},flush=True)
