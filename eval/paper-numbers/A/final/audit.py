"""Compile only. No proof, key, mock, or timing operation is invoked."""
import sys,os,json,hashlib,subprocess,concurrent.futures,gzip
from pathlib import Path
from analyze import ROOT,OUT,rows,key

def worker(i,variant):
 r=rows()[i];dest=OUT/'records'/key(r)
 sys.path.insert(0,str(OUT/'compiler' if variant=='instrumented' else ROOT))
 os.environ['ZINNIA_LAYER_INPUT']=str(dest/(variant+'.layer.txt'))
 (dest/(variant+'.layer.txt')).unlink(missing_ok=True)
 if variant=='instrumented':
  os.environ['ZINNIA_AUDIT_DYNAMIC']=str(dest/'dynamic.jsonl')
  os.environ['ZINNIA_AUDIT_IR']=str(dest/'layer.json')
  (dest/'dynamic.jsonl').unlink(missing_ok=True)
 from benchmarking.harness.kernels import compile_kernel
 from benchmarking.harness.adapters.zinnia_common import ir_sha256
 try:
  c=compile_kernel(ROOT/r['kernel'],r['function'])
  result={'compiled':True,'ir_sha256':ir_sha256(c.get_ir_stmts()),'layout':c.get_output_layout()}
  if variant=='instrumented':
   for name in ['layer.json',variant+'.layer.txt']:
    p=dest/name
    with p.open('rb') as src,gzip.open(str(p)+'.gz','wb') as dst:
     import shutil
     shutil.copyfileobj(src,dst)
    p.unlink()
 except BaseException as e: result={'error':repr(e)}
 (dest/(variant+'.json')).write_text(json.dumps(result,indent=2))
 print(i,variant,key(r),result.get('error','ok'),flush=True)

def main(variant):
 with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('A_JOBS','16'))) as pool:
  codes=list(pool.map(lambda i:subprocess.run([str(ROOT/'.venv/bin/python'),str(OUT/'audit.py'),'worker',str(i),variant],cwd=OUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}).returncode,range(len(rows()))))
 assert all(c==0 for c in codes),codes
if __name__=='__main__':
 if sys.argv[1]=='worker': worker(int(sys.argv[2]),sys.argv[3])
 else:main(sys.argv[1])
