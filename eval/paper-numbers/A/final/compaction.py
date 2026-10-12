import sys,os,json,subprocess,concurrent.futures
from analyze import ROOT,OUT,rows,key

def worker(i):
 r=rows()[i];dest=OUT/'records'/key(r)
 sys.path.insert(0,str(OUT/'compiler'))
 os.environ['ZINNIA_AUDIT_COMPACTION']=str(dest/'compaction.jsonl')
 (dest/'compaction.jsonl').unlink(missing_ok=True)
 from benchmarking.harness.kernels import compile_kernel
 from benchmarking.harness.adapters.zinnia_common import ir_sha256
 c=compile_kernel(ROOT/r['kernel'],r['function'])
 digest=ir_sha256(c.get_ir_stmts())
 assert digest==json.loads((dest/'stock.json').read_text())['ir_sha256'],key(r)
 (dest/'compaction_check.json').write_text(json.dumps({'ir_sha256':digest}))
 print(i,key(r),'ok',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:worker(int(sys.argv[1]))
 else:
  # Compaction always produces a DynamicNDArray. All observed dynamic-type
  # programs are revisited, including temporaries and dead results.
  ids=[i for i,r in enumerate(rows()) if (OUT/'records'/key(r)/'dynamic.jsonl').exists()]
  with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('A_JOBS','16'))) as pool:
   codes=list(pool.map(lambda i:subprocess.run([str(ROOT/'.venv/bin/python'),str(OUT/'compaction.py'),str(i)],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}).returncode,ids))
  assert all(c==0 for c in codes)
