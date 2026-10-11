import os,json,pathlib,subprocess,time,re,concurrent.futures,sys
BASE=pathlib.Path(__file__).resolve().parent
ENV=os.environ.copy();ENV.update(TMPDIR=str(BASE/'tmp'),MEMRUN_MAX='8G',ZSHARP_STDLIB_PATH='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/src/third_party/ZoKrates/zokrates_stdlib/stdlib')
BIN='/home/zhantong/.cache/zkcc-reconfirm/circ-stock/target/release/examples/'
PREFIX=['/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh','timeout','300']
M=json.load(open(BASE/'manifest.json'))
def compile_one(p):
 m=M[p];d=BASE/p
 result={'program':p,'category':m.get('category'),'as_written':None,'rewrite':None,'correctness':{'A':'not run: no accepted attempt','B':'not run: no accepted attempt'},'notes':[m.get('reason','')]}
 for a in ['as_written','rewrite']:
  spec=m.get(a)
  if spec is None:continue
  if 'file' not in spec:result[a]=spec;continue
  cmd=PREFIX+[BIN+'circ','--language','zsharp',str(d/spec['file']),'r1cs','--action','count']
  (d/(a+'.command.json')).write_text(json.dumps({'argv':cmd,'env':{k:ENV[k] for k in ['TMPDIR','MEMRUN_MAX','ZSHARP_STDLIB_PATH']}},indent=2))
  t=time.monotonic()
  with open(d/(a+'.log'),'w') as log:
   proc=subprocess.run(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT,cwd=BASE)
  log=(d/(a+'.log')).read_text();match=re.search(r'Final r1cs: (\d+) constraints',log)
  status='accepted' if proc.returncode==0 and match else 'timeout' if proc.returncode==124 else 'out of memory' if proc.returncode in [-9,137] or 'oom-kill' in log.lower() else 'rejected'
  entry={'status':status,'error':None if status=='accepted' else '\n'.join(log.splitlines()[:18]),'constraints':int(match[1]) if match else None,'returncode':proc.returncode,'seconds':round(time.monotonic()-t,3)}
  if a=='rewrite':entry['rewrites']=spec['rewrites']
  result[a]=entry
  if status=='accepted':result['category']='pending_correctness';result['accepted_attempt']=a
  elif status in ['timeout','out of memory']:result['category']='timeout_or_oom'
  else:result['category']='circ_error'
 if result['category']=='circ_error' and (d/'as_written.zok').exists() and 'integer' in (d/'as_written.zok').read_text():
  result['rewrite']={'status':'needs gadget','error':'R1CS rejects native signed integer arithmetic. Preserving scale 2^32 products requires signed wide fixed-point rescaling; custom arithmetic gadgets are forbidden.','constraints':None,'rewrites':[]}
  result['category']='needs_gadget: fixed-point multiplication'
 if result['category'] not in ['pending_correctness','timeout_or_oom'] and m.get('rejection_category'):
  result['category']=m['rejection_category']
  result['rewrite']=m['rejection_rewrite']
 (d/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(p,result['category'],[(k,result[k]['status'] if result[k] else None) for k in ['as_written','rewrite']],flush=True)
if __name__=='__main__':
 selected=sys.argv[1:] or list(M)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(compile_one,selected))
