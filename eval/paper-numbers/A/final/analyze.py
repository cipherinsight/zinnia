import os, sys, json, hashlib, concurrent.futures, subprocess, re
from pathlib import Path
ROOT=Path(os.environ.get('ZINNIA_CHECKOUT','/home/zhantong/zinnia-eval-run/zinnia-sizefix'))
OUT=Path(__file__).resolve().parent
# Acceptance of the final compiler (epoch fz, 2569c9ed, remote); same 214 as the older account.
ACCOUNT=OUT/'accept-merged.json'
sys.path.insert(0,str(ROOT))
def rows(): return [r for r in json.loads(ACCOUNT.read_text())['results'] if r['outcome']=='accepted_correct']
def key(r): return r['dataset']+'__'+r['program']+'__'+r['function']
def worker(i):
 r=rows()[i]; dest=OUT/'records'/key(r); dest.mkdir(parents=True,exist_ok=True)
 os.environ['ZINNIA_LAYER_INPUT']=str(dest/'layer.txt')
 (dest/'layer.txt').unlink(missing_ok=True)
 from benchmarking.harness.kernels import compile_kernel,load_kernel_module
 from zinnia import ZKCircuit
 from zinnia.compile.zinnia_compiler import ZinniaCompiler
 from zinnia.config.zinnia_config import ZinniaConfig
 result={'row':r,'source_sha256':hashlib.sha256((ROOT/r['kernel']).read_bytes()).hexdigest()}
 try:
  compiled=compile_kernel(ROOT/r['kernel'],r['function'])
  result['compiled']=True
  result['layout']=compiled.get_output_layout()
  result['inputs']=[p.export() for p in compiled.program_inputs]
 except BaseException as e: result['compile_error']=repr(e)
 try:
  circuit=ZKCircuit.from_method(getattr(load_kernel_module(ROOT/r['kernel']),r['function']))
  result['circuit_attrs']=list(vars(circuit))
  phase=circuit.analyze_shapes(pretty=False)
  (dest/'phase0.json').write_text(json.dumps(phase))
  result['phase0_rejected']=phase.get('rejected')
 except BaseException as e: result['phase0_error']=repr(e)
 (dest/'result.json').write_text(json.dumps(result,indent=2))
 print(i,key(r),result.get('compile_error','ok'),result.get('phase0_error',result.get('phase0_rejected')),flush=True)
def main():
 (OUT/'records').mkdir(exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('A_JOBS','16'))) as pool:
  list(pool.map(lambda i:subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'worker',str(i)],cwd=OUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}).returncode,range(len(rows()))))
if __name__=='__main__':
 if len(sys.argv)>1: worker(int(sys.argv[2]))
 else: main()
