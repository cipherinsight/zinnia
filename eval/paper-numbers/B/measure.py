import os,sys,json,ast,hashlib,subprocess,concurrent.futures,traceback
from pathlib import Path
ROOT=Path('/home/zhantong/Projects/zinnia-project/wt/npb-hw')
HERE=Path(__file__).resolve().parent
ACCEPT=Path('/home/zhantong/Projects/zinnia-project/zinnia-src/eval/final-analysis/fin/remote/post-freeze/data-fin/accept_final/results.json')
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from benchmarking.harness.case_sets import load_acceptance_programs
from benchmarking.harness.fixtures import load_fixture
from benchmarking.harness.kernels import compile_kernel

def entries():
    accepted={(r['dataset'],r['program'],r['function']) for r in json.loads(ACCEPT.read_text())['results'] if r['outcome']=='accepted_correct'}
    return [p for p in load_acceptance_programs(ROOT/'benchmarking/datasets/case_sets/acceptance.json') if (p.dataset,p.program,p.function) in accepted]
def ident(p):return f'{p.dataset}__{p.program}__{p.function}'
def census(p):
    c=compile_kernel(p.kernel_path,p.function)
    return dict(id=ident(p),kernel=str(p.kernel_path),fixture=str(p.fixture_path),function=p.function,layout=c.get_output_layout(),ir_count=len(c.get_ir_stmts()))
def main():
    ps=entries(); out=HERE/'census';out.mkdir(exist_ok=True)
    if len(sys.argv)>1 and sys.argv[1]=='--one':
        p=next(p for p in ps if ident(p)==sys.argv[2])
        try:r=census(p)
        except BaseException as e:r=dict(id=ident(p),error=str(e),traceback=traceback.format_exc())
        (out/(ident(p)+'.json')).write_text(json.dumps(r,indent=2));return
    def run(p):
        target=out/(ident(p)+'.json')
        if target.exists():return
        with (out/(ident(p)+'.log')).open('w') as log:
            try:
                r=subprocess.run(['systemd-run','--user','--scope','-q','-p','MemoryMax=16G',sys.executable,str(HERE/'measure.py'),'--one',ident(p)],stdout=log,stderr=log,timeout=600,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                if not target.exists():target.write_text(json.dumps(dict(id=ident(p),error=f'exit {r.returncode}')))
            except subprocess.TimeoutExpired:target.write_text(json.dumps(dict(id=ident(p),error='resource timeout 600s')))
        print(ident(p),flush=True)
    print('accepted',len(ps),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,ps))
if __name__=='__main__':main()
