#!/usr/bin/env python3
"""Compile only, collecting deterministic SMT counters; never measures local times."""
import argparse, concurrent.futures, hashlib, json, os, re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path('/home/zhantong/Projects/zinnia-project')
REPO = ROOT / 'wt/npb-hw'
ACCEPT = ROOT / 'eval-redo/final-analysis/fin/remote/post-freeze/data-fin/accept_final/results.json'
TIMES = ROOT / 'eval-redo/compile-timing/remote/ct_fin.jsonl'
CORPUS = {'ds1000_numpy':'DS-1000', 'npbench':'NPBench', 'pythran':'Pythran'}
def manifest():
    accepted = [r for r in json.loads(ACCEPT.read_text())['results'] if r['outcome']=='accepted_correct']
    rows = [dict(id='acceptance:'+r['dataset']+':'+r['program']+'::'+r['function'], group='acceptance', corpus=CORPUS[r['dataset']], name=r['program']+'::'+r['function'], kernel=r['kernel'], function=r['function']) for r in accepted]
    keys = {(r['program'],r['function']) for r in accepted}
    for r in map(json.loads,TIMES.read_text().splitlines()):
        if r['config']=='default' and r['set']=='perf':
            rows.append(dict(id='perf:'+r['name'],group='perf',corpus='DS-1000', name=r['name'], kernel='benchmarking/'+r['kernel'].split('/benchmarking/')[1], function=r['function'], acceptance_match=(r['name'].removeprefix('ds1000_'),r['function']) in keys))
    assert len({r['id'] for r in rows})==len(rows)
    return sorted(rows,key=lambda r:r['id'])
def child(index):
    sys.path.insert(0,str(REPO))
    from benchmarking.harness.kernels import compile_kernel
    row=manifest()[index]
    try:
        compile_kernel(REPO/row['kernel'],row['function'],backend='halo2')
        result={'ok':True}
    except BaseException as e:
        result={'ok':False,'error':f'{type(e).__name__}: {e}'}
    print('@@RESULT@@'+json.dumps(result),flush=True)
def run_one(i,row,observe=True):
    env={k:v for k,v in os.environ.items() if not k.startswith('ZINNIA_') and k!='LD_PRELOAD'}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',ZINNIA_Z3_TRACE='1',ZINNIA_SMT_LOG_TELEMETRY='1')
    if observe:
        env.update(LD_PRELOAD=str(HERE/'z3_observer.so'),TASK_C_Z3_LIB=str(next((REPO/'.venv/lib').glob('python*/site-packages/z3/lib/libz3.so.4.16'))))
    p=subprocess.run([str(REPO/'.venv/bin/python'),'-B',str(HERE/'collect.py'),'--child',str(i)],cwd=HERE,env=env,capture_output=True,text=True)
    result=[json.loads(l.removeprefix('@@RESULT@@')) for l in p.stdout.splitlines() if l.startswith('@@RESULT@@')]
    out=dict(row,rc=p.returncode,**(result[-1] if result else {'ok':False,'error':'no result'}))
    queries=[]; pending=None; counters={}
    for line in p.stderr.splitlines():
        m=re.match(r'\[task-c-z3\] result=(-?\d+) reason=(.*)',line)
        if m:
            assert pending is None
            pending={'z3_result':int(m[1]),'reason_unknown':m[2]}
        m=re.match(r'\[zinnia-z3\] site=(\S+) result=(\S+) rlimit=(\d+) ms=\S+(.*)',line)
        if m:
            q=dict(site=m[1],result=m[2],steps=int(m[3]),skipped='compilation budget spent' in m[4])
            if observe and not q['skipped']:
                assert pending is not None, line
                q.update(pending); pending=None
            queries.append(q)
        m=re.match(r'\s*(queries_\w+)\s*=\s*(\d+)',line)
        if m: counters[m[1]]=counters.get(m[1],0)+int(m[2])
    assert pending is None
    out.update(queries=queries,resolver_counters=counters,kernel_sha256=hashlib.sha256((REPO/row['kernel']).read_bytes()).hexdigest())
    # Deliberately do not persist stdout/stderr: compiler telemetry contains local durations.
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--child',type=int); ap.add_argument('--jobs',type=int,default=4); ap.add_argument('--verify',action='store_true'); args=ap.parse_args()
    if args.child is not None: child(args.child); sys.exit()
    assert 1<=args.jobs<=8
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='3ebe0558f25592a0c1f8b6cf9bf7ad09f3ddb1d1'
    rows=manifest(); (HERE/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    if args.verify:
        old=[json.loads(l) for l in (HERE/'local_counts.jsonl').read_text().splitlines()]
        # For each corpus, repeat the program with most actual Unknowns without observer.
        chosen=[max((r for r in old if r['corpus']==c and r['group']=='acceptance'),key=lambda r:sum(q['result']=='Unknown' and not q['skipped'] for q in r['queries'])) for c in CORPUS.values()]
        checks=[]
        for r in chosen:
            i=next(i for i,v in enumerate(rows) if v['id']==r['id']); new=run_one(i,rows[i],False)
            old_queries=[{k:v for k,v in q.items() if k not in ('z3_result','reason_unknown')} for q in r['queries']]
            assert old_queries==new['queries'] and r['ok']==new['ok'] and r['resolver_counters']==new['resolver_counters']
            checks.append({'id':r['id'],'identical':True})
        (HERE/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    else:
        subprocess.run(['cc','-shared','-fPIC','-O2','-o',str(HERE/'z3_observer.so'),str(HERE/'z3_observer.c'),'-ldl'],check=True)
        with (HERE/'local_counts.jsonl').open('w') as f, concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futures=[pool.submit(run_one,i,r) for i,r in enumerate(rows)]
            for n,future in enumerate(concurrent.futures.as_completed(futures),1):
                r=future.result(); f.write(json.dumps(r)+'\n'); f.flush()
                if n%20==0 or not r['ok']: print(n,'/',len(rows),r['id'],'ok=',r['ok'],flush=True)
