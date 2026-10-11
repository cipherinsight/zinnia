import concurrent.futures, json, os, pathlib, subprocess, sys, datetime, queue, time, threading
run=pathlib.Path(os.environ['RUN'])
wt=run/'zinnia-zk42'
out=run/'zk42'
out.mkdir(exist_ok=True)
ids=[e['case'] for e in json.loads((run/'zk42/cases.json').read_text())['cases']]
if len(sys.argv)>1: ids=sys.argv[1].split(',')
tag=os.environ.get('ZK42_TAG','zk42')
entries={e['case']: e for f in ('ds1000_perf.json','npbench_pythran_perf.json') for e in json.loads((wt/'benchmarking/datasets/case_sets'/f).read_text())['cases']}
cs=out/(tag+'-cases.json')
assert all(entries[c]['status']=='perf' for c in ids)
cs.write_text(json.dumps({'cases':[entries[c] for c in ids]},indent=2))
subprocess.run(['git','rev-parse','HEAD'],cwd=wt,stdout=(out/(tag+'-commit.txt')).open('w'),check=True)
pending=queue.Queue()
for case in ids: pending.put(case)
failures=[]
def external_jobs(j):
    busy=[]
    if j==3 and not (out/"build-ready").exists(): busy.append("zk42-build")
    for proc in pathlib.Path('/proc').glob('[0-9]*'):
        try:
            parts=(proc/'cmdline').read_bytes().decode().split('\0')
            if not parts or 'python' not in pathlib.Path(parts[0]).name: continue
            argv=' '.join(parts)
            if 'zinnia-literal' not in argv: continue
            # Existing zk20 wrapper will launch its remaining zpost on group 1.
            if j==1 and 'python' in argv and 'zk20-run.py' in argv:
                busy.append(int(proc.name)); continue
            if ('benchmarking.harness run ' in argv or 'zk20-verify-warmups.py' in argv) and 'python' in argv:
                affinity=os.sched_getaffinity(int(proc.name))
                if affinity & set(range(8*j,8*j+8)): busy.append(int(proc.name))
        except (OSError,UnicodeError,ProcessLookupError): pass
    return busy

def shard(j):
    cpus=f'{8*j}-{8*j+7}'
    while True:
        while external_jobs(j):
            if pending.empty(): return
            time.sleep(5)
        try: case=pending.get_nowait()
        except queue.Empty: return
        print('ASSIGN',case,cpus,datetime.datetime.now().isoformat(),flush=True)
        for phase,systems,reps in [('zpre','zinnia_halo2,zinnia_noir',5),('zkvm','sp1,risc0'+(',cairo' if case.startswith('ds1000') else ''),1),('zpost','zinnia_halo2,zinnia_noir',5)]:
            if phase=='zkvm':
                while not (out/'build-ready').exists(): time.sleep(5)
            label=f'{tag}-{case}-{phase}'
            cap='64G' if 'resnet' in case else '30G'
            cmd=['taskset','-c',cpus,str(wt/'.venv/bin/python'),'-m','benchmarking.harness','run','--label',label,'--case-set',str(cs),'--cases',case,'--systems',systems,'--repeats',str(reps),'--threads','8','--memory-max',cap,'--timeout','14400','--results-dir',str(out/'epochs')]
            with (out/(tag+f'-commands-s{j}.jsonl')).open('a') as f: f.write(json.dumps({'time':datetime.datetime.now().isoformat(),'cwd':str(wt),'argv':cmd})+'\n')
            with (out/(label+'.log')).open('w') as log: p=subprocess.run(cmd,cwd=wt,stdout=log,stderr=subprocess.STDOUT)
            print(label,'exit',p.returncode,datetime.datetime.now().isoformat(),flush=True)
            if p.returncode:
                failures.append({'case':case,'phase':phase,'returncode':p.returncode})
                break
        pending.task_done()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(shard,range(4)))
(out/'queue-failures.json').write_text(json.dumps(failures,indent=2))
if failures: raise RuntimeError(str(failures))
print('ALL_SHARDS_FINISHED',flush=True)
