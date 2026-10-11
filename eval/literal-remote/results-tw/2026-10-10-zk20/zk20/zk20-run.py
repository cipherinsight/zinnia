import concurrent.futures, json, os, pathlib, subprocess, sys, datetime
run=pathlib.Path(os.environ['RUN'])
wt=run/'zinnia-literal'
out=run/'zk20'
out.mkdir(exist_ok=True)
ids=['ds1000_case'+x for x in '0313 0330 0356 0357 0365 0370 0382 0413 0417 0418 0419 0430 0456 0457 0458 0463 0502'.split()]+['npbench_deep_learning_resnet_resnet__resnet_basicblock','npbench_polybench_floyd_warshall_floyd_warshall__kernel','pythran_ccopy__CCOPY']
if len(sys.argv)>1: ids=sys.argv[1].split(',')
tag=os.environ.get('ZK20_TAG','zk20')
entries={e['case']: e for f in ('ds1000_perf.json','npbench_pythran_perf.json') for e in json.loads((wt/'benchmarking/datasets/case_sets'/f).read_text())['cases']}
cs=out/(tag+'-cases.json')
assert all(entries[c]['status']=='perf' for c in ids)
cs.write_text(json.dumps({'cases':[entries[c] for c in ids]},indent=2))
subprocess.run(['git','rev-parse','HEAD'],cwd=wt,stdout=(out/(tag+'-commit.txt')).open('w'),check=True)
def shard(j):
    cpus=f'{8*j}-{8*j+7}'
    for case in ids[j::4]:
        for phase,systems,reps in [('zpre','zinnia_halo2,zinnia_noir',5),('zkvm','sp1,risc0'+(',cairo' if case.startswith('ds1000') else ''),1),('zpost','zinnia_halo2,zinnia_noir',5)]:
            label=f'{tag}-{case}-{phase}'
            cap='64G' if 'resnet' in case else '30G'
            cmd=['taskset','-c',cpus,str(wt/'.venv/bin/python'),'-m','benchmarking.harness','run','--label',label,'--case-set',str(cs),'--cases',case,'--systems',systems,'--repeats',str(reps),'--threads','8','--memory-max',cap,'--timeout','14400','--results-dir',str(out/'epochs')]
            with (out/(tag+f'-commands-s{j}.jsonl')).open('a') as f: f.write(json.dumps({'time':datetime.datetime.now().isoformat(),'cwd':str(wt),'argv':cmd})+'\n')
            with (out/(label+'.log')).open('w') as log: p=subprocess.run(cmd,cwd=wt,stdout=log,stderr=subprocess.STDOUT)
            print(label,'exit',p.returncode,datetime.datetime.now().isoformat(),flush=True)
            if p.returncode: raise RuntimeError(f'{label} failed: {p.returncode}')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(shard,range(4)))
print('ALL_SHARDS_FINISHED',flush=True)
