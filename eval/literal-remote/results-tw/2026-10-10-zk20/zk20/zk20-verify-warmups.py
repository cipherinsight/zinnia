import json, os, pathlib, sys
os.sched_setaffinity(0, set(range(8, 16)))  # reuse the final measurement shard
run=pathlib.Path(os.environ['RUN'])
wt=run/'zinnia-literal'
sys.path.insert(0,str(wt))
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.protocol import Epoch, RunConfig, StepRunner
results=[]
for epoch in sorted((run/'zk20/epochs').glob('*')):
    cases={e['case']:PerfCase.from_json(e) for e in json.loads((epoch/'case_set.json').read_text())['cases']}
    for path in sorted((epoch/'raw').glob('*/*.json')):
        if path.parent.name.startswith('_'): continue
        rec=json.loads(path.read_text())
        warmup=rec.get('warmup') or {}
        proof=warmup.get('prove') or {}
        if not proof.get('proof_path'): continue
        system=path.parent.name
        case=cases[rec['case']]
        runner=StepRunner(Epoch(epoch),RunConfig(systems=[system],threads=8,step_timeout_s=14400,memory_max='64G'))
        proof_dir=runner.epoch.work(system,case.case)/'proofs/warmup'
        response=runner.run(system,'verify',case,'B','verify-warmup-extra',proof_dir=proof_dir,proof_path=pathlib.Path(proof['proof_path']))
        result={'epoch':epoch.name,'system':system,'case':case.case,'output_check':warmup.get('output_check'),'verify':response}
        results.append(result)
        (run/'zk20/warmup-verification.json').write_text(json.dumps(results,indent=2))
        print(epoch.name,system,response['ok'] and response.get('result',{}).get('verified'),flush=True)
assert all(r['verify']['ok'] and r['verify']['result']['verified'] and r['output_check']['status']=='match' for r in results), 'warmup check failed'
print('ALL_WARMUPS_VERIFIED',len(results),flush=True)
