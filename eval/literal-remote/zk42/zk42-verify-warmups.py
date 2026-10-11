import json, os, pathlib, sys, subprocess, time
run=pathlib.Path(os.environ['RUN'])
wt=run/'zinnia-zk42'
sys.path.insert(0,str(wt))
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.protocol import Epoch, RunConfig, StepRunner
while any(not p.with_name(p.name.replace('-plan.json','-finished.json')).exists() for p in (run/'zk42').glob('retry*-plan.json')):
    time.sleep(5)
results=[]
for epoch in sorted((run/'zk42/epochs').glob('*')):
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
        (run/'zk42/warmup-verification.json').write_text(json.dumps(results,indent=2))
        print(epoch.name,system,response['ok'] and response.get('result',{}).get('verified'),flush=True)
assert len(results)>=289, f'Unexpectedly few available warmups: {len(results)}'
assert all(r['verify']['ok'] and r['verify']['result']['verified'] and r['output_check']['status']=='match' for r in results), 'warmup check failed'
print('ALL_WARMUPS_VERIFIED',len(results),flush=True)

subprocess.run([sys.executable,str(run/'zk42/zk42-verify-cairo-binaries.py')],check=True)

subprocess.run([sys.executable,str(run/'zk42/zk42-final-audit.py')],check=True)
