"""Run every inventoried program through the actual participant execution path."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from ..site.execution import Runner
from .tasks import TASKS, fixtures
from .reference import RUST_VARIANTS
from .validate import environment

def main():
    root=Path(__file__).parent
    jobs=[]
    for t in TASKS:
        cases=[dict(zip(t.arguments,[v.tolist() for v in values])) for p in t.profiles for values in fixtures(t,p)]
        for name,body in t.variants.items():
            code='def kernel('+', '.join(t.arguments)+'):\n'+'\n'.join('    '+line for line in body.splitlines())+'\n'
            jobs.append(('zinnia',t.name,name,code,cases))
        for name,code in RUST_VARIANTS[t.name].items():
            jobs.append(('rust',t.name,name,code,cases))
    with tempfile.TemporaryDirectory(prefix='zinnia-v4-gate-') as runtime:
        runner=Runner(runtime,timeout=180)
        def run(job):
            lang,task,variant,code,cases=job
            result=runner.execute(lang,task,code,cases)
            row={'condition':lang,'task':task,'variant':variant,'source_sha256':hashlib.sha256(code.encode()).hexdigest(),'status':result['status'],'engine':result.get('engine'),'passed':sum(c['passed'] for c in result.get('cases',[])),'total':len(cases),'seconds':result.get('seconds')}
            if row['status']!='passed':row['error']=result
            print(f'{lang}/{task}/{variant}: {row["status"]}',flush=True)
            return row
        try:
            with ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(run,jobs))
        finally:runner.pool.shutdown()
    report={'environment':environment(),'rows':rows,'passed':sum(r['status']=='passed' for r in rows),'total':len(rows)}
    report['site_sha256']={str(p.relative_to(root.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [(root/'content.py'),(root/'reference.py'),(root.parent/'site/content.py'),(root.parent/'site/worker.py'),(root.parent/'site/execution.py')]}
    (root/'site-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{report["passed"]}/{report["total"]} participant-engine implementations passed',flush=True)
    raise SystemExit(0 if report['passed']==report['total'] else 1)

if __name__=='__main__':main()
