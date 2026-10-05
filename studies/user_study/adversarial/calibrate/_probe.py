import sys,json,copy,time,subprocess,hashlib
from pathlib import Path
import numpy as np
from studies.user_study.tasks import TASKS,oracle,signatures
from studies.user_study.pilots.validate import specialize,instrument,inputs_for
from studies.user_study.validate import python_function,environment
P=Path(__file__).parent
T=next(t for t in TASKS if t.name=='calibrate')
def worker(name,pi):
    from zinnia import ZKCircuit
    text=(P/(name+'.py')).read_text()
    r={'name':name,'profile':pi,'status':'fail','stage':'python','python_checked':0,'checked_inputs':0,'negative_control':False}
    start=time.monotonic()
    try:
        text=specialize(text,T,T.profiles[pi]);cases=inputs_for(T,pi)
        # Force thresholds and domain endpoints beyond shared fixtures.
        n=T.profiles[pi]['N']
        for v in [-512,-3,-2,0,124,125,126,4096]:cases.append([np.full(n,v,dtype=np.int64)])
        ans=[oracle(T,T.profiles[pi],c) for c in cases];fn=python_function(text)
        for c,a in zip(cases,ans):
            z=np.asarray(fn(*copy.deepcopy(c)));assert z.shape==a.shape;np.testing.assert_array_equal(z,a);r['python_checked']+=1
        r['stage']='compile_original';ZKCircuit.from_source('kernel',text).compile();r['original_compiled']=True
        r['stage']='compile_checked';checked=ZKCircuit.from_source('kernel',instrument(text,signatures(T,T.profiles[pi])[1]));checked.compile()
        r['stage']='execute'
        for c,a in zip(cases,ans):
            proof=checked.prove(*copy.deepcopy(c),a,backend='mock');assert proof.proof_bytes_hex=='mock_satisfied';r['checked_inputs']+=1
        r['stage']='negative_control';bad=ans[-1].copy();bad.flat[0]+=1
        assert checked.prove(*copy.deepcopy(cases[-1]),bad,backend='mock').proof_bytes_hex!='mock_satisfied';r['negative_control']=True
        r.update(status='pass',stage='complete')
    except BaseException as e:r['error']=type(e).__name__+': '+str(e)
    r['seconds']=round(time.monotonic()-start,3)
    return r
if __name__=='__main__':
    if '--worker' in sys.argv:
        print('RESULT '+json.dumps(worker(sys.argv[-2],int(sys.argv[-1]))));sys.exit()
    inv=json.loads((P/'inventory.json').read_text());rows=[]
    env=environment()
    for item in inv:
        for pi in range(len(T.profiles)):
            try:
                x=subprocess.run([sys.executable,'-m','studies.user_study.adversarial.calibrate._probe','--worker',item['name'],str(pi)],capture_output=True,text=True,timeout=60)
                result=[l[7:] for l in x.stdout.splitlines() if l.startswith('RESULT ')]
                row=json.loads(result[-1]) if result else {'name':item['name'],'profile':pi,'status':'fail','stage':'worker','error':x.stderr[-3000:],'returncode':x.returncode}
            except subprocess.TimeoutExpired:row={'name':item['name'],'profile':pi,'status':'fail','stage':'timeout','error':'60 second timeout'}
            rows.append(row);print(item['name'],pi,row['status'],row.get('error','')[:220],flush=True)
            (P/'baseline.json').write_text(json.dumps({'environment':env,'results':rows},indent=2)+'\n')
