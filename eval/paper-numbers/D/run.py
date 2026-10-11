#!/usr/bin/env python3
import os, sys, json, re, subprocess, hashlib
from pathlib import Path
D=Path(__file__).resolve().parent
WT=Path('/home/zhantong/Projects/zinnia-project/wt/npb-hw')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
sys.path[:0]=[str(WT),str(WT/'benchmarking')]
os.environ['ZINNIA_SKIP_RELEASE_BUILD']='1'
os.environ['TMPDIR']=str(D/'tmp'); (D/'tmp').mkdir(exist_ok=True)
FIG=WT.parents[1]/'zinnia-paper/oopsla-2027-r1/fig/tikz/walkthrough.tex'
if len(sys.argv)==1:
    text=FIG.read_text(); body=re.search(r'\np = np.argsort.*?\n\\end\{lstlisting\}',text,re.S)[0].split('\\end{lstlisting}')[0].strip()
    source='def kernel(a: Integer[Array, "5 6"], key: Integer[Array, "6"]):\n'+''.join('    '+s+'\n' for s in body.splitlines())
    (D/'kernel.py').write_text(source)
    a=[[0]*6 for _ in range(5)]
    for i,j in [(2,0),(3,0),(1,3),(3,3),(1,5),(2,5)]: a[i][j]=1
    b=[[0]*6 for _ in range(5)]; b[4][2]=-7
    fixture={'case':'walkthrough','function':'kernel','parameters':[{'name':'a','kind':'array','dtype':'int','shape':[5,6]},{'name':'key','kind':'array','dtype':'int','shape':[6]}], 'inputs':{'A':{'a':a,'key':[20,40,60,10,30,50]},'B':{'a':b,'key':[2,-1,2,0,-1,3]}}}
    (D/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    for mode in ['default','local']:
        for target in ['halo2','noir']:
            env=dict(os.environ)
            for k in list(env):
                if k.startswith('ZINNIA_ABLATE_') or k in ['ZINNIA_FORCE_CONSTRUCTION','ZINNIA_SMT_ENABLE','ZINNIA_REQ_DISABLE']: env.pop(k)
            if mode=='local': env['ZINNIA_ABLATE_JOINT']='1'
            subprocess.run(['systemd-run','--user','--scope','-q','-p','MemoryMax=16G',str(WT/'.venv/bin/python'),str(D/'run.py'),mode,target],env=env,cwd=D,check=True)
    results={f'{m}_{t}':json.loads((D/f'{m}_{t}/result.json').read_text()) for m in ['default','local'] for t in ['halo2','noir']}
    (D/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    sys.exit()
mode,target=sys.argv[1:]; out=D/f'{mode}_{target}'; out.mkdir(exist_ok=True)
trace=out/'selection.jsonl'; trace.unlink(missing_ok=True)
os.environ['ZINNIA_LAYER_REPORT']=str(trace)
from zinnia import ZKCircuit
from zinnia.config.zinnia_config import ZinniaConfig
from benchmarking.harness.fixtures import load_fixture
from benchmarking.harness.acceptance.validate import _mock_publics
from benchmarking.harness.adapters.zinnia_common import describe_compiled,zinnia_publics
from benchmarking.harness.reference import run_kernel_with_arguments
from benchmarking.harness.outputs import reference_outputs,check_outputs
from benchmarking.harness.case_sets import Tolerance
from benchmarking.ablations.common import summarize_report
compiled=ZKCircuit('kernel',(D/'kernel.py').read_text(),config=ZinniaConfig(backend=target)).compile()
stmts=compiled.get_ir_stmts(); (out/'ir.json').write_text(json.dumps(stmts,indent=2))
info=describe_compiled(compiled); (out/'layout.json').write_text(json.dumps(info,indent=2))
fixture=load_fixture(D/'fixture.json'); checks={}
for name in ['A','B']:
    args=fixture.arguments(name)
    mock=compiled.prove(*args,backend='mock') # mock interpreter only; no cryptographic proof or keys
    values,satisfied=_mock_publics(mock)
    returned,after=run_kernel_with_arguments(D/'kernel.py','kernel',args)
    expected,shapes=reference_outputs(returned,after,args,info['output_layout'])
    check=check_outputs(expected,zinnia_publics(values,info),Tolerance(0,0,'exact integer'),expected_shapes=shapes).to_json()
    checks[name]={'satisfied':satisfied,'check':check,'numpy_result':returned.tolist(),'numpy_shape':list(returned.shape),'mock_publics':values}
    assert satisfied and check['status']=='match', checks[name]
result={'program':'walkthrough/kernel','mode':mode,'target':target,'commit':subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip(),'source_sha256':hashlib.sha256((D/'kernel.py').read_bytes()).hexdigest(),'checks':checks}
if target=='halo2':
    from zinnia._zinnia_core import halo2_base_cell_stats
    result['stats']=json.loads(halo2_base_cell_stats(compiled._compiled_ir))
    result['cells']=result['stats']['advice']+result['stats']['lookup']
else:
    from noir_gates import gate_count,NOIR_FPC,NARGO,BB
    source=compiled.emit_structural(*fixture.arguments('B'),backend='noir'); (out/'main.nr').write_text(source)
    result.update(gate_count(source,'noir_fpc',NOIR_FPC))
    result['tools']={p:subprocess.check_output([p,'--version'],text=True,stderr=subprocess.STDOUT).strip() for p in [NARGO,BB]}
result['layer']=summarize_report(trace.read_text().splitlines())
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(mode,target,json.dumps({k:v for k,v in result.items() if k not in ['layer','checks']}),flush=True)
