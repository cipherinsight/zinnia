"""Adversarial task audit: compile unrewritten programs and check independent oracles."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from ..tasks import TASKS, signatures, oracle
from ..pilots.validate import instrument, inputs_for
from ..validate import environment, python_function


def instrument_module(source, shape):
    """Add the oracle contract to the entry while retaining its source module."""
    tree = ast.parse(source)
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == 'kernel':
            tree.body[index] = ast.parse(instrument(ast.unparse(node), shape)).body[0]
            return ast.unparse(ast.fix_missing_locations(tree))
    raise ValueError('No kernel function found')


def check(task_name, path, profile_index=0, backend='mock', fixture=None):
    from zinnia import ZKCircuit
    task=next(t for t in TASKS if t.name==task_name)
    profile=dict(task.profiles[profile_index])
    if task_name=='smooth':profile['R']=3
    text=Path(path).read_text()
    row={'task':task_name,'path':str(path),'profile':profile_index,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'status':'fail','stage':'prepare','original_compiled':False,'checked_compiled':False,'python_cases':0,'compiled_cases':0,'negative_control':False}
    started=time.monotonic()
    try:
        tree=ast.parse(text)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
        shapes,out_shape=signatures(task,profile)
        if [a.arg for a in fn.args.args]!=list(shapes):raise ValueError('Keep the task argument names')
        for arg in fn.args.args:
            dims=' '.join(map(str,shapes[arg.arg]))
            arg.annotation=ast.parse(f'Integer[Array, "{dims}"]',mode='eval').body
        typed=ast.unparse(ast.fix_missing_locations(tree))
        values=inputs_for(task,profile_index)
        if fixture is not None:values=[values[fixture]]
        expected=[oracle(task,profile,case) for case in values]
        row['stage']='python_oracle'
        reference=python_function(typed)
        for case,answer in zip(values,expected):
            actual=np.asarray(reference(*copy.deepcopy(case)))
            assert actual.shape==answer.shape,'incorrect output shape'
            np.testing.assert_array_equal(actual,answer)
            row['python_cases']+=1
        row['stage']='compile_original'
        ZKCircuit.from_source('kernel',typed).compile()
        row['original_compiled']=True
        row['stage']='compile_checked'
        checked=ZKCircuit.from_source('kernel',instrument_module(typed,out_shape))
        checked.compile()
        row['checked_compiled']=True
        row['stage']='execute'
        for case,answer in zip(values,expected):
            proof=checked.prove(*copy.deepcopy(case),answer,backend=backend)
            assert proof.proof_bytes_hex=='mock_satisfied' if backend=='mock' else checked.verify(proof),'compiled answer rejected'
            row['compiled_cases']+=1
        row['stage']='negative_control'
        wrong=expected[-1].copy();wrong.flat[0]+=1
        proof=checked.prove(*copy.deepcopy(values[-1]),wrong,backend='mock')
        assert proof.proof_bytes_hex!='mock_satisfied','incorrect answer accepted'
        row.update(status='pass',stage='complete',negative_control=True)
    except BaseException as exc:
        if isinstance(exc,(SystemExit,KeyboardInterrupt)):raise
        row['error']=f'{type(exc).__name__}: {exc}'
    row['seconds']=round(time.monotonic()-started,3)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True)
    parser.add_argument('--file',type=Path)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--profile',type=int)
    parser.add_argument('--fixture',type=int)
    parser.add_argument('--backend',choices=['mock','halo2'],default='mock')
    parser.add_argument('--timeout',type=int,default=45)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(check(args.task,args.file,args.profile,args.backend,args.fixture)))
        return
    directory=args.directory or Path(__file__).parent/args.task
    inventory=directory/'inventory.json'
    files=[args.file] if args.file else ([directory/item['path'] for item in json.loads(inventory.read_text())] if inventory.exists() else sorted(directory.glob('*.py')))
    files=[p for p in files if not p.name.startswith('_')]
    task=next(t for t in TASKS if t.name==args.task)
    rows=[]
    for path in files:
        for profile in ([args.profile] if args.profile is not None else range(len(task.profiles))):
            cmd=[sys.executable,'-m','studies.user_study.adversarial.check','--worker','--task',args.task,'--file',str(path),'--profile',str(profile),'--backend',args.backend]
            if args.fixture is not None:cmd+=['--fixture',str(args.fixture)]
            try:
                proc=subprocess.run(cmd,capture_output=True,text=True,timeout=args.timeout)
                row=json.loads(proc.stdout.strip().splitlines()[-1])
            except (subprocess.TimeoutExpired,ValueError,IndexError) as exc:
                row={'path':str(path),'task':args.task,'profile':profile,'status':'fail','stage':'worker','error':str(exc)}
            rows.append(row)
            if row['status']!='pass':print(f'{path.name}/{profile}: {row["stage"]}: {row.get("error","")}',flush=True)
    report={'environment':environment(),'task':args.task,'backend':args.backend,'rows':rows,'passed':sum(r['status']=='pass' for r in rows),'total':len(rows)}
    if args.output:
        args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{report["passed"]}/{report["total"]} passed',flush=True)
    raise SystemExit(0 if rows and report['passed']==len(rows) else 1)


if __name__=='__main__':main()
