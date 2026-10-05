import copy,json,time,sys
from pathlib import Path
import numpy as np
from zinnia import ZKCircuit
from studies.user_study.tasks import TASKS,signatures,oracle
from studies.user_study.pilots.validate import specialize,instrument,inputs_for
from studies.user_study.validate import python_function
root=Path(__file__).parent
task=next(t for t in TASKS if t.name=='background')
rows=[]
for item in json.loads((root/'inventory.json').read_text()):
  for pi,profile in enumerate(task.profiles):
    row={'name':item['name'],'profile':pi,'status':'fail','python_correct':False,'stage':'python_oracle','checked_inputs':0,'negative_control_passed':False}
    try:
      source=specialize((root/item['path']).read_text(),task,profile)
      cases=inputs_for(task,pi)
      answers=[oracle(task,profile,case) for case in cases]
      fn=python_function(source)
      for case,answer in zip(cases,answers):
        got=np.asarray(fn(*copy.deepcopy(case)))
        assert got.shape==answer.shape
        np.testing.assert_array_equal(got,answer)
      row['python_correct']=True
      row['python_checked_inputs']=len(cases)
      row['stage']='compile_original'
      ZKCircuit.from_source('kernel',source).compile()
      row['stage']='compile_checked'
      circ=ZKCircuit.from_source('kernel',instrument(source,signatures(task,profile)[1]))
      circ.compile()
      row['stage']='execute'
      for case,answer in zip(cases,answers):
        proof=circ.prove(*copy.deepcopy(case),answer,backend='mock')
        assert proof.proof_bytes_hex=='mock_satisfied','correct answer rejected'
        row['checked_inputs']+=1
      row['stage']='negative_control'
      wrong=answers[-1].copy();wrong.flat[0]+=1
      proof=circ.prove(*copy.deepcopy(cases[-1]),wrong,backend='mock')
      assert proof.proof_bytes_hex!='mock_satisfied','incorrect answer accepted'
      row.update(negative_control_passed=True,status='pass',stage='complete')
    except BaseException as exc:
      if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
      row['error']=f'{type(exc).__name__}: {exc}'
    rows.append(row)
    print(json.dumps(row),flush=True)
    (root/'baseline-results.json').write_text(json.dumps(rows,indent=2)+'\n')
