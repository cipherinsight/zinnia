"""Classify exported compiler values and structured pre-lowering IR, never source text."""
import json,gzip,hashlib,collections
from pathlib import Path
from analyze import ROOT,OUT,ACCOUNT,rows,key

def classify(r):
 d=OUT/'records'/key(r)
 stock=json.loads((d/'stock.json').read_text()); audited=json.loads((d/'instrumented.json').read_text())
 assert stock.get('compiled') and audited.get('compiled'),(key(r),stock,audited)
 assert stock==audited,('instrumentation changed compilation',key(r))
 assert (d/'stock.layer.txt').read_bytes()==gzip.open(d/'instrumented.layer.txt.gz','rb').read(),key(r)
 dynamic=[json.loads(x) for x in (d/'dynamic.jsonl').read_text().splitlines()] if (d/'dynamic.jsonl').exists() else []
 runtime=[v for v in dynamic if any(s['constant'] is None for s in [v['length'],v['rank'],*v['shape']])]
 ir=json.load(gzip.open(d/'layer.json.gz','rt')); taint={}; evidence={}; narrow={}; families=collections.Counter()
 def mark(label,s,target=evidence):
  target.setdefault(label,{'count':0,'first_statement':s});target[label]['count']+=1
 for s in ir:
  inst=s['ir_instance'];c=inst['__class__'];data=inst.get('ir_data',{});args=s['arguments'];sid=s['stmt_id']
  taint[sid]=c.startswith(('ReadInteger','ReadFloat','ReadBoolean','ReadHash','Witness','ReadExternal')) or any(taint.get(a,False) for a in args)
  if c in ['ArrayReadIR','ArrayWriteIR']:
   positions=args[1:1+len(data['axes'])]
   if any(taint[a] for a in positions):
    mark(c,s);mark(c,s,narrow)
  elif c in ['RomReadIR','RamReadIR','RamWriteIR']:
   if taint[args[1]]:mark(c,s);mark(c,s,narrow)
  elif c=='ArrayOpIR':
   k=data['kind'];name=k if isinstance(k,str) else next(iter(k));families[name]+=1
   shared=data['shared']; selected=[]
   if name=='Sort': selected=args[:shared] # permutation depends on input data
   elif name=='Window': selected=args[shared-2*k[name]['rank']:shared]
   elif name=='Insert': selected=args[shared-1:shared]
   elif name=='PermutationScatter': selected=args[:shared//2]
   elif name=='MaskWrite': selected=args[-1:]
   if selected and any(taint[a] for a in selected):mark(name,s)
   # A runtime read of the output of any array operator.
   if data['position'] is None and name=='Sort' and any(taint[a] for a in args[shared:]):
    mark('SortRuntimeRead',s);mark('SortRuntimeRead',s,narrow)
 lowering_evidence=dict(evidence)
 compactions=[json.loads(x) for x in (d/'compaction.jsonl').read_text().splitlines()] if (d/'compaction.jsonl').exists() else []
 if dynamic: assert (d/'compaction_check.json').exists(),key(r)
 if compactions: evidence['MaskCompaction']={'count':len(compactions),'first_operation':compactions[0]}
 return {'id':key(r),'corpus':{'ds1000_numpy':'DS-1000','npbench':'NPBench','pythran':'Pythran'}[r['dataset']],
  'program':r['program'],'function':r['function'],'kernel':r['kernel'],
  'runtime_shape':bool(runtime),'dynamic_type':bool(dynamic),'runtime_shape_evidence':runtime,
  'lowering_runtime_index':bool(lowering_evidence),'shape_or_insert':bool(runtime) or 'Insert' in evidence,
  'runtime_index':bool(evidence),'explicit_runtime_address':bool(narrow),'index_evidence':evidence,
  'explicit_address_evidence':narrow,'operator_counts':dict(families),'ir_sha256':stock['ir_sha256']}

def main():
 records=[classify(r) for r in rows()]
 (OUT/'classification.json').write_text(json.dumps(records,indent=2))
 for field in ['runtime_shape','dynamic_type','runtime_index','explicit_runtime_address']:
  print(field,dict(collections.Counter(r['corpus'] for r in records if r[field])))
 for r in records:
  if r['runtime_shape'] or r['runtime_index']:
   print(r['id'],r['runtime_shape'],','.join(r['index_evidence']))
if __name__=='__main__':main()
