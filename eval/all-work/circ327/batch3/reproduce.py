"""Re-run retained Z# attempts with original caps, then validate every acceptance.
Usage: .venv/bin/python reproduce.py [program ...]
Structural/gadget assessments remain documentary because no complete legal Z# exists.
"""
from study import *

def decoder(name):
 if name=='case0303':return lambda v,l:np.array(v[2:2+int(v[0])*int(v[1])])/S
 if name=='case0362':return lambda v,l:np.array(v[1:]).reshape(8,6)[:,:int(v[0])]/S
 if name=='case0412':return lambda v,l:np.array(v[1:1+int(v[0])])/S
 if name=='case0481':return lambda v,l:v[1:1+int(v[0])]
 if name=='case0507':return lambda v,l:np.array(v[2:]).reshape(5,6)[:int(v[0]),:int(v[1])]
 if name=='case0388':return lambda v,l:np.array(v[3:3+int(v[0])*int(v[1])*int(v[2])])/S
 if name=='case0418':return lambda v,l:np.array(v[1:]).reshape(8,6)[:,:int(v[0])]/S
 if name=='pythran_factorize_naive':return lambda v,l:v[1:1+int(v[0])]
 if name=='pythran_make_decision':return lambda v,l:np.array(v).reshape(-1,2)[:,::-1]/S
 if name=='case0468':
  def windows(v,l):
   out=[]
   for k in range(16):
    b=v[k*18:(k+1)*18];out.append(np.array(b[2:]).reshape(4,4)[:int(b[0]),:int(b[1])])
   return out
  return windows
 return None

if __name__=='__main__':
 selected=sys.argv[1:] or [x['program'] for x in BATCH]
 for name in selected:
  r=json.load(open(ROOT/name/'result.json'))
  for attempt in ['as_written','rewrite']:
   if not (ROOT/name/(attempt+'.zok')).exists():continue
   rewrites=r['rewrite'].get('rewrites',[]) if r['rewrite'] else []
   r=compiled(name,attempt,rewrites,r)
   scale=1 if name in ['case0313','case0321','case0340','case0369','case0434','case0491','pythran_quicksort','pythran_fibo_seq'] else S
   validate(name,attempt,decoder(name),scale)
   r=json.load(open(ROOT/name/'result.json'))
