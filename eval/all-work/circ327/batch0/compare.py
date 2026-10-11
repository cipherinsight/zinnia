import json,pathlib,re,ast,numpy as np
from run import ROOT,P,spec

def parse(text):
 text=re.sub(r'(-?\d+)f\b',r'\1',text)
 text=re.sub(r'\btrue\b','True',text);text=re.sub(r'\bfalse\b','False',text)
 text=re.sub(r'\bOutput\s*\{','{',text)
 text=re.sub(r'\b([A-Za-z_]\w*)\s*:',r'"\1":',text)
 return ast.literal_eval(text.strip())
def signed(x):
 if isinstance(x,list):return [signed(v) for v in x]
 if isinstance(x,dict):return {k:signed(v) for k,v in x.items()}
 return x-P if isinstance(x,int) and x>P//2 else x

def decode(x,s):
 x=signed(x);kind=s.get('decode');scale=s.get('scale',1)
 if kind=='filter':x=x['values'][:x['length']]
 elif kind=='slice':x=[row[:x['width']] for row in x['values']]
 elif kind=='crop':x=[row[:x['width']] for row in x['values'][:x['height']]] if x['height'] else []
 elif kind=='grouping':x=[x['values'][:x['length']],x['sizes'][:x['length']]]
 elif kind=='multitype':return [np.array(x['x'],dtype=float)/2**32,x['y']]
 elif kind=='nussinov':x=[row[:x['size']] for row in x['table'][:x['size']]]
 elif kind=='callgraph':x=[row[:x['size']] for row in x['values'][:x['size']]]
 elif kind=='empty':x=[row[:x['shape'][1]] for row in x['values'][:x['shape'][0]]]
 elif kind=='emin':return [x['converged'],x['energy']/2**32,x['rOH']/2**32,x['thetaHOH']/2**32]
 return np.array(x,dtype=float)/scale

def compare(a,b):
 if isinstance(a,list) and any(isinstance(x,np.ndarray) for x in a):return all(compare(x,y) for x,y in zip(a,b)) and len(a)==len(b)
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 return a.shape==b.shape and bool(np.allclose(a,b,rtol=1e-6,atol=1e-6,equal_nan=True))

def main():
 for n,s in spec.items():
  d=ROOT/n
  if not (d/'result.json').exists():continue
  r=json.load(open(d/'result.json'));mode='rewrite' if s.get('rewrite') else 'as_written'
  if r[mode]['status']!='accepted':continue
  for label in ['A','B']:
   try:
    got=decode(parse((d/f'{mode}.{label}.zxi.log').read_text()),s)
    want=json.load(open(d/f'{label}.expected.json'))
    ok=compare(got,want)
    r['correctness'][label]={'status':'pass' if ok else 'wrong','rtol':1e-6,'atol':1e-6,'decoded_shape':list(np.shape(got)) if s.get('decode')!='multitype' else 'tuple(array[1], scalar)'}
    def serial(v):
     if isinstance(v,np.ndarray):return v.tolist()
     if isinstance(v,list):return [serial(y) for y in v]
     return v
    (d/f'{label}.decoded.json').write_text(json.dumps(serial(got)))
   except Exception as e:r['correctness'][label]={'status':'unverified','error':repr(e)}
  (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
  print(n,[(k,v['status']) for k,v in r['correctness'].items()])
if __name__=='__main__':main()
