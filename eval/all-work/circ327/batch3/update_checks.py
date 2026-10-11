from pathlib import Path
p=Path('study.py');s=p.read_text()
s=s.replace("if isinstance(v,dict):v=np.array(v['real'])+1j*np.array(v['imag'])", "if isinstance(v,dict):v=np.array(v['real'])+1j*np.array(v['imag'])\n   else:\n    v=np.array(v);v=v[...,0]+1j*v[...,1]")
s=s.replace("for p in f['parameters']:rec(p['name'],f['inputs'][label][p['name']],p['dtype'])", """for p in f['parameters']:
  v=f['inputs'][label][p['name']]
  if p['dtype']=='complex':
   def comp(key,v,depth):
    if depth:
     for i,el in enumerate(v):comp(key+'.'+str(i),el,depth-1)
    else:
     rec(key+'.re',v[0],'float');rec(key+'.im',v[1],'float')
   comp(p['name'],v,len(p.get('shape',[])))
  elif name=='pythran_factorize_naive':entries.append('('+p['name']+' #x'+format(int(v),'08x')+')')
  else:rec(p['name'],v,p['dtype'])""")
s=s.replace("vals=[]\n  if run.returncode==0:","vals=[]\n  if run.returncode==0 and name=='pythran_factorize_naive':\n   vals=[int(v,16) for v in re.findall(r'0x([0-9a-fA-F]+)',run.stdout)]\n  if run.returncode==0:")
s=s.replace("return np.asarray(v).reshape(-1).tolist()", """a=np.asarray(v)
    if np.iscomplexobj(a):return np.stack([a.real,a.imag],axis=-1).reshape(-1).tolist()
    return a.reshape(-1).tolist()""")
p.write_text(s)
