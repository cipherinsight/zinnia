import pathlib,re
for f in pathlib.Path('.').glob('*/*.zok'):
 if f.parent.name=='probes':continue
 s=f.read_text()
 s=re.sub(r'struct (\w+) \{ ([^{}\n]+) \}',lambda m:'struct '+m[1]+' {\n    '+m[2].replace('; ','\n    ')+'\n}',s)
 m=re.search(r'def main\((.*?)\) -> \((.*?)\):',s)
 if m:
  ts=m[2].split(', '); ret=re.search(r'    return (.*?)\n?$',s)
  # split return expressions, these are simple identifiers or arithmetic.
  vals=ret[1].split(', ')
  struct='struct Result {\n'+''.join(f'    {t} v{i}\n' for i,t in enumerate(ts))+'}\n\n'
  s=s[:ret.start()]+ '    return Result { '+', '.join(f'v{i}: {v}' for i,v in enumerate(vals))+' }\n'
  s=s.replace('-> ('+m[2]+'):', '-> Result:')
  s=struct+s
 f.write_text(s)
