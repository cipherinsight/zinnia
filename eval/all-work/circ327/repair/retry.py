import pathlib,json,re,sys,shutil
B=pathlib.Path(__file__).resolve().parent
for n in sys.argv[1:]:
 d=B/n;m=json.load(open(d/'translation.json'));mode=m['mode'];r=json.load(open(d/'result.json'))
 (d/'before-native-field.result.json').write_text(json.dumps(r,indent=2));s=(d/(mode+'.zok')).read_text()
 if mode=='rewrite':shutil.copyfile(d/'rewrite.zok',d/'before-native-field.zok')
 s=re.sub(r'from "../native(?:_v1)?"','from "../native_field"',s)
 s=re.sub(r'\bu64\b','field',s);s=re.sub(r'(\d+)u64\b',r'\1',s)
 (d/'rewrite.zok').write_text(s)
 m['mode']='rewrite';m['preserve_attempt']=True;m['rewrites']=m.get('rewrites',[])+['Use wide native field products, then native bit unpacking for the Q32 shift; retain native unsigned division and all loop bounds. This avoids bit-blasting every accumulation into a u64 adder.']
 (d/'translation.json').write_text(json.dumps(m,indent=2))
