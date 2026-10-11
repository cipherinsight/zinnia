"""Verify the actual timed Cairo binary bytes after lossless typed decoding."""
import hashlib,json,os,pathlib,subprocess,sys
run=pathlib.Path(os.environ['RUN']);wt=run/'zinnia-zk42';out=run/'zk42'
sys.path.insert(0,str(wt/'benchmarking/baselines/cairo'))
import cairo_baseline as cairo
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
decoder=out/'zk42-decode-binary'
results=[]
for raw in sorted((out/'epochs').glob('*/raw/cairo/*.json')):
 rec=json.loads(raw.read_text())
 for tag,sample in [('warmup',rec['warmup']),*((s['tag'],s) for s in rec['samples'])]:
  proof=sample['prove'];binary=pathlib.Path(proof['proof_path'])
  decoded=binary.with_name(binary.stem+'-decoded.json')
  result={'epoch':raw.parents[2].name,'case':rec['case'],'tag':tag,'binary':str(binary),'binary_sha256':sha(binary),'decoder_sha256':sha(decoder)}
  try:
   assert result['binary_sha256']==proof['proof_sha256'],'binary hash differs from timed artifact'
   cmd=[str(decoder),str(binary),str(decoded)]
   subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=120)
   data=json.loads(decoded.read_text())
   regenerated=json.loads(pathlib.Path(proof['details']['json_proof_path']).read_text())
   assert data['claim']==regenerated['claim'],'binary and original JSON claims differ'
   publics={}
   cairo.decode_output(publics,cairo.output_schema(wt/'benchmarking/datasets/ds1000'/rec['case']),cairo.program_output(cairo.proof_output(data)))
   assert publics['publics']==sample['publics']['values'],'binary public outputs differ'
   assert sample['output_check']['status']=='match','original output check failed'
   verify_cmd=[cairo.SCARB,'verify','--proof-file',str(decoded)]
   verify=subprocess.run(verify_cmd,cwd=wt/'benchmarking/baselines/cairo',capture_output=True,text=True,timeout=120)
   result.update({'decode_command':cmd,'verify_command':verify_cmd,'decoded_json_sha256':sha(decoded),'same_proof_as_regenerated_json':data==regenerated,'output_check':'match','verify_returncode':verify.returncode,'verify_stdout':verify.stdout,'verify_stderr':verify.stderr,'verified':verify.returncode==0})
   assert result['verified'],'scarb rejected decoded binary'
  except Exception as exc:
   result.update({'verified':False,'error':str(exc)})
  results.append(result)
  (out/'cairo-binary-verification.json').write_text(json.dumps(results,indent=2))
  print('CAIRO_BINARY',rec['case'],tag,result['verified'],flush=True)
assert len(results)==2*len(list((out/'epochs').glob('*/raw/cairo/*.json'))),len(results)
assert all(r['verified'] and r['output_check']=='match' for r in results)
print('ALL_CAIRO_BINARY_PROOFS_VERIFIED',len(results),flush=True)
