import pathlib,json,sys,shutil
B=pathlib.Path(__file__).resolve().parent
for n in sys.argv[1:]:
 d=B/n;r=json.load(open(d/'result.json'));assert r['category'] in ['timeout_or_oom','circ_error','wrong']
 assert r.get('rewrite') and r['rewrite']['status']!='accepted'
 for name in ['result.json','rewrite.zok','rewrite.compile.log','rewrite.command.json']:
  if (d/name).exists():shutil.copyfile(d/name,d/('field-loop.'+name))
 f=d/'rewrite.zok';s=f.read_text();assert '../native_field_v1' in s;f.write_text(s.replace('../native_field_v1','../native_field'))
