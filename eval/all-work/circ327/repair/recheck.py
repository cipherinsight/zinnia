import run,json,sys
for name in sys.argv[1:]:
 d=run.BASE/name;r=json.load(open(d/'result.json'));m=json.load(open(d/'translation.json'));mode=m['mode'];p=json.load(open(d/'entry.json'))
 assert r[mode]['status']=='accepted'
 r['correctness']=run.check(d,p,mode,(d/(mode+'.zok')).read_text())
 r['category']=('accepted_as_written' if mode=='as_written' else 'accepted_with_rewrite') if all(v['status']=='pass' for v in r['correctness'].values()) else 'wrong'
 (d/'result.json').write_text(json.dumps(r,indent=2));print(name,r['category'],flush=True)
