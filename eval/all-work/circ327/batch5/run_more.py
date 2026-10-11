import study as s
import sys
for p in s.BATCH:
 if p['program'] in s.spec and (len(sys.argv)==1 and not (s.BASE/p['program']/'result.json').exists() or p['program'] in sys.argv[1:]):
  s.process(p)
