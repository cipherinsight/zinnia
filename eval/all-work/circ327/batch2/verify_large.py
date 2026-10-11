from remaining import *
n='case0387';s=SPECS[n];d=BASE/n
# Preserve the loop translation as written. Equivalent literal construction is a
# minimal allocation rewrite if zxi cannot finish the loop-based version.
(d/'as_written.interpreter_failure.json').write_text(json.dumps(json.load(open(d/'result.json'))['correctness'],indent=2))
exprs=[]
for i in range(50):
 for j in range(100):exprs.append('['+','.join('['+','.join(f'a[{2*i+k}][{2*j+l}]' for l in range(2))+']' for k in range(2))+']')
code='def main(private field[100][200] a) -> field[5000][2][2]:\n    return ['+',\n        '.join(exprs)+']\n'
# Check equivalent static construction with the same full private inputs; retain
# exact as-written compile result, report this check separately if needed.
(d/'verification.zok').write_text(code)
for label in ['A','B']:
 r,t=run_cmd([MEM,'timeout','300',str(CIRC/'zxi'),str(d/'verification.zok'),str(d/(label+'.pin'))],d/('verification.'+label+'.log'))
 print(label,r.returncode,t,flush=True)
