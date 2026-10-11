import json,pathlib
r=pathlib.Path('.')
s=json.load(open('spec.json'))
for n in ['case0365','case0401']:
 p=r/n/'result.json';d=json.load(open(p));(r/n/'original_attempt.json').write_text(json.dumps(d['as_written'],indent=2))
s['case0365'].update(rewrite=['Replace insertion scatter with a guarded gather into the known 101 output positions; inserting one scalar into 100 elements always yields length 101.'],reason='Native insertion scatter exhausted 8 GB; fixed output length permits guarded gather.')
(r/'case0365/rewrite.zok').write_text('''const field HALF = 26217937587563095239723870254092982918845276250263818911301829349969290592256
def main(private field[100] a, private field pos, private field element) -> field[101]:
    field p = if pos > HALF then pos+100 else pos fi
    field[101] result = [0; 101]
    for field i in 0..101 do
        field before = a[if i < 100 then i else 0 fi]
        field after = a[if i > 0 then i-1 else 0 fi]
        result[i] = if i == p then element*4294967296 else if i < p then before else after fi fi
    endfor
    return result
''')
s['case0401'].update(rewrite=['Represent the empty 3x0 array with 3x1 zero padding and public dimensions [3,0].'],reason='CirC rejects zero-element arrays with Empty array.',decode='empty')
(r/'case0401/rewrite.zok').write_text('''struct Output {
    field[3][1] values
    field[2] shape
}
def main() -> Output:
    return Output {values: [[0]; 3], shape: [3, 0]}
''')
(r/'spec.json').write_text(json.dumps(s,indent=2))
