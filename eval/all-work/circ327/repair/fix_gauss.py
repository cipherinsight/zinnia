import pathlib,json,re
B=pathlib.Path(__file__).resolve().parent;d=B/'pythran_gauss';s=(d/'rewrite.zok').read_text();r=json.load(open(d/'result.json'))
(d/'before-flat.result.json').write_text(json.dumps(r,indent=2));(d/'before-flat.zok').write_text(s)
# Keep the public signature/shape, but store the mutable matrix linearly to avoid tuple-update lowering.
s=s.replace('    for u64 i in 1..5 do','    u64[16] flat_a = [0u64; 16]\n    for u32 row in 0..4 do\n        for u32 col in 0..4 do\n            flat_a[row*4u32+col] = a[row][col]\n        endfor\n    endfor\n    for u64 i in 1..5 do',1)
start=s.index('    for u64 i in 1..5 do');s=s[:start]+re.sub(r'\ba\[([^\[\]]+)\]\[([^\[\]]+)\]',lambda m:'flat_a[('+m[1]+')*4u64+('+m[2]+')]',s[start:])
(d/'rewrite.zok').write_text(s)
m=json.load(open(d/'translation.json'));m['rewrites'].append('Store the mutable 4x4 matrix in a flat sixteen-element buffer after the frontend succeeds but tuple-update optimization crashes; preserve every row/column operation.');(d/'translation.json').write_text(json.dumps(m,indent=2))
