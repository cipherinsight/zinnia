import json,pathlib,textwrap
ROOT=pathlib.Path(__file__).resolve().parent
SRC=pathlib.Path('/home/zhantong/Projects/zinnia-project/wt/literal')
BATCH=json.load(open(ROOT.parent/'batch0.json'))
P=52435875175126190479447740508185965837690552500527637822603658699938581184513
H=P//2
spec={}
def add(n,code,scale=1,rewrite=None,reason='',decode=None):
 spec[n]=dict(scale=scale,reason=reason,decode=decode,rewrite=rewrite)
 d=ROOT/n;d.mkdir(exist_ok=True)
 (d/('rewrite.zok' if rewrite else 'as_written.zok')).write_text(textwrap.dedent(code).strip()+'\n')
 if rewrite:(d/'as_written.zok').write_text('// Not expressible as written: '+reason+'\n// See kernel.py and result.json for source line and bounded rewrite.\n')
header=f'const field HALF = {H}\n'
lt='def lt(field x, field y) -> bool:\n    return x + HALF < y + HALF\n'
add('case0291','''def main(private field[2][5][5][2] a) -> field[4]:
    return [2, 5, 5, 2]
''')
add('case0310',header+lt+'''def main(private field[8][6] a) -> field:
    field best = a[0][0]
    field result = 0
    for field i in 0..8 do
        for field j in 0..6 do
            bool take = lt(a[i][j], best)
            result = if take then i * 6 + j else result fi
            best = if take then a[i][j] else best fi
        endfor
    endfor
    return result
''')
add('case0318',header+'''def main(private field[8][6] a, private field[6] permutation) -> field[8][6]:
    field[6] c = [0; 6]
    for field i in 0..6 do
        field ix = if permutation[i] > HALF then permutation[i] + 6 else permutation[i] fi
        c[ix] = i
    endfor
    field[8][6] result = [[0; 6]; 8]
    for field i in 0..8 do
        for field j in 0..6 do
            result[i][j] = a[i][c[j]]
        endfor
    endfor
    return result
''',2**32,reason='Native dynamic scatter and gather; negative NumPy indices normalized by array length. Undefined np.empty entries for non-permutations remain outside deterministic valid executions.')
add('case0337','''def main(private field[8][10] a) -> field[8]:
    field[8][10] flipped = [[0; 10]; 8]
    for field i in 0..8 do
        for field j in 0..10 do
            flipped[i][j] = a[i][9-j]
        endfor
    endfor
    field[8] result = [0; 8]
    for field i in 0..8 do
        result[i] = flipped[i][i]
    endfor
    return result
''',2**32)
add('case0343','''def main(private field[5][6] X) -> field[30]:
    field[30] result = [0; 30]
    for field j in 0..6 do
        for field i in 0..5 do
            result[j*5+i] = X[i][j]
        endfor
    endfor
    return result
''',reason='List append count is syntactically fixed at 30, independent of input values.')
add('case0359','''def main(private field[3][3] a) -> field[3][2]:
    field[3][2] result = [[0; 2]; 3]
    for field i in 0..3 do
        for field j in 0..2 do
            result[i][j] = a[i][j]
        endfor
    endfor
    return result
''',2**32)
add('case0365',header+'''def main(private field[100] a, private field pos, private field element) -> field[101]:
    field p = if pos > HALF then pos + 100 else pos fi
    field[101] result = [0; 101]
    for field i in 0..100 do
        field ix = if i < p then i else i+1 fi
        result[ix] = a[i]
    endfor
    result[p] = element * 4294967296
    return result
''',2**32)
add('case0385','''def main(private field[100][200] a) -> field[5000][2][2]:
    field[5000][2][2] result = [[[0; 2]; 2]; 5000]
    for field j in 0..100 do
        for field i in 0..50 do
            for field k in 0..2 do
                for field l in 0..2 do
                    result[j*50+i][k][l] = a[i*2+k][j*2+l]
                endfor
            endfor
        endfor
    endfor
    return result
''',2**32)
add('case0401','''def main() -> field[3][0]:
    field[0] empty = [0; 0]
    return [empty; 3]
''',2**32)
add('case0437',header+lt+'''def main(private field[10][5] a) -> bool[10][5]:
    bool[10][5] mask = [[false; 5]; 10]
    for field i in 0..10 do
        field m = a[i][0]
        for field j in 1..5 do
            m = if lt(a[i][j], m) then a[i][j] else m fi
        endfor
        for field j in 0..5 do
            mask[i][j] = m == a[i][j]
        endfor
    endfor
    return mask
''')
add('case0477','''def main(private field[4] i) -> field[4][4]:
    field[4][4] result = [[0; 4]; 4]
    for field j in 0..4 do
        result[j][j] = i[j]
    endfor
    return result
''',2**32)
add('case0503',header+'''def main(private field[13][13][5] a, private field[13][13] b) -> field[13][13]:
    field[13][13] result = [[0; 13]; 13]
    for field i in 0..13 do
        for field j in 0..13 do
            field ix = if b[i][j] > HALF then b[i][j] + 5 else b[i][j] fi
            result[i][j] = a[i][j][ix]
        endfor
    endfor
    return result
''',2**32)
add('pythran_multitype','''struct Output {
    field[1] x
    field y
}
def main(private field[1] x, private field y) -> Output:
    field[1] result = if y != 0 then x else [x[0]+x[0]] fi
    return Output {x: result, y: y}
''',decode='multitype')
add('pythran_specialconvolve','''def main(private field[16][16] a) -> field[14][14]:
    field[14][16] rowconvol = [[0; 16]; 14]
    for field i in 0..14 do
        for field j in 0..16 do
            rowconvol[i][j] = a[i+1][j] + a[i][j] + a[i+2][j]
        endfor
    endfor
    field[14][14] colconvol = [[0; 14]; 14]
    for field i in 0..14 do
        for field j in 0..14 do
            colconvol[i][j] = rowconvol[i][j+1] + rowconvol[i][j] + rowconvol[i][j+2] - 9*a[i+1][j+1]
        endfor
    endfor
    return colconvol
''')
# Stable bounded filters, retaining lengths in public outputs.
for n,params,N,setup,pred,body,scale in [
 ('case0443','private field[20] A, private field[3] B',20,'', 'member','''        bool member = false
        for field j in 0..3 do
            member = member || A[i] == B[j]
        endfor
''',1),
 ('case0462','private field[20] a',20,'','keep','''        bool changed = if i == 0 then true else a[i] != a[if i == 0 then 0 else i-1 fi] fi
        bool keep = changed && a[i] != 0
''',1)]:
 val='A[i]' if n=='case0443' else 'a[i]'
 code=f'''struct Output {{
    field[{N}] values
    field length
}}
def main({params}) -> Output:
    field[{N}] result = [0; {N}]
    field count = 0
    for field i in 0..{N} do
'''+body+f'''        field ix = if count < {N} then count else 0 fi
        result[ix] = if {pred} then {val} else result[ix] fi
        count = count + if {pred} then 1 else 0 fi
    endfor
    return Output {{values: result, length: count}}
'''
 add(n,code,scale,rewrite=['Use capacity 20, stable guarded compaction and public length; selection cannot exceed the 20 input entries.'],reason=f'kernel.py:{11 if n=="case0443" else 14}: Boolean selection has value-dependent length.',decode='filter')
add('case0353',header+lt+'''struct Output {
    field[2][3] values
    field length
}
def main(private field[2][3] A, private field[7][3] B) -> Output:
    field[3] dims = [0; 3]
    for field j in 0..3 do
        field m = A[0][j]
        for field i in 0..2 do
            m = if lt(m, A[i][j]) then A[i][j] else m fi
        endfor
        for field i in 0..7 do
            m = if lt(m, B[i][j]) then B[i][j] else m fi
        endfor
        dims[j] = m+1
    endfor
    field[2] ai = [0; 2]
    field[7] bi = [0; 7]
    for field i in 0..2 do
        ai[i] = (A[i][0]*dims[1]+A[i][1])*dims[2]+A[i][2]
    endfor
    for field i in 0..7 do
        bi[i] = (B[i][0]*dims[1]+B[i][1])*dims[2]+B[i][2]
    endfor
    field[2][3] result = [[0; 3]; 2]
    field count = 0
    for field i in 0..2 do
        bool found = false
        for field j in 0..7 do
            found = found || ai[i] == bi[j]
        endfor
        field ix = if count < 2 then count else 0 fi
        result[ix] = if !found then A[i] else result[ix] fi
        count = count + if !found then 1 else 0 fi
    endfor
    return Output {values: result, length: count}
''',rewrite=['Pad to two rows, compact with guards and return public length; filtering two input rows can retain at most two.'],reason='kernel.py:12: Boolean row selection has dynamic length.',decode='filter')
# Python slice clamping uses signed field comparisons; static capacity is input width.
add('case0391',header+lt+'''struct Output {
    field[20][10] values
    field width
}
def main(private field[20][10] a, private field low, private field high) -> Output:
    field lo = if low > HALF then low+10 else low fi
    field hi = if high > HALF then high+10 else high fi
    lo = if lt(lo, 0) then 0 else if lo > 10 then 10 else lo fi fi
    hi = if lt(hi, 0) then 0 else if hi > 10 then 10 else hi fi fi
    field width = if hi > lo then hi-lo else 0 fi
    field[20][10] result = [[0; 10]; 20]
    for field i in 0..20 do
        for field j in 0..10 do
            field ix = if j < width then lo+j else 0 fi
            result[i][j] = if j < width then a[i][ix] else 0 fi
        endfor
    endfor
    return Output {values: result, width: width}
''',2**32,rewrite=['Clamp signed slice endpoints as NumPy does; pad output to 20x10 and return width, since slicing cannot exceed ten columns.'],reason='kernel.py:11: low:high determines output width.',decode='slice')
# Enumerate possible bin widths rather than use a custom integer division gadget.
code=header+lt+'''struct Output {
    field[8] values
    field length
}
def main(private field[8] data, private field bin_size) -> Output:
    field[8] result = [0; 8]
    field count = 0
'''
for b in range(1,9):
 code+=f'    for field i in 0..{8//b} do\n        field m = data[i*{b}]\n        for field j in 1..{b} do\n            m = if lt(data[i*{b}+j], m) then data[i*{b}+j] else m fi\n        endfor\n        result[i] = if bin_size == {b} then m else result[i] fi\n    endfor\n    count = if bin_size == {b} then {8//b} else count fi\n'
code+='    return Output {values: result, length: count}\n'
add('case0415',code,2**32,rewrite=['Pad at most eight bins and return length; enumerate the eight possible nonempty bin widths with guards (larger positive widths produce no bins).'],reason='kernel.py:11: reshape and number of bins depend on bin_size.',decode='filter')
add('pythran_grouping','''struct Output {
    field[5] values
    field[5] sizes
    field length
}
def main(private field[5] values) -> Output:
    field[5] diff = [1, 0, 0, 0, 0]
    for field i in 1..5 do
        diff[i] = values[i]-values[i-1]
    endfor
    field[6] idx = [0; 6]
    field count = 0
    for field i in 0..5 do
        idx[count] = if diff[i] != 0 then i else idx[count] fi
        count = count + if diff[i] != 0 then 1 else 0 fi
    endfor
    idx[count] = 5
    field[5] out = [0; 5]
    field[5] sizes = [0; 5]
    for field i in 0..5 do
        field ix = if i < count then idx[i] else 0 fi
        out[i] = if i < count then values[ix] else 0 fi
        sizes[i] = if i < count then idx[i+1]-idx[i] else 0 fi
    endfor
    return Output {values: out, sizes: sizes, length: count}
''',rewrite=['Pad index array to six and both outputs to five with public group count; five entries imply at most five runs.'],reason='kernel.py:11-12: where and returned arrays have data-dependent length.',decode='grouping')
add('pythran_pselect','''def main(private field n) -> field[2]:
    field[2] result = [0; 2]
    bool[2] ks = [n != 0, n == 0]
    for field i in 0..2 do
        result[i] = if ks[i] then 4294967296 else 8589934592 fi
    endfor
    return result
''',2**32,rewrite=['Inline the two possible callees under their original guards; allocate the list at its fixed two-element length.'],reason='kernel.py:13-16: runtime function-valued variable a is not a Z# type.')
add('case0510','''struct Output {
    field[10][10] values
    field height
    field width
}
def main(private field[10][10] im) -> Output:
    bool[10][10] mask = [[false; 10]; 10]
    bool[10] rows = [false; 10]
    bool[10] cols = [false; 10]
    for field i in 0..10 do
        for field j in 0..10 do
            mask[i][j] = im[i][j] == 0
            rows[i] = rows[i] || !mask[i][j]
            cols[j] = cols[j] || !mask[i][j]
        endfor
    endfor
    field rmin = 10
    field rmax = 0
    field cmin = 10
    field cmax = 0
    for field i in 0..10 do
        rmin = if rows[i] && i < rmin then i else rmin fi
        rmax = if rows[i] then i else rmax fi
        cmin = if cols[i] && i < cmin then i else cmin fi
        cmax = if cols[i] then i else cmax fi
    endfor
    field height = if rmin < 10 then rmax-rmin+1 else 0 fi
    field width = if rmin < 10 then cmax-cmin+1 else 0 fi
    field[10][10] result = [[0; 10]; 10]
    for field i in 0..10 do
        for field j in 0..10 do
            bool active = i < height && j < width
            field r = if active then rmin+i else 0 fi
            field c = if active then cmin+j else 0 fi
            result[i][j] = if active then im[r][c] else 0 fi
        endfor
    endfor
    return Output {values: result, height: height, width: width}
''',2**32,rewrite=['Use ten-entry row/column presence masks and guarded extrema; pad the cropped result to 10x10 with height and width, bounded by the input shape.'],reason='kernel.py:12-17: flatnonzero and crop produce value-dependent lengths.',decode='crop')
# Save source and fixtures for independent reproducibility, without importing Zinnia.
for p in BATCH:
 d=ROOT/p['program'];d.mkdir(exist_ok=True)
 f=SRC/p['kernel'];(d/'kernel.py').write_text(f.read_text())
 fixture=f.parent/'fixture.json'
 if not fixture.exists():fixture=f.parent/f'fixture.{p["function"]}.json'
 if fixture.exists():(d/'fixture.json').write_text(fixture.read_text())
(ROOT/'spec.json').write_text(json.dumps(spec,indent=2))
