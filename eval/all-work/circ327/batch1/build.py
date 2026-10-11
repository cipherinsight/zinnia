import json,pathlib,textwrap,shutil
ROOT=pathlib.Path('/home/zhantong/Projects/zinnia-project/wt/literal')
BATCH=json.load(open('../batch1.json')); META={}; P=52435875175126190479447740508185965837690552500527637822603658699938581184513
SIGN=f'''def lt(field a, field b) -> bool:
    bool an = a > {P//2}
    bool bn = b > {P//2}
    return if an == bn then a < b else an fi

'''
def put(p,code,scale=1,attempt='as_written',rewrites=None,reason='',decode='plain',types=None):
 d=pathlib.Path(p);d.mkdir(exist_ok=True)
 (d/(attempt+'.zok')).write_text('// Reals are encoded at scale 2^32; signed field values use centered residues.\n'+textwrap.dedent(code).strip()+'\n')
 m=META.setdefault(p,{});m.update(scale=scale,decode=decode,types=types or {},reason=reason)
 m[attempt]={'file':attempt+'.zok','rewrites':rewrites or []}
def blocked(p,cat,line,reason,first='not expressible',rewrites=None):
 d=pathlib.Path(p);d.mkdir(exist_ok=True)
 META[p]={'category':cat,'reason':reason,'line':line,'as_written':{'status':first,'error':f'kernel.py:{line}: {reason}','constraints':None},'rewrite':{'status':cat.split(':')[0],'error':reason,'constraints':None,'rewrites':rewrites or []}}
 (d/'assessment.md').write_text(f'# {p}\n\nSource: kernel.py:{line}\n\n{reason}\n\nAs-written: {first}.\nMinimal rewrite: {cat}.\n')
for p in BATCH:
 d=pathlib.Path(p['program']);d.mkdir(exist_ok=True); src=ROOT/p['kernel']; shutil.copyfile(src,d/'kernel.py')
 fs=src.parent/'fixture.json'
 if not fs.exists():fs=src.parent/('fixture.'+p['function']+'.json')
 if fs.exists():shutil.copyfile(fs,d/'fixture.json')
 META[p['program']]={}
blocked('case0295','not_expressible: dynamic shape',11,'The second zeros dimension is a.max()+1; Integer[50] gives no upper bound on values, so no finite padding capacity is implied.')
put('case0301','''
def main(private field[20] A, private field ncol) -> (field[20], field, field):
    field cols = if ncol == -1 then 20 else ncol fi
    field rows = 20 / cols
    return A, rows, cols
''',2**32,'rewrite',['Replace value-dependent reshape by its 20 elements plus public row/column dimensions; 20 source elements bound the output size.'],decode='reshape')
META['case0301']['as_written']={'status':'not expressible','error':'kernel.py:11: reshape has value-dependent dimensions (-1, ncol).','constraints':None}
put('case0311',SIGN+'''
def main(private field[8][6] a) -> (u32, u32):
    field best = a[0][0]
    u32 idx = 0
    for u32 i in 0..8 do
        for u32 j in 0..6 do
            bool bigger = lt(best, a[i][j])
            idx = if bigger then i * 6 + j else idx fi
            best = if bigger then a[i][j] else best fi
        endfor
    endfor
    return idx % 8, idx / 8
''')
put('case0319','''
def main(private field[8][6][8] a, private field[8] permutation) -> field[8][6][8]:
    field[8] c = [0; 8]
    for field i in 0..8 do
        field idx = if permutation[i] > 7 then permutation[i] + 8 else permutation[i] fi
        c[idx] = i
    endfor
    field[8][6][8] result = a
    for u32 i in 0..8 do
        result[i] = a[c[i]]
    endfor
    return result
''',2**32)
blocked('case0325','needs_gadget: sin',11,'np.sin(np.deg2rad(number)) and np.sin(number) need fixed-point sine; no supplied Z# implementation.','needs gadget')
blocked('case0331','not_expressible: dynamic loop',11,'np.gcd expands to a Euclidean remainder loop; unconstrained Integer operands give no declared iteration bound.')
put('case0338','''
def main(private field[8][8] a) -> field[2][8]:
    field[2][8] r = [[0; 8]; 2]
    for u32 i in 0..8 do
        r[0][i] = a[i][i]
        r[1][i] = a[i][7-i]
    endfor
    return r
''',2**32)
put('case0345','''
from "EMBED" import integer_to_field

def main(private integer[8][6] a, private integer col, private integer multiply_number) -> integer[8]:
    integer idx = col - 1
    field ci = integer_to_field(if idx < 0 then idx + 6 else idx fi)
    for u32 i in 0..8 do
        a[i][ci] = a[i][ci] * multiply_number / 4294967296
    endfor
    integer[8] r = [0int; 8]
    integer acc = 0
    for u32 i in 0..8 do
        acc = acc + a[i][ci]
        r[i] = acc
    endfor
    return r
''',2**32)
# bounded Boolean selection
put('case0354','''
struct Output { field[9][3] values; field length }
def main(private field[2][3] A, private field[7][3] B) -> Output:
    field[3] dims = [0; 3]
    for u32 k in 0..3 do
        field m = A[0][k]
        for u32 i in 0..2 do
            m = if A[i][k] > m then A[i][k] else m fi
        endfor
        for u32 j in 0..7 do
            m = if B[j][k] > m then B[j][k] else m fi
        endfor
        dims[k] = m + 1
    endfor
    field[2] ai = [0; 2]
    field[7] bi = [0; 7]
    for u32 i in 0..2 do
        ai[i] = (A[i][0]*dims[1]+A[i][1])*dims[2]+A[i][2]
    endfor
    for u32 j in 0..7 do
        bi[j] = (B[j][0]*dims[1]+B[j][1])*dims[2]+B[j][2]
    endfor
    field[9][3] result = [[0; 3]; 9]
    field count = 0
    for u32 i in 0..2 do
        bool found = false
        for u32 j in 0..7 do
            found = found || ai[i] == bi[j]
        endfor
        result[count] = if !found then A[i] else result[count] fi
        count = count + if !found then 1 else 0 fi
    endfor
    for u32 j in 0..7 do
        bool found = false
        for u32 i in 0..2 do
            found = found || ai[i] == bi[j]
        endfor
        result[count] = if !found then B[j] else result[count] fi
        count = count + if !found then 1 else 0 fi
    endfor
    return Output { values: result, length: count }
''',attempt='rewrite',rewrites=['Pad both Boolean selections and concatenation to 9 rows and return logical length; 2+7 input rows bound the capacity.'],decode='compact')
META['case0354']['as_written']={'status':'not expressible','error':'kernel.py:12-13: Boolean selections and append have value-dependent row counts.','constraints':None}
put('case0360','''
def main(private field[4][4] a) -> field[3][4]:
    return [a[0], a[1], a[3]]
''',2**32)
blocked('case0366','needs_gadget: sort',12,'np.insert with an arbitrary position vector uses stable sorting of the adjusted insertion positions; a custom sorting gadget is prohibited. Output capacity itself is static (105 rows).','needs gadget')
blocked('case0375','needs_gadget: sort',11,'np.sort(x) requires a sorting implementation; no supplied Z# sort routine, and handwritten sorting networks are forbidden.','needs gadget')
put('case0386','''
def main(private field[100][200] a) -> field[19701][2][2]:
    field[19701][2][2] r = [[[0; 2]; 2]; 19701]
    for u32 i in 0..99 do
        for u32 j in 0..199 do
            r[i*199+j] = [[a[i][j], a[i][j+1]], [a[i+1][j], a[i+1][j+1]]]
        endfor
    endfor
    return r
''',2**32)
put('case0392',SIGN+'''
struct Output { field[20][10] values; field length }
def clip(field x) -> field:
    field v = if lt(x, 0) then x + 20 else x fi
    return if lt(v, 0) then 0 else if v > 20 then 20 else v fi fi

def main(private field[20][10] a, private field low, private field high) -> Output:
    field lo = clip(low)
    field hi = clip(high)
    field n = if hi > lo then hi-lo else 0 fi
    field[20][10] r = [[0; 10]; 20]
    for field i in 0..20 do
        field ix = if i < n then lo+i else 0 fi
        r[i] = if i < n then a[ix] else [0; 10] fi
    endfor
    return Output { values: r, length: n }
''',2**32,'rewrite',['Normalize Python slice bounds, pad to 20 rows, and return length; slicing a 20-row array yields at most 20 rows.'],decode='compact')
META['case0392']['as_written']={'status':'not expressible','error':'kernel.py:11: low:high has value-dependent row count.','constraints':None}
put('case0402','''
def main(private field[5] dims, private field[8][9][8][8][8] a, private field[5] index) -> field:
    field result = 0
    field stride = 1
    for u32 i in 0..5 do
        result = result + index[i]*stride
        stride = stride*dims[i]
    endfor
    return result
''')
put('case0409','''
def main(private field[8][6] x, private field[8][6] y) -> field[8][6]:
    field[8][6] x_new = x
    field[8][6] y_new = y
    field[8][6] z = [[0; 6]; 8]
    for u32 i in 0..8 do
        for u32 j in 0..6 do
            z[i][j] = x_new[i][j] + y_new[i][j]
        endfor
    endfor
    return z
''',2**32)
blocked('case0416','needs_gadget: division',11,'The dynamic reshape can be padded to 8x6 (six columns bound the number of bins), but mean(axis=-1) then needs signed fixed-point division by bin_size.',rewrites=['Pad to 8x6 and preserve logical shape, capacity implied by six input columns.'])
blocked('case0425','not_expressible: dynamic shape',11,'np.arange(m) and the 20xm output depend on unconstrained Integer m; no finite output capacity is implied.')
blocked('case0431','needs_gadget: sqrt',11,'np.std computes a square root of a data-dependent variance; fixed-point sqrt is not supplied.','needs gadget')
blocked('case0438','needs_gadget: sqrt',11,'np.corrcoef normalizes covariance by data-dependent standard deviations, requiring sqrt and fixed-point division.','needs gadget')
for p,body in [('case0444','(lt(B[0], A[i]) && lt(A[i], B[1])) || (lt(B[1], A[i]) && lt(A[i], B[2]))'),('case0463','a[i][0] != 0 && (i == 0 || a[i][0] != a[if i > 0 then i-1 else 0 fi][0])')]:
 arg='private field[20] A, private field[3] B' if p=='case0444' else 'private field[20][1] a'
 val='A[i]' if p=='case0444' else 'a[i][0]'
 put(p,SIGN+f'''
struct Output {{ field[20] values; field length }}
def main({arg}) -> Output:
    field[20] r = [0; 20]
    field n = 0
    for u32 i in 0..20 do
        bool keep = {body}
        r[n] = if keep then {val} else r[n] fi
        n = n + if keep then 1 else 0 fi
    endfor
    return Output {{ values: r, length: n }}
''',attempt='rewrite',rewrites=['Pad Boolean selection to 20 entries and return logical length; the input contains only 20 entries.'],decode='compact')
 META[p]['as_written']={'status':'not expressible','error':f'kernel.py:{11 if p=="case0444" else 14}: Boolean selection has value-dependent output length.','constraints':None}
blocked('case0453','needs_gadget: sqrt',11,'Data-dependent L2 norms need square root and fixed-point normalization division.','needs gadget')
put('case0478','''
def main(private field[4][4] a) -> field[4][4]:
    field[4] save = [a[0][0], a[1][1], a[2][2], a[3][3]]
    a = [[0; 4]; 4]
    for u32 i in 0..4 do
        a[i][i] = save[i]
    endfor
    return a
''')
# where indices have dynamic shape in the literal source; the mask rewrite removes index materialization.
put('case0489',SIGN+'''
def main(private field[100][50] arr) -> field[100][50]:
    field[100][50] result = arr
    for u32 i in 0..100 do
        for u32 j in 0..50 do
            arr[i][j] = if lt(result[i][j], -42949672960) then 0 else arr[i][j] fi
            arr[i][j] = if !lt(result[i][j], 64424509440) then 128849018880 else arr[i][j] fi
            arr[i][j] = if !lt(result[i][j], -42949672960) && lt(result[i][j], 64424509440) then arr[i][j]+21474836480 else arr[i][j] fi
        endfor
    endfor
    return arr
''',2**32,'rewrite',['Replace dynamic np.where index arrays and Boolean scatter by guarded element updates over the fixed 100x50 input shape.'])
META['case0489']['as_written']={'status':'not expressible','error':'kernel.py:12-14: np.where index arrays have data-dependent length.','constraints':None}
blocked('case0498','not_expressible: dynamic shape',11,'np.pad output dimensions come from unconstrained Integer[2] shape; source dimensions do not bound padding above.')
put('case0504','''
def main(private field[13][13][5] a, private field[13][13] b) -> field:
    field[13][13] arr = [[0; 13]; 13]
    for u32 i in 0..13 do
        for u32 j in 0..13 do
            field ix = if b[i][j] > 4 then b[i][j]+5 else b[i][j] fi
            arr[i][j] = a[i][j][ix]
        endfor
    endfor
    field result = 0
    for u32 i in 0..13 do
        for u32 j in 0..13 do
            result = result + arr[i][j]
        endfor
    endfor
    return result
''',2**32)
blocked('npbench_azimint_hist_azimint_hist','not_expressible: dynamic shape',11,'Both histograms have npt bins; unconstrained Integer npt has no upper bound, including when there are only 16 samples.')
put('npbench_crc16_crc16','''
def main(private u64[16] data, private u64 poly) -> u64:
    u64 crc = 65535
    for u32 i in 0..16 do
        u64 cur_byte = 255 & data[i]
        for u32 k in 0..8 do
            crc = if ((crc & 1) ^ (cur_byte & 1)) != 0 then (crc >> 1) ^ poly else crc >> 1 fi
            cur_byte = cur_byte >> 1
        endfor
    endfor
    crc = (!crc) & 65535
    crc = (crc << 8) | ((crc >> 8) & 255)
    return crc & 65535
''',types={'data':'u64','poly':'u64'})
blocked('npbench_go_fast_go_fast','needs_gadget: tanh',13,'np.tanh on private diagonal elements requires a fixed-point transcendental routine.','needs gadget')
blocked('npbench_polybench_deriche_deriche','needs_gadget: exp',11,'Private alpha is used in np.exp and real exponentiation, plus fixed-point division.','needs gadget')
blocked('npbench_polybench_seidel_2d_seidel_2d','not_expressible: dynamic loop',12,'The outer loop runs TSTEPS-1 times; only N is bounded by requires, not TSTEPS.')
for p in ['npbench_pythran_arc_distance_arc_distance','pythran_arc_distance']:
 blocked(p,'needs_gadget: sin',14,'Private angle arrays need sin, cos, sqrt, and arctan2; no corresponding fixed-point standard-library routines.','needs gadget')
blocked('pythran_crotg','needs_gadget: sqrt',13,'Complex magnitude and NORM require fixed-point sqrt, followed by division.','needs gadget')
blocked('pythran_lu','needs_gadget: division',19,'M[j,line] / pivot needs signed fixed-point division on private data.','needs gadget')
blocked('pythran_monte_carlo','not_expressible: dynamic loop',18,'numSteps is constrained only to be nonzero; its range loops have no finite declared upper bound. Sine and division are additional obstacles.')
blocked('pythran_sexy_primes','not_expressible: dynamic shape',14,'The primes output and range(9,x+1) depend on unconstrained Integer x; no finite capacity follows from the signature.')
blocked('pythran_train_eq','not_expressible: dynamic shape',36,'err has TrSyms elements with no bound on TrSyms; os may be zero, so the fixed E shape cannot bound the number of iterations.')
blocked('pythran_weights','needs_gadget: division',20,'weights[i] = 1 / total needs fixed-point reciprocal of a private match count; handwritten division or lookup gadgets are forbidden.','needs gadget')
put('pythran_multitype','''
def main(private field n, private field m) -> field:
    assert(n <= 1024)
    assert(m <= 1024)
    return n*m
''')
put('pythran_pythagorean_triples','''
def main(private field[1][3] previous) -> field[3][3]:
    field[9][3] matrices = [[-1,2,2],[-2,1,2],[-2,2,3],[1,2,2],[2,1,2],[2,2,3],[1,-2,2],[2,-1,2],[2,-2,3]]
    field[3][3] r = [[0; 3]; 3]
    for u32 i in 0..9 do
        field acc = 0
        for u32 j in 0..3 do
            acc = acc + matrices[i][j] * previous[0][j]
        endfor
        r[i/3][i%3] = acc
    endfor
    return r
''')
# Native signed fixed point: use Z# integer multiplication and native truncating division, never field division as a substitute.
put('npbench_polybench_bicg_bicg','''
def main(private integer[16][13] A, private integer[13] p, private integer[16] r) -> (integer[13], integer[16]):
    integer[13] s = [0int; 13]
    integer[16] q = [0int; 16]
    for u32 i in 0..16 do
        for u32 j in 0..13 do
            s[j] = s[j] + r[i]*A[i][j]/4294967296
            q[i] = q[i] + A[i][j]*p[j]/4294967296
        endfor
    endfor
    return s, q
''',2**32)
put('pythran_allpairs','''
def main(private integer[16][16] X, private integer[16][16] Y) -> integer[16][16]:
    integer[16][16] D = [[0int; 16]; 16]
    for u32 i in 0..16 do
        for u32 j in 0..16 do
            integer s = 0
            for u32 k in 0..16 do
                integer d = X[i][k] - Y[j][k]
                s = s + d*d/4294967296
            endfor
            D[i][j] = s
        endfor
    endfor
    return D
''',2**32)
put('pythran_euclidean_distance_square','''
def main(private integer[1][3] x1, private integer[5][3] x2) -> integer[1][5]:
    integer[1][5] r = [[0int; 5]; 1]
    for u32 j in 0..5 do
        integer dot = 0
        integer s1 = 0
        integer s2 = 0
        for u32 k in 0..3 do
            dot = dot + x1[0][k]*x2[j][k]/4294967296
            s1 = s1 + x1[0][k]*x1[0][k]/4294967296
            s2 = s2 + x2[j][k]*x2[j][k]/4294967296
        endfor
        r[0][j] = -2*dot+s1+s2
    endfor
    return r
''',2**32)
put('npbench_polybench_gemver_gemver','''
def main(private integer alpha, private integer beta, private integer[16][16] A, private integer[16] u1, private integer[16] v1, private integer[16] u2, private integer[16] v2, private integer[16] w, private integer[16] x, private integer[16] y, private integer[16] z) -> (integer[16][16], integer[16], integer[16]):
    for u32 i in 0..16 do
        for u32 j in 0..16 do
            A[i][j] = A[i][j] + u1[i]*v1[j]/4294967296 + u2[i]*v2[j]/4294967296
        endfor
    endfor
    for u32 j in 0..16 do
        integer s = 0
        for u32 i in 0..16 do
            s = s + (beta*y[i]/4294967296)*A[i][j]/4294967296
        endfor
        x[j] = x[j]+s+z[j]
    endfor
    for u32 i in 0..16 do
        integer s = 0
        for u32 j in 0..16 do
            s = s + (alpha*A[i][j]/4294967296)*x[j]/4294967296
        endfor
        w[i] = w[i]+s
    endfor
    return A, w, x
''',2**32)
put('npbench_polybench_k2mm_k2mm','''
def main(private integer alpha, private integer beta, private integer[14][15] A, private integer[15][15] B, private integer[15][16] C, private integer[14][16] D) -> integer[14][16]:
    integer[14][15] tmp = [[0int; 15]; 14]
    for u32 i in 0..14 do
        for u32 j in 0..15 do
            for u32 k in 0..15 do
                tmp[i][j] = tmp[i][j] + (alpha*A[i][k]/4294967296)*B[k][j]/4294967296
            endfor
        endfor
    endfor
    for u32 i in 0..14 do
        for u32 j in 0..16 do
            integer s = 0
            for u32 k in 0..15 do
                s = s + tmp[i][k]*C[k][j]/4294967296
            endfor
            D[i][j] = s+beta*D[i][j]/4294967296
        endfor
    endfor
    return D
''',2**32)
# Remaining translations are added in build_more.py before serialization.
exec(pathlib.Path('build_more.py').read_text())
pathlib.Path('manifest.json').write_text(json.dumps(META,indent=2)+'\n')
