import pathlib,json,textwrap,re
BASE=pathlib.Path(__file__).resolve().parent
HEAD='from "../native" import neg, absv, ab, lt, divc, mulq, mul, divq, ui, it, integer_to_field\n\n'
def put(n,code,rewrites=None,notes=''):
 d=BASE/n;mode='rewrite' if rewrites else 'as_written'
 (d/(mode+'.zok')).write_text(HEAD+textwrap.dedent(code).strip()+'\n')
 (d/'translation.json').write_text(json.dumps({'mode':mode,'rewrites':rewrites or [],'notes':notes or 'Native u64 Q32 arithmetic with explicit signs, wide product parts, and quotient/remainder digits; all source outputs retained.'},indent=2))
def change(n,fn):
 d=BASE/n;m=json.load(open(d/'translation.json'))['mode'];f=d/(m+'.zok');f.write_text(fn(f.read_text()))
put('pythran_multitype__times','''def main(private u64 n, private u64 m) -> u64:
    return n*m''')
put('pythran_multitype__check2','''struct Output {
    u64[1] x
    bool[1] y
}
def main(private u64[1] x, private bool[1] y) -> Output:
    u64[1] r = if y[0] then x else [x[0]+x[0]] fi
    return Output {x: r, y: y}''')
change('pythran_allpairs__sqr_dists',lambda s:s.replace('X[i][k] - Y[j][k]','X[j][k] - Y[i][k]'))
put('pythran_spectraldns__cross2','''def main(private u64[3][2][5][7][2] c, private u64[3][2][5][7] a, private u64[3][2][5][7][2] b) -> u64[3][2][5][7][2]:
    for u32 i in 0..2 do
        for u32 j in 0..5 do
            for u32 k in 0..7 do
                for u32 ri in 0..2 do
                    c[0][i][j][k][ri] = mulq(a[0][i][j][k],b[2][i][j][k][ri])-mulq(a[2][i][j][k],b[1][i][j][k][ri])
                    c[1][i][j][k][ri] = mulq(a[2][i][j][k],b[0][i][j][k][ri])-mulq(a[0][i][j][k],b[2][i][j][k][ri])
                    c[2][i][j][k][ri] = mulq(a[0][i][j][k],b[1][i][j][k][ri])-mulq(a[1][i][j][k],b[0][i][j][k][ri])
                endfor
                for u32 t in 0..3 do
                    u64 real = c[t][i][j][k][0]
                    c[t][i][j][k][0] = 0u64-c[t][i][j][k][1]
                    c[t][i][j][k][1] = real
                endfor
            endfor
        endfor
    endfor
    return c''')
for n,starta,startb in [('pythran_fibo',0,1),('pythran_fibo_seq',1,1)]:
 body='''def main(private u64 n) -> u64[12]:
    u64[12] a = [0u64; 12]
    u64[12] b = [0u64; 12]
    a[0] = STARTA
    b[0] = STARTB
    for u32 i in 0..1024 do
        u64 carry = 0u64
        u64[12] next = [0u64; 12]
        for u32 j in 0..12 do
            u64 s = a[j]+b[j]
            u64 v = s+carry
            carry = if s < a[j] || v < s then 1u64 else 0u64 fi
            next[j] = v
        endfor
        u64[12] old = a
        a = if it(i) < n then NEXTA else a fi
        b = if it(i) < n then NEXTB else b fi
    endfor
    return a'''.replace('STARTA',str(starta)+'u64').replace('STARTB',str(startb)+'u64').replace('NEXTA','b' if n=='pythran_fibo' else 'next').replace('NEXTB','next' if n=='pythran_fibo' else 'old')
 put(n,body,['Use 1024 guarded iterations from the explicit n<=1024 contract; twelve u64 limbs preserve the entire Fibonacci result (F1026 < 2^768).'+(' Replace repeated recursion by the same Fibonacci recurrence.' if n=='pythran_fibo' else '')])
put('case0347','''def main(private u64[8][6] a, private u64 row, private u64 divide_number) -> u64:
    u64 ix = row-1u64
    ix = if neg(ix) then ix+8u64 else ix fi
    for u32 j in 0..6 do
        a[ix][j] = divq(a[ix][j],divide_number)
    endfor
    u64 result = a[ix][0]
    for u32 j in 1..6 do
        result = mulq(result,a[ix][j])
    endfor
    return result''')
for n in ['case0416','case0418']:
 put(n,'''def main(private u64[8][6] data, private u64 bin_size) -> u64[49]:
    u64 count = 6u64/bin_size
    u64[49] out = [0u64; 49]
    out[0] = count
    for u32 i in 0..8 do
        for u64 j in 0..6 do
            u64 total = 0u64
            for u64 k in 0..6 do
                total = total + if k >= j*bin_size && k < (j+1u64)*bin_size then data[i][INDEX] else 0u64 fi
            endfor
            out[1u64+it(i)*6u64+j] = if j < count then divc(total,bin_size) else 0u64 fi
        endfor
    endfor
    return out'''.replace('INDEX','5u64-k' if n=='case0418' else 'k'),['Pad each row to six means and return the logical column count; at most six nonempty bins fit in each six-element row.'])
put('pythran_weights','''def main(private u64[5][2] input_data, private u64 threshold) -> u64[5]:
    u64[5] weights = [0u64; 5]
    for u32 i in 0..5 do
        u64 total = 0u64
        for u32 j in 0..5 do
            u64 count = 0u64
            for u32 k in 0..2 do
                count = count + if input_data[i][k] == input_data[j][k] then 1u64 else 0u64 fi
            endfor
            total = total + if lt(threshold*2u64,count*4294967296u64) then 1u64 else 0u64 fi
        endfor
        weights[i] = 4294967296u64/total
    endfor
    return weights''')
put('pythran_lap','''def main(private u64[16] var, private u64 dh2) -> u64[16]:
    u64[16] lap = [0u64; 16]
    for u32 i in 0..16 do
        u32 left = if i == 0 then 1u32 else i-1u32 fi
        u64 v = mulq(5726623061u64,var[left])+mulq(5726623061u64,var[(i+1u32)%16u32])
        v = v-mulq(10737418240u64,var[i])-divc(var[(i+14u32)%16u32],12u64)-divc(var[(i+2u32)%16u32],12u64)
        lap[i] = divq(v,dh2)
    endfor
    return lap''')
put('pythran_hasting','''def main(private u64[3] y, private u64 t, private u64 a1, private u64 a2, private u64 b1, private u64 b2, private u64 d1, private u64 d2) -> u64[3]:
    u64 v1 = divq(mulq(mulq(a1,y[0]),y[1]),4294967296u64+mulq(b1,y[0]))
    u64 v2 = divq(mulq(mulq(a2,y[1]),y[2]),4294967296u64+mulq(b2,y[1]))
    return [mulq(y[0],4294967296u64-y[0])-v1, v1-v2-mulq(d1,y[1]), v2-mulq(d2,y[2])]''')
put('npbench_polybench_trisolv_trisolv','''def main(private u64[16][16] L, private u64[16] x, private u64[16] b) -> u64[16]:
    for u32 i in 0..16 do
        u64 s = 0u64
        for u32 j in 0..i do
            s = s+mulq(L[i][j],x[j])
        endfor
        x[i] = divq(b[i]-s,L[i][i])
    endfor
    return x''')
LU='''    for u32 i in 0..15 do
        for u32 j in 0..i do
            u64 s = 0u64
            for u32 k in 0..j do
                s = s+mulq(A[i][k],A[k][j])
            endfor
            A[i][j] = divq(A[i][j]-s,A[j][j])
        endfor
        for u32 j in i..15 do
            u64 s = 0u64
            for u32 k in 0..i do
                s = s+mulq(A[i][k],A[k][j])
            endfor
            A[i][j] = A[i][j]-s
        endfor
    endfor
'''
put('npbench_polybench_lu_lu','def main(private u64[15][15] A) -> u64[15][15]:\n'+LU+'    return A')
put('npbench_polybench_ludcmp_ludcmp','''struct Output {
    u64[15] x
    u64[15] y
}
def main(private u64[15][15] A, private u64[15] b) -> Output:
'''+LU+'''    u64[15] x = [0u64; 15]
    u64[15] y = [0u64; 15]
    for u32 i in 0..15 do
        u64 s = 0u64
        for u32 j in 0..i do
            s = s+mulq(A[i][j],y[j])
        endfor
        y[i] = b[i]-s
    endfor
    for u32 t in 0..15 do
        u32 i = 14u32-t
        u64 s = 0u64
        for u32 j in 0..15 do
            s = s + if j > i then mulq(A[i][j],x[j]) else 0u64 fi
        endfor
        x[i] = divq(y[i]-s,A[i][i])
    endfor
    return Output {x: x, y: y}''')
put('npbench_polybench_durbin_durbin','''def main(private u64[16] r) -> u64[16]:
    u64[16] y = [0u64; 16]
    u64 alpha = 0u64-r[0]
    u64 beta = 4294967296u64
    y[0] = alpha
    for u32 k in 1..16 do
        beta = mulq(beta,4294967296u64-mulq(alpha,alpha))
        u64 dot = 0u64
        for u32 j in 0..k do
            dot = dot+mulq(r[k-1u32-j],y[j])
        endfor
        alpha = 0u64-divq(r[k]+dot,beta)
        u64[16] old = y
        for u32 j in 0..k do
            y[j] = y[j]+mulq(alpha,old[k-1u32-j])
        endfor
        y[k] = alpha
    endfor
    return y''')
for f in ['factorMatrix0','factorMatrix1']:
 put('pythran_lu__'+f,'''def main(private u64[5][5] M) -> u64[5][5]:
    for u32 line in 0..4 do
        u32 cmax = line
        u64 vmax = absv(M[line][line])
        for u32 i in line..5 do
            bool take = absv(M[i][line]) > vmax
            cmax = if take then i else cmax fi
            vmax = if take then absv(M[i][line]) else vmax fi
        endfor
        for u32 j in START..5 do
            u64 t = M[line][j]
            M[line][j] = M[cmax][j]
            M[cmax][j] = t
        endfor
        u64 pivot = M[line][line]
        for u32 j in line+1u32..5 do
            u64 v = divq(M[j][line],pivot)
            for u32 k in line..5 do
                M[j][k] = M[j][k]-mulq(v,M[line][k])
            endfor
        endfor
    endfor
    return M'''.replace('START','0' if f=='factorMatrix0' else 'line'))
put('pythran_tax','''def main(private u64[10] d) -> u64:
    u64 total = 0u64
    u64 denom = 0u64
    for u32 i in 0..10 do
        u64 x = d[i]
        u64 s3 = x-59137326579712u64
        u64 s4 = x-37881611550720u64
        u64 p3 = mulq(s3,9611u64)+1029503661u64
        u64 p4 = mulq(s4,43262u64)+601295421u64
        u64 v = if lt(1100817682853888u64,x) then mulq(x,1932735283u64)-69426127978578u64 else if lt(232173047119872u64,x) then mulq(x,1803886264u64)-36401799938703u64 else if lt(59137326579712u64,x) then mulq(s3,p3)+4035422406246u64 else if lt(37881611550720u64,x) then mulq(s4,p4) else 0u64 fi fi fi fi
        total = total+v
        denom = denom+x
    endfor
    return divq(total,denom)''',['Replace four value-dependent filtered segments by ten guarded entries each; each segment is a subset of the ten inputs.'])
# Correct rounded literals directly from the source decimal constants.
change('pythran_tax',lambda s:s.replace('9611u64',str(round(2.2376e-6*2**32))+'u64').replace('43262u64',str(round(1.00727e-5*2**32))+'u64').replace('1029503661u64',str(round(.2397*2**32))+'u64').replace('69426127978578u64',str(round(16164.53*2**32))+'u64').replace('36401799938703u64',str(round(8475.44*2**32))+'u64').replace('4035422406246u64',str(round(939.57*2**32))+'u64'))
put('npbench_polybench_covariance_covariance','''def main(private u64 M, private u64 float_n, private u64[16][14] data) -> u64[197]:
    u64[14] mean = [0u64; 14]
    for u32 j in 0..14 do
        for u32 i in 0..16 do
            mean[j] = mean[j]+data[i][j]
        endfor
        mean[j] = divc(mean[j],16u64)
        for u32 i in 0..16 do
            data[i][j] = data[i][j]-mean[j]
        endfor
    endfor
    u64[197] cov = [0u64; 197]
    cov[0] = M
    for u32 i in 0..14 do
        for u32 j in i..14 do
            u64 s = 0u64
            for u32 k in 0..16 do
                s = s+mulq(data[k][i],data[k][j])
            endfor
            u64 v = if it(i) < M && it(j) < M then divq(s,float_n-4294967296u64) else 0u64 fi
            cov[1u32+i*14u32+j] = v
            cov[1u32+j*14u32+i] = v
        endfor
    endfor
    return cov''',['Pad cov to 14x14 and return M; valid executions have 0<=M<=14 because nonempty assignments use the fourteen columns of data.'])
put('pythran_cronbach','''def main(private u64[16][16] itemscores) -> u64:
    u64 itemvars = 0u64
    u64[16] tscores = [0u64; 16]
    for u32 i in 0..16 do
        u64 mean = 0u64
        for u32 j in 0..16 do
            mean = mean+itemscores[i][j]
            tscores[j] = tscores[j]+itemscores[i][j]
        endfor
        mean = divc(mean,16u64)
        u64 v = 0u64
        for u32 j in 0..16 do
            u64 d = itemscores[i][j]-mean
            v = v+mulq(d,d)
        endfor
        itemvars = itemvars+divc(v,15u64)
    endfor
    u64 mean = 0u64
    for u32 j in 0..16 do
        mean = mean+tscores[j]
    endfor
    mean = divc(mean,16u64)
    u64 variance = 0u64
    for u32 j in 0..16 do
        u64 d = tscores[j]-mean
        variance = variance+mulq(d,d)
    endfor
    variance = divc(variance,15u64)
    return mulq(4581298449u64,4294967296u64-divq(itemvars,variance))''')
