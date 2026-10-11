exec(open('build.py').read().split('# Save source and fixtures')[0].split("add('case0291'")[0])
spec=json.load(open(ROOT/'spec.json'))
# Native signed integer fixed-point operations: no custom R1CS gadget.
add('case0422','''def main(private integer[50] a, private integer[50] b) -> integer[50]:
    integer[99] joined = [0; 99]
    for u32 i in 0..49 do
        joined[i] = b[i+1]
    endfor
    for u32 i in 0..50 do
        joined[i+49] = b[i]
    endfor
    integer[50] result = [0; 50]
    for u32 i in 0..50 do
        integer acc = 0
        for u32 j in 0..50 do
            acc = acc + a[j]*joined[49-i+j]/4294967296
        endfor
        result[i] = acc
    endfor
    return result
''',2**32,reason='Native signed integer multiplication and division by the Q32 scale; NumPy valid correlation reverses offsets when its second input is longer.')
add('case0485','''def main(private integer[5][6][3] A, private integer[3][3] B) -> integer[5][6][3]:
    integer[5][6][3] result = [[[0; 3]; 6]; 5]
    for u32 i in 0..5 do
        for u32 j in 0..6 do
            for u32 k in 0..3 do
                for u32 l in 0..3 do
                    result[i][j][k] = result[i][j][k] + A[i][j][l]*B[l][k]/4294967296
                endfor
            endfor
        endfor
    endfor
    return result
''',2**32)
add('npbench_polybench_atax_atax','''def main(private integer[13][16] A, private integer[16] x) -> integer[16]:
    integer[13] tmp = [0; 13]
    for u32 i in 0..13 do
        for u32 j in 0..16 do
            tmp[i] = tmp[i] + A[i][j]*x[j]/4294967296
        endfor
    endfor
    integer[16] result = [0; 16]
    for u32 j in 0..16 do
        for u32 i in 0..13 do
            result[j] = result[j] + tmp[i]*A[i][j]/4294967296
        endfor
    endfor
    return result
''',2**32)
add('npbench_polybench_gemm_gemm','''def main(private integer alpha, private integer beta, private integer[14][15] C, private integer[14][16] A, private integer[16][15] B) -> integer[14][15]:
    integer[14][16] scaled = [[0; 16]; 14]
    for u32 i in 0..14 do
        for u32 j in 0..16 do
            scaled[i][j] = alpha*A[i][j]/4294967296
        endfor
    endfor
    integer[14][15] product = [[0; 15]; 14]
    for u32 i in 0..14 do
        for u32 j in 0..15 do
            for u32 k in 0..16 do
                product[i][j] = product[i][j] + scaled[i][k]*B[k][j]/4294967296
            endfor
            C[i][j] = product[i][j] + beta*C[i][j]/4294967296
        endfor
    endfor
    return C
''',2**32,reason='Return mutated C as the observable output of the in-place kernel.')
add('npbench_polybench_trmm_trmm','''def main(private integer alpha, private integer[13][13] A, private integer[13][16] B) -> integer[13][16]:
    for u32 i in 0..13 do
        for u32 j in 0..16 do
            integer dot = 0
            for u32 k in i+1..13 do
                dot = dot + A[k][i]*B[k][j]/4294967296
            endfor
            B[i][j] = B[i][j] + dot
        endfor
    endfor
    for u32 i in 0..13 do
        for u32 j in 0..16 do
            B[i][j] = B[i][j]*alpha/4294967296
        endfor
    endfor
    return B
''',2**32,reason='Return mutated B, preserving loop order and dot products.')
add('pythran_rosen','''def main(private integer[16] x) -> integer:
    integer[15] t0 = [0; 15]
    integer[15] t1 = [0; 15]
    for u32 i in 0..15 do
        integer d = x[i+1] - x[i]*x[i]/4294967296
        t0[i] = 100*(d*d/4294967296)
        integer e = 4294967296-x[i]
        t1[i] = e*e/4294967296
    endfor
    integer result = 0
    for u32 i in 0..15 do
        result = result+t0[i]+t1[i]
    endfor
    return result
''',2**32)
add('case0452','''def main(private integer[5][4] X) -> integer[5][4]:
    integer[5] l1 = [0; 5]
    for u32 i in 0..5 do
        for u32 j in 0..4 do
            l1[i] = l1[i] + if X[i][j] < 0 then -X[i][j] else X[i][j] fi
        endfor
    endfor
    integer[5][4] result = [[0; 4]; 5]
    for u32 i in 0..5 do
        for u32 j in 0..4 do
            result[i][j] = X[i][j]*4294967296/l1[i]
        endfor
    endfor
    return result
''',2**32)
add('pythran_lapl3d','''def main(private integer[16][16][3] image) -> integer[14][14][3]:
    integer[14][14][3] out = [[[0; 3]; 14]; 14]
    integer valmax = 0
    for u32 i in 0..14 do
        for u32 j in 0..14 do
            for u32 k in 0..3 do
                integer v = 4*image[i+1][j+1][k]-image[i][j+1][k]-image[i+2][j+1][k]-image[i+1][j][k]-image[i+1][j+2][k]
                out[i][j][k] = if v < 0 then -v else v fi
                valmax = if out[i][j][k] > valmax then out[i][j][k] else valmax fi
            endfor
        endfor
    endfor
    valmax = (if valmax > 4294967296 then valmax else 4294967296 fi)+4
    for u32 i in 0..14 do
        for u32 j in 0..14 do
            for u32 k in 0..3 do
                out[i][j][k] = out[i][j][k]*4294967296/valmax
            endfor
        endfor
    endfor
    return out
''',2**32)
add('pythran_emin','''struct Output {
    bool converged
    integer energy
    integer rOH
    integer thetaHOH
}
def main() -> Output:
    integer c = 21474836
    integer rOH = 42949672960
    integer thetaHOH = 773094113280
    integer E = 0
    integer grOH = 0
    integer grthetaHOH = 0
    for u32 i in 0..1000000 do
        integer r = rOH-4080218931
        integer t = thetaHOH-448824082432
        E = 100*(r*r/4294967296)+50*(t*t/4294967296)
        grOH = 100*r
        grthetaHOH = 100*t
        integer ag = if grOH < 0 then -grOH else grOH fi
        integer at = if grthetaHOH < 0 then -grthetaHOH else grthetaHOH fi
        bool active = ag > 858993459 || at > 8589934592
        rOH = if active then rOH-c*grOH/4294967296 else rOH fi
        thetaHOH = if active then thetaHOH-c*grthetaHOH/4294967296 else thetaHOH fi
    endfor
    integer ag = if grOH < 0 then -grOH else grOH fi
    integer at = if grthetaHOH < 0 then -grthetaHOH else grthetaHOH fi
    return Output {converged: ag > 858993459 || at > 8589934592, energy: E, rOH: rOH, thetaHOH: thetaHOH}
''',2**32,reason='Preserve all 1,000,000 iterations, even after updates cease; constants encoded at Q32.')
add('npbench_polybench_nussinov_nussinov','''struct Output {
    field[14][14] table
    field size
}
def main(private field N, private field[14] seq) -> Output:
    field[14][14] table = [[0; 14]; 14]
    for field rev in 0..14 do
        field i = 13-rev
        for field j in 0..14 do
            bool active = i < N && j < N && j > i
            field jm = if j > 0 then j-1 else 0 fi
            field ip = if i+1 < 14 then i+1 else 0 fi
            field v = table[i][j]
            v = if active && j > 0 && table[i][jm] > v then table[i][jm] else v fi
            v = if active && i+1 < N && table[ip][j] > v then table[ip][j] else v fi
            field score = if seq[i]+seq[j] == 3 then 1 else 0 fi
            field diag = table[ip][jm] + if i+1 < j then score else 0 fi
            v = if active && j > 0 && i+1 < N && diag > v then diag else v fi
            for field k in 0..14 do
                field kp = if k+1 < 14 then k+1 else 0 fi
                field candidate = table[i][k]+table[kp][j]
                v = if active && k > i && k < j && candidate > v then candidate else v fi
            endfor
            table[i][j] = if active then v else table[i][j] fi
        endfor
    endfor
    return Output {table: table, size: N}
''',rewrite=['Pad table to 14x14 and guard loops using N; valid executions require 0 <= N <= 14 because for N > 14 the recurrence reads seq[N-1].'],reason='kernel.py:17-19: table allocation and loops depend on N.',decode='nussinov')
add('pythran_approximated_callgraph','''struct Output {
    field[1024][1024] values
    field size
}
def main(private field size) -> Output:
    field[1024][1024] result = [[0; 1024]; 1024]
    for field i in 0..1024 do
        for field j in 0..1024 do
            result[i][j] = if i < size && j < size then i+j else 0 fi
        endfor
    endfor
    return Output {values: result, size: size}
''',rewrite=['Inline lambda and call; allocate a 1024x1024 padded result, guard both loops and return size, using the explicit 0 <= size <= 1024 precondition.'],reason='kernel.py:15-16: two runtime-sized lists and loop bounds.',decode='callgraph')
(ROOT/'spec.json').write_text(json.dumps(spec,indent=2))
