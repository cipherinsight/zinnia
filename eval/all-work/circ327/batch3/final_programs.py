from study import *
COMPLEX='''struct Complex {
    field re
    field im
}
def cadd(Complex a, Complex b) -> Complex:
    return Complex {re: a.re+b.re, im: a.im+b.im}
def cmul(Complex a, Complex b) -> Complex:
    return Complex {re: mulq(a.re,b.re)-mulq(a.im,b.im), im: mulq(a.re,b.im)+mulq(a.im,b.re)}
'''
result('pythran_make_decision','needs_gadget: sqrt','kernel.py:14: abs of private complex differences needs a square root as written; the following square allows an exact squared-modulus simplification.','needs_gadget',rewrites=['Use abs(z)^2 = real(z)^2 + imag(z)^2, and compare the common Q64 squared distances without rescaling.','Preserve first argmin and select the original Q32 complex symbol. This needs only native arithmetic/comparisons, no search gadget.'])
program('pythran_make_decision','Complex[16]','''    Complex[16] out = [Complex {re: 0, im: 0}; 16]
    for u32 i in 0..16 do
        field dr0 = E[i].re-symbols[0].re
        field di0 = E[i].im-symbols[0].im
        field best = dr0*dr0+di0*di0
        field bi = 0
        for field j in 1..8 do
            field dr = E[i].re-symbols[j].re
            field di = E[i].im-symbols[j].im
            field dist = dr*dr+di*di
            bool take = dist < best
            bi = if take then j else bi fi
            best = if take then dist else best fi
        endfor
        out[i] = symbols[bi]
    endfor
    return out''','rewrite',extra=COMPLEX,types={'E':'Complex','symbols':'Complex'})
result('pythran_cdotu','not_expressible: dynamic loop','kernel.py:15,24: private range(N); the declared precondition bounds N by 1024. N=0 returns None, otherwise a complex number.',rewrites=['Use 1024 guarded iterations, from the declared precondition, including repeated accesses when INCX or INCY is zero.','Represent complex numbers by real/imaginary Q32 fields and expose a presence tag for the N<=0 return.','Normalize valid negative array indices against length four and guard inactive accesses.'])
program('pythran_cdotu','field[3]','''    Complex ctemp = Complex {re: 0, im: 0}
    field ix = if neg(INCX) then (-N+1)*INCX else 0 fi
    field iy = if neg(INCY) then (-N+1)*INCY else 0 fi
    bool unit = INCX == 1 && INCY == 1
    for field k in 0..1024 do
        bool active = k < N
        field xidx = if unit then k else ix fi
        field yidx = if unit then k else iy fi
        xidx = if neg(xidx) then xidx+4 else xidx fi
        yidx = if neg(yidx) then yidx+4 else yidx fi
        xidx = if active then xidx else 0 fi
        yidx = if active then yidx else 0 fi
        Complex term = cmul(CX[xidx], CY[yidx])
        ctemp = if active then cadd(ctemp, term) else ctemp fi
        ix = if active && !unit then ix+INCX else ix fi
        iy = if active && !unit then iy+INCY else iy fi
    endfor
    return [if N == 0 then 0 else 1 fi, ctemp.re, ctemp.im]''','rewrite',extra=COMPLEX,types={'CX':'Complex','CY':'Complex'})
result('pythran_deriv','not_expressible: dynamic loop','kernel.py:18,27: private range(n), range(n+1); declared n<=1024 and dg has four slots, so every valid call has n<=3.',rewrites=['Use guarded iterations 1..3 and 2..4; dg[n] forces n<=3 on valid calls and n>=0 is explicitly required.','Represent complex arithmetic by pairs of Q32 real fields using native multiply/add and constant rescaling.'])
program('pythran_deriv','Complex[4]','''    Complex zero = Complex {re: 0, im: 0}
    Complex[4] dh2 = [zero; 4]
    Complex ci = Complex {re: 0, im: 4294967296}
    dh1[0] = cmul(cmul(Complex {re: 2147483648, im: 0}, ci), Complex {re: sig, im: 0})
    Complex exp1 = Complex {re: -2147483648, im: 0}
    dh2[0] = alp
    Complex exp2 = Complex {re: -4294967296, im: 0}
    dh3[0] = Complex {re: -2*nu, im: 0}
    Complex exp3 = Complex {re: -4294967296, im: 0}
    for field i in 1..3 do
        bool active = i < n
        dh1[i] = if active then cmul(dh1[i-1],exp1) else dh1[i] fi
        exp1.re = if active then exp1.re-4294967296 else exp1.re fi
        dh2[i] = if active then cmul(dh2[i-1],exp2) else dh2[i] fi
        exp2.re = if active then exp2.re-4294967296 else exp2.re fi
        dh3[i] = if active then cmul(cmul(Complex {re: -nu, im: 0},dh3[i-1]),exp3) else dh3[i] fi
        exp3.re = if active then exp3.re-4294967296 else exp3.re fi
    endfor
    dg[0] = Complex {re: 4294967296, im: 0}
    dg[1] = cadd(cadd(dh1[0],dh2[0]),dh3[0])
    for field i in 2..4 do
        bool active = i <= n
        dg[i] = if active then cadd(cadd(dh1[i-1],dh2[i-1]),dh3[i-1]) else dg[i] fi
        for field j in 1..i do
            Complex term = cmul(cmul(bin[j-1][i-1],cadd(cadd(dh1[j-1],dh2[j-1]),dh3[j-1])),dg[i-j])
            dg[i] = if active then cadd(dg[i],term) else dg[i] fi
        endfor
    endfor
    return dg''','rewrite',extra=COMPLEX,types={'alp':'Complex','dg':'Complex','dh1':'Complex','dh3':'Complex','bin':'Complex'})
# LeNet: every valid reshape followed by fc1w[16,120] has N=2,C_before_fc1=16.
result('npbench_deep_learning_lenet_lenet','not_expressible: dynamic shape','kernel.py:39: reshape(x,(N,C_before_fc1)) uses private dimensions. x has 32 elements and the following matmul has inner dimension 16, forcing C_before_fc1=16,N=2 on valid calls.',rewrites=['Keep N and C_before_fc1 private and assert the only dimensions valid for the following matmul: N=2 and C_before_fc1=16.','Materialize the fixed 2x16 reshape; all convolutions, pooling, biases and dense layers are translated with their original static loop extents.'])
body='''    assert(N == 2)
    assert(C_before_fc1 == 16)
    field[2][12][12][6] c1 = [[[[0; 6]; 12]; 12]; 2]
    for u32 i in 0..12 do
        for u32 j in 0..12 do
            for u32 n in 0..2 do
                for u32 co in 0..6 do
                    field sum = 0
                    for u32 ki in 0..5 do
                        for u32 kj in 0..5 do
                            sum = sum + mulq(input[n][i+ki][j+kj][0],conv1[ki][kj][0][co])
                        endfor
                    endfor
                    sum = sum + conv1bias[co]
                    c1[n][i][j][co] = if neg(sum) then 0 else sum fi
                endfor
            endfor
        endfor
    endfor
    field[2][6][6][6] p1 = [[[[0; 6]; 6]; 6]; 2]
    for u32 i in 0..6 do
        for u32 j in 0..6 do
            for u32 n in 0..2 do
                for u32 co in 0..6 do
                    field best = c1[n][2*i][2*j][co]
                    for u32 ki in 0..2 do
                        for u32 kj in 0..2 do
                            field v = c1[n][2*i+ki][2*j+kj][co]
                            best = if lt(best,v) then v else best fi
                        endfor
                    endfor
                    p1[n][i][j][co] = best
                endfor
            endfor
        endfor
    endfor
    field[2][2][2][16] c2 = [[[[0; 16]; 2]; 2]; 2]
    for u32 i in 0..2 do
        for u32 j in 0..2 do
            for u32 n in 0..2 do
                for u32 co in 0..16 do
                    field sum = 0
                    for u32 ki in 0..5 do
                        for u32 kj in 0..5 do
                            for u32 ci in 0..6 do
                                sum = sum + mulq(p1[n][i+ki][j+kj][ci],conv2[ki][kj][ci][co])
                            endfor
                        endfor
                    endfor
                    sum = sum + conv2bias[co]
                    c2[n][i][j][co] = if neg(sum) then 0 else sum fi
                endfor
            endfor
        endfor
    endfor
    field[2][16] flat = [[0; 16]; 2]
    for u32 n in 0..2 do
        for u32 co in 0..16 do
            field best = c2[n][0][0][co]
            for u32 i in 0..2 do
                for u32 j in 0..2 do
                    field v = c2[n][i][j][co]
                    best = if lt(best,v) then v else best fi
                endfor
            endfor
            flat[n][co] = best
        endfor
    endfor
'''
prev='flat'
for layer,dim,outer in [(1,16,120),(2,120,84),(3,84,10)]:
 dest='f'+str(layer)
 body+=f'''    field[2][{outer}] {dest} = [[0; {outer}]; 2]
    for u32 n in 0..2 do
        for u32 j in 0..{outer} do
            field sum = 0
            for u32 k in 0..{dim} do
                sum = sum + mulq({prev}[n][k], fc{layer}w[k][j])
            endfor
            sum = sum + fc{layer}b[j]
'''
 body+=f'            {dest}[n][j] = '+('if neg(sum) then 0 else sum fi' if layer<3 else 'sum')+'\n        endfor\n    endfor\n'
 prev=dest
body+='    return f3'
program('npbench_deep_learning_lenet_lenet','field[2][10]',body,'rewrite')
