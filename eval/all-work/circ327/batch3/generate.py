from study import *

# Static, direct translations.
for name,rows,cols in [('case0313',8,6),('case0321',5,6)]:
 body=f'''    field best = a[0][0]
    field bi = 0
    field bj = 0
    for field i in 0..{rows} do
        for field j in 0..{cols} do
            bool better = lt(best, a[i][j])
            bi = if better then i else bi fi
            bj = if better then j else bj fi
            best = if better then a[i][j] else best fi
        endfor
    endfor
'''
 body+='    return bi * 6 + bj\n' if name=='case0313' else '    return [bi, bj]\n'
 program(name,'field' if name=='case0313' else 'field[2]',body)
program('case0334','field[50]','''    field[50] result = [0; 50]
    for u32 i in 0..50 do
        result[i] = divc(a[i] + b[i] + c[i], 3)
    endfor
    return result''')
program('case0340','field[30]','''    field[30] result = [0; 30]
    for u32 i in 0..5 do
        for u32 j in 0..6 do
            result[i * 6 + j] = X[i][j]
        endfor
    endfor
    return result''')
program('case0369','bool','''    bool result = true
    for u32 i in 0..3 do
        for u32 j in 0..3 do
            result = result && (absv(a[i][j] - a[i][0]) * 100000 <= absv(a[i][0]))
        endfor
    endfor
    return result''')
program('case0398','field[30]','''    field[30] B = [0; 30]
    for u32 k in 0..30 do
        B[k] = if k == 0 then a * A[k] else a * A[k] + b * B[k - 1] fi
    endfor
    return B''')
program('case0434','field[4][4]','''    for u32 z in 0..2 do
        field r = if neg(zero_rows[z]) then zero_rows[z] + 4 else zero_rows[z] fi
        for u32 j in 0..4 do
            a[r][j] = 0
        endfor
    endfor
    for u32 z in 0..2 do
        field c = if neg(zero_cols[z]) then zero_cols[z] + 4 else zero_cols[z] fi
        for u32 i in 0..4 do
            a[i][c] = 0
        endfor
    endfor
    return a''')
program('case0449','field[3][4][2]','''    field[3][4][2] dists = [[[0; 2]; 4]; 3]
    for u32 i in 0..3 do
        for u32 j in 0..4 do
            dists[i][j][0] = x_dists[i][j]
            dists[i][j][1] = y_dists[i][j]
        endfor
    endfor
    return dists''')
program('case0491','field','''    field result = 0
    for u32 i in 0..6 do
        bool close = absv(s1[i][0] - s2[i][0]) * 100000000 <= 4294967296 + 1000 * absv(s2[i][0])
        result = result + if close then 0 else 1 fi
    endfor
    return result''')
program('case0500','field[18][3]','''    field[18][3] result = [[0; 3]; 18]
    for u32 i in 0..18 do
        for u32 j in 0..3 do
            result[i][j] = a[3 * i + j]
        endfor
    endfor
    return result''')

# Dynamic shape rewrites: fixed capacities derive only from array signature.
result('case0303','not_expressible: dynamic shape','kernel.py:12-13: A[:col] and reshape (-1,ncol) have private dimensions. At most 23 retained elements; output packs row/column metadata and 23 padded elements.',rewrites=['Keep ncol private; pad flattened output to 23 elements with public row and column counts.'])
program('case0303','field[25]','''    field col = divc(23, ncol) * ncol
    field[25] out = [0; 25]
    out[0] = divc(23, ncol)
    out[1] = ncol
    for field i in 0..23 do
        out[i+2] = if i < col then A[i] else 0 fi
    endfor
    return out''','rewrite')
result('case0362','not_expressible: dynamic shape','kernel.py:12-13: boolean filtering and np.delete have value-dependent lengths. Deleting columns from an 8x6 array leaves at most six columns.',rewrites=['Replace filtered deletion indices with a six-column keep mask; compact into an 8x6 padded result and expose surviving column count.'])
program('case0362','field[49]','''    bool[6] keep = [true; 6]
    for u32 d in 0..4 do
        bool active = !lt(6, del_col[d])
        field idx = del_col[d] - 1
        idx = if neg(idx) then idx + 6 else idx fi
        for field j in 0..6 do
            keep[j] = keep[j] && !(active && idx == j)
        endfor
    endfor
    field[49] out = [0; 49]
    field n = 0
    for field j in 0..6 do
        for field i in 0..8 do
            out[1 + i * 6 + n] = if keep[j] then a[i][j] else out[1 + i * 6 + n] fi
        endfor
        n = n + if keep[j] then 1 else 0 fi
    endfor
    out[0] = n
    return out''','rewrite')
result('case0412','not_expressible: dynamic shape','kernel.py:11: Boolean filtering returns between zero and ten elements.',rewrites=['Compact nonnegative entries into ten padded slots and expose length; the input signature has exactly ten elements.'])
program('case0412','field[11]','''    field[11] out = [0; 11]
    field n = 0
    for u32 i in 0..10 do
        bool keep = !neg(x[i])
        out[1 + n] = if keep then x[i] else out[1 + n] fi
        n = n + if keep then 1 else 0 fi
    endfor
    out[0] = n
    return out''','rewrite')
result('case0481','not_expressible: dynamic shape','kernel.py:12: nonzero returns a private-length list of at most nine positions.',rewrites=['Compact matching positions into nine padded slots and expose length; capacity is the signature length nine.'])
program('case0481','field[10]','''    field[10] out = [0; 10]
    field n = 0
    for field i in 0..9 do
        bool keep = x[i] == a && y[i] == b
        out[1 + n] = if keep then i else out[1 + n] fi
        n = n + if keep then 1 else 0 fi
    endfor
    out[0] = n
    return out''','rewrite')
result('case0507','not_expressible: dynamic shape','kernel.py:12-17: nonzero row/column lists and crop have private dimensions. A crop of the 5x6 input contains at most 30 elements.',rewrites=['Track first and last nonzero row/column with static scans; return a padded 5x6 crop and public row/column counts (0,0 for empty).'])
program('case0507','field[32]','''    field rmin = 5
    field rmax = 0
    field cmin = 6
    field cmax = 0
    bool any = false
    for field i in 0..5 do
        for field j in 0..6 do
            bool nz = im[i][j] != 0
            rmin = if nz && i < rmin then i else rmin fi
            rmax = if nz && i > rmax then i else rmax fi
            cmin = if nz && j < cmin then j else cmin fi
            cmax = if nz && j > cmax then j else cmax fi
            any = any || nz
        endfor
    endfor
    field[32] out = [0; 32]
    field nr = if any then rmax - rmin + 1 else 0 fi
    field nc = if any then cmax - cmin + 1 else 0 fi
    out[0] = nr
    out[1] = nc
    for field i in 0..5 do
        for field j in 0..6 do
            field ri = if i < nr then i + rmin else 0 fi
            field cj = if j < nc then j + cmin else 0 fi
            out[2 + i*6+j] = if i < nr && j < nc then im[ri][cj] else 0 fi
        endfor
    endfor
    return out''','rewrite')
# Floating polynomial kernels use multiplication followed by constant Q32 rescale.
program('pythran_harris','field[3][5]','''    field[3][5] out = [[0; 5]; 3]
    for u32 i in 0..3 do
        for u32 j in 0..5 do
            field dx = I[i+1][j+1] - I[i][j+1]
            field dy = I[i+1][j+1] - I[i+1][j]
            field A = mulq(dx, dx)
            field B = mulq(dy, dy)
            field C = mulq(dx, dy)
            field tr = A + B
            field det = mulq(A, B) - mulq(C, C)
            out[i][j] = det - mulq(mulq(214748365, tr), tr)
        endfor
    endfor
    return out''')
body='    for u32 i in 0..16 do\n        field z = X[i] - 4294967296\n        field acc = z\n'
for n in range(2,10):
 exp='z'
 for j in range(n-1):exp=f'mulq({exp}, z)'
 body+=f'        acc = acc {"-" if n%2==0 else "+"} divc({exp}, {n})\n'
body+='        Y[i] = acc\n    endfor\n    return Y'
program('pythran_ln','field[16]',body)
program('npbench_polybench_syr2k_syr2k','field[13][13]','''    for u32 i in 0..13 do
        for u32 j in 0..i+1 do
            C[i][j] = mulq(C[i][j], beta)
        endfor
        for u32 k in 0..9 do
            for u32 j in 0..i+1 do
                C[i][j] = C[i][j] + mulq(mulq(A[j][k], alpha), B[i][k]) + mulq(mulq(B[j][k], alpha), A[i][k])
            endfor
        endfor
    endfor
    return C''')
