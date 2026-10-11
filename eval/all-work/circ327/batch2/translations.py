from study import *
blocked('case0296','not_expressible: dynamic shape','kernel.py:11: width a.max()+1 has no bound in Integer[50]; fixture maxima are not domain bounds.')
blocked('case0411','not_expressible: dynamic shape','kernel.py:26: both crop dimensions high_index-low_index are unbounded integers; 5x5 image does not bound padded crop size.')
blocked('case0426','not_expressible: dynamic shape','kernel.py:11: np.arange(m) and the 20 by m output have no bound on m.')
blocked('case0499','not_expressible: dynamic shape','kernel.py:15-16,20: requested padding shape consists of two unbounded integer inputs.')
blocked('npbench_azimint_naive_azimint_naive','not_expressible: dynamic shape','kernel.py:12: output length npt is unbounded; npt != 0 gives no capacity bound. Mean also needs fixed-point division.')
blocked('npbench_mandelbrot1_mandelbrot1','not_expressible: dynamic shape','kernel.py:10-14: xn and yn are unbounded input dimensions, and maxiter has no domain bound.')
blocked('pythran_shallow_water','not_expressible: dynamic shape','kernel.py:10,49-51: H and W are unrestricted input dimensions; I also supplies an unbounded iteration count.')
blocked('pythran_train_eq','not_expressible: dynamic shape','kernel.py:13-14: TrSyms has no upper bound; os=0 permits arbitrarily many valid equal-sized windows, so E.shape does not bound it.')
blocked('pythran_collatz_modified','not_expressible: dynamic loop','kernel.py:10,13: both while True loops have no syntactic or declared-domain bound; cannot invent a Collatz stopping bound.')
blocked('pythran_euler_challenge14','not_expressible: dynamic loop','kernel.py:29: MAX_NUM has no declared bound; recursive Collatz series_length at line 20 also lacks a bound.')
blocked('pythran_guerre','not_expressible: dynamic loop','kernel.py:24: nter is unrestricted, so the convergence loop has no domain-implied constant bound. Complex division and magnitude also require gadgets.')
blocked('pythran_babylonian','not_expressible: dynamic loop','kernel.py:12: Newton while loop and seen set grow with an unrestricted Integer input; no finite constant iteration bound follows from this signature.')
for n,line,g in [('case0326',11,'asin'),('case0355',11,'sort'),('case0381',11,'sort'),('case0454',12,'division'),('npbench_polybench_cholesky2_cholesky2',10,'sqrt'),('pythran_ising',42,'exp'),('pythran_make_circle_two_points',44,'sqrt'),('pythran_pivot',11,'sqrt'),('pythran_ldpc_decoder',16,'log')]:
 blocked(n,'needs_gadget: '+g,f'kernel.py:{line}: requires {g}; no matching fixed-point intrinsic in installed Z# stdlib. The brief forbids hand-writing this gadget.')
blocked('case0332','needs_gadget: gcd','kernel.py:11: np.gcd needs Euclidean iteration; unrestricted Integer signature gives no fixed iteration bound. No gcd intrinsic in installed stdlib.')
blocked('case0417','needs_gadget: division','kernel.py:12: dynamic intermediate/output can be padded to 8 (input length), but mean divides signed fixed-point sums by private bin_size; requires fixed-point division.', ['Pad output to 8 with logical length; guard bins by input length.'])
blocked('pythran_queens_numba','needs_gadget: search','kernel.py:21-33: recursive backtracking search with append/pop. Eight board rows bound live queens to 8; a padded stack still requires a search gadget, forbidden by brief.', ['Capacity 8 for queens and explicit success tag; board has only eight mutually nonattacking rows.'])
# Plain fixed-shape data movement and reductions. Signed field values use centered residues.
spec('case0312','''def main(private field[8][6] a) -> field[2]:
    field best = a[0][0]
    field[2] result = [0; 2]
    for u32 i in 0..8 do
        for u32 j in 0..6 do
            bool take = lt(best, a[i][j])
            result = if take then [u32_to_field(i), u32_to_field(j)] else result fi
            best = if take then a[i][j] else best fi
        endfor
    endfor
    return result
''','int',reason='Argmax scan uses strict comparison to preserve first tie, followed by static-shape unravel.')
spec('case0320',SPECS['case0312']['code'].split('def main',1)[1].join(['def main','']).replace('[8][6]','[5][6]').replace('0..8','0..5').replace('lt(best, a[i][j])','lt(a[i][j], best)'),'int',reason='Argmin scan preserves first tie; unravel uses the declared 5x6 shape.')
spec('case0339','''def main(private field[8][10] a) -> field[2][8]:
    field[8][8] b = [[0; 8]; 8]
    for u32 i in 0..8 do
        for u32 j in 0..8 do
            b[i][j] = a[i][j]
        endfor
    endfor
    field[2][8] result = [[0; 8]; 2]
    for u32 i in 0..8 do
        result[0][i] = b[i][i]
        result[1][i] = b[i][7-i]
    endfor
    return result
''',reason='Static slicing, diagonal extraction, fliplr, and vstack.')
spec('case0361','''def main(private field[6][6] a) -> field[6][4]:
    field[6][4] result = [[0; 4]; 6]
    u32[4] kept = [1,3,4,5]
    for u32 i in 0..6 do
        for u32 j in 0..4 do
            result[i][j] = a[i][kept[j]]
        endfor
    endfor
    return result
''',reason='Deleted columns [0,2] are source constants; output is statically 6x4.')
spec('case0368','''def main(private field[3][3] a) -> bool:
    bool result = true
    for u32 i in 0..3 do
        for u32 j in 0..3 do
            result = result && !lt(ab(a[0][j]), ab(a[i][j]-a[0][j])*100000)
        endfor
    endfor
    return result
''','bool',reason='isclose with atol=0 and rtol=1e-5: exact rational constant comparison in Q32 units, then all.')
spec('case0387','''def main(private field[100][200] a) -> field[5000][2][2]:
    field[5000][2][2] result = [[[0; 2]; 2]; 5000]
    for u32 i in 0..50 do
        for u32 j in 0..100 do
            for u32 k in 0..2 do
                for u32 l in 0..2 do
                    result[i*100+j][k][l] = a[i*2+k][j*2+l]
                endfor
            endfor
        endfor
    endfor
    return result
''',reason='Static reshape/swapaxes/reshape mapped in C order; all 20,000 output elements retained.')
spec('case0403','''def main(private field[5] dims, private field[8][9][8][8][8] a, private field[5] index) -> field:
    field result = 0
    for u32 i in 0..5 do
        result = result*dims[i]+index[i]
    endfor
    return result
''','int',reason='C-order ravel_multi_index recurrence; a is retained as an unused private input, as in source.')
spec('case0433','''def main(private field[4][4] a, private field zero_rows, private field zero_cols) -> field[4][4]:
    u32 r = idx(if neg(zero_rows) then zero_rows+4 else zero_rows fi)
    u32 c = idx(if neg(zero_cols) then zero_cols+4 else zero_cols fi)
    for u32 j in 0..4 do
        a[r][j] = 0
    endfor
    for u32 i in 0..4 do
        a[i][c] = 0
    endfor
    return a
''','int',reason='Private dynamic row/column writes preserved, including valid NumPy negative indices.')
spec('case0448','''def main(private field[3][4] x_dists, private field[3][4] y_dists) -> field[3][4][2]:
    field[3][4][2] result = [[[0; 2]; 4]; 3]
    for u32 i in 0..3 do
        for u32 j in 0..4 do
            result[i][j][0] = x_dists[i][j]
            result[i][j][1] = y_dists[i][j]
        endfor
    endfor
    return result
''',reason='Static transpose/vstack/transpose retaining both components.')
spec('case0480','''def main(private field[8] x, private field[8] y, private field a, private field b) -> field:
    u32 result = 0
    bool best = false
    for u32 i in 0..8 do
        bool hit = x[i] == a && y[i] == b
        result = if hit && !best then i else result fi
        best = best || hit
    endfor
    return if x[result] != a || y[result] != b then -1 else u32_to_field(result) fi
''','int',reason='Boolean argmax scan then original private-index verification and -1 fallback.')
spec('case0490','''def main(private field[5][50] arr, private field[5] n1, private field[5] n2) -> field[5][50]:
    for u32 i in 0..5 do
        field[50] temp = arr[i]
        for u32 j in 0..50 do
            arr[i][j] = if lt(temp[j], n1[i]*4294967296) then 0 else arr[i][j] fi
        endfor
        for u32 j in 0..50 do
            arr[i][j] = if !lt(temp[j], n2[i]*4294967296) then 128849018880 else arr[i][j] fi
        endfor
        for u32 j in 0..50 do
            arr[i][j] = if !lt(temp[j], n1[i]*4294967296) && lt(temp[j], n2[i]*4294967296) then arr[i][j]+21474836480 else arr[i][j] fi
        endfor
    endfor
    return arr
''',rewrite=['Replace value-dependent np.where index arrays with guarded writes over all 50 positions in each row.'],reason='kernel.py:13-15: np.where produces dynamic index vectors; each has capacity 50 from the row shape.')
spec('case0505','''def main(private field[13][13][5] a, private field[13][13] b) -> field:
    field[13][13] arr = [[0; 13]; 13]
    for u32 i in 0..13 do
        for u32 j in 0..13 do
            u32 k = idx(if neg(b[i][j]) then b[i][j]+5 else b[i][j] fi)
            arr[i][j] = a[i][j][k]
        endfor
    endfor
    field total = 0
    field selected = 0
    for u32 i in 0..13 do
        for u32 j in 0..13 do
            for u32 k in 0..5 do
                total = total+a[i][j][k]
            endfor
            selected = selected+arr[i][j]
        endfor
    endfor
    return total-selected
''',reason='Private take_along_axis index retained, then both sums; handles negative indices.')
spec('case0302','''def main(private field[20] A, private field nrow) -> field[22]:
    field rows = if nrow == -1 then 1 else nrow fi
    field cols = 0
    for u32 i in 1..21 do
        cols = if rows*u32_to_field(i) == 20 then u32_to_field(i) else cols fi
    endfor
    field[22] result = [0; 22]
    result[0] = rows*4294967296
    result[1] = cols*4294967296
    for u32 i in 0..20 do
        result[i+2] = A[i]
    endfor
    return result
''',rewrite=['Encode dynamic matrix as (rows, columns, 20 C-order elements).'],reason='kernel.py:11: reshape has private dimensions; every valid reshape has exactly 20 elements, with dimensions divisors of 20.')
# nrow=-1 with (-1,-1) is invalid; no need special result for invalid input.
spec('case0393','''def main(private field[20][10] a, private field low, private field high) -> field[201]:
    high = if lt(high, 10) then high else 10 fi
    field lo = if neg(low) then low+10 else low fi
    field hi = if neg(high) then high+10 else high fi
    lo = if neg(lo) then 0 else if lt(10,lo) then 10 else lo fi fi
    hi = if neg(hi) then 0 else if lt(10,hi) then 10 else hi fi fi
    field width = if lt(lo, hi) then hi-lo else 0 fi
    field[201] result = [0; 201]
    result[0] = width*4294967296
    for u32 i in 0..20 do
        for u32 j in 0..10 do
            result[1+i*10+j] = if lt(u32_to_field(j), width) then a[i][idx(lo+u32_to_field(j))] else 0 fi
        endfor
    endfor
    return result
''',rewrite=['Normalize slice endpoints; pad each row to 10; return logical width.'],reason='kernel.py:12: private slice width is at most 10 from input shape, including negative/clipped endpoints.')
spec('pythran_zero','''def main(private field n, private field m) -> field[1048578]:
    field[1048578] result = [0; 1048578]
    result[0] = m
    result[1] = n
    return result
''','int',rewrite=['Pad the output to 1024x1024 zeros and retain both logical dimensions.'],reason='kernel.py:11: nested list shape is private; @requires explicitly bounds both dimensions by 1024.')

# Statically expand the reshape mapping to avoid quadratic interpreter retention
# from repeated writes to a 20,000-element immutable array.
_tiles=[]
for _i in range(50):
 for _j in range(100):
  _tiles.append('['+','.join('['+','.join(f'a[{2*_i+_k}][{2*_j+_l}]' for _l in range(2))+']' for _k in range(2))+']')
SPECS['case0387']['code']='def main(private field[100][200] a) -> field[5000][2][2]:\n    return ['+',\n        '.join(_tiles)+']\n'
SPECS['case0387']['reason']+=' Static expansion uses an array literal to avoid interpreter retention of successive whole-array assignments.'
