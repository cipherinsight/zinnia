from study import *
# Bounded shape changes, still using native operators.
result('case0388','not_expressible: dynamic shape','kernel.py:11-12: crop and reshape have private dimensions. At most 100*200=20000 input elements can survive.',rewrites=['Pad flattened patches to 20000 values and expose patch count and side length; every surviving element comes from the declared 100x200 input.','Use native integer quotient expressions to map each padded output index back to its patch and input position.'])
program('case0388','field[20003]','''    field nr = divc(100, patch_size)
    field nc = divc(200, patch_size)
    field total = nr * nc * patch_size * patch_size
    field[20003] out = [0; 20003]
    out[0] = nr * nc
    out[1] = patch_size
    out[2] = patch_size
    for field k in 0..20000 do
        bool active = k < total
        field q = divc(k, patch_size * patch_size)
        field ri = divc(q, if nc == 0 then 1 else nc fi) * patch_size + divc(k, patch_size) % patch_size
        field cj = (q % (if nc == 0 then 1 else nc fi)) * patch_size + k % patch_size
        ri = if active then ri else 0 fi
        cj = if active then cj else 0 fi
        out[k+3] = if active then a[ri][cj] else 0 fi
    endfor
    return out''','rewrite')
result('case0418','not_expressible: dynamic shape','kernel.py:12: private bin_size controls reshape and output width; at most six bins per row.',rewrites=['Use six padded bins per row and expose the actual bin count; declared row width is six.','Replace private reduction length with six guarded iterations.'])
program('case0418','field[49]','''    field nb = divc(6, bin_size)
    field[49] out = [0; 49]
    out[0] = nb
    for field i in 0..8 do
        for field j in 0..6 do
            field sum = 0
            for field k in 0..6 do
                bool take = j < nb && k >= j*bin_size && k < (j+1)*bin_size
                sum = sum + if take then data[i][5-k] else 0 fi
            endfor
            out[1+i*6+j] = if j < nb then divc(sum, bin_size) else 0 fi
        endfor
    endfor
    return out''','rewrite')
result('case0468','not_expressible: dynamic shape','kernel.py:13-23: private size controls slice lengths in a list of 16 windows. Every window is a slice of the declared 4x4 input.',rewrites=['Pad each of 16 windows to 4x4 and expose each window row/column count.','Normalize and clip Python slice endpoints against four; replace private slice loops with guarded 4x4 loops.'])
extra='''def slice_index(field v) -> field:
    field n = if neg(v) then v+4 else v fi
    return if neg(n) then 0 else if n > 4 then 4 else n fi fi
'''
program('case0468','field[288]','''    field rw = divc(size[0], 2)
    rw = rw - if neg(size[0]) && absv(size[0]) % 2 != 0 then 1 else 0 fi
    field cw = divc(size[1], 2)
    cw = cw - if neg(size[1]) && absv(size[1]) % 2 != 0 then 1 else 0 fi
    field[288] out = [0; 288]
    for field j in 0..4 do
        field ymin = if lt(j-cw, 0) then 0 else j-cw fi
        field ymax = if lt(j+cw+1, 4) then j+cw+1 else 4 fi
        ymin = slice_index(ymin)
        ymax = slice_index(ymax)
        for field i in 0..4 do
            field xmin = if lt(i-rw, 0) then 0 else i-rw fi
            field xmax = if lt(i+rw+1, 4) then i+rw+1 else 4 fi
            xmin = slice_index(xmin)
            xmax = slice_index(xmax)
            field nr = if xmax > xmin then xmax-xmin else 0 fi
            field nc = if ymax > ymin then ymax-ymin else 0 fi
            field base = (j*4+i)*18
            out[base] = nr
            out[base+1] = nc
            for field r in 0..4 do
                for field c in 0..4 do
                    field ri = if r < nr then xmin+r else 0 fi
                    field ci = if c < nc then ymin+c else 0 fi
                    out[base+2+r*4+c] = if r < nr && c < nc then a[ri][ci] else 0 fi
                endfor
            endfor
        endfor
    endfor
    return out''','rewrite',extra=extra)
result('npbench_spmv_spmv','not_expressible: dynamic shape','kernel.py:12-14: private CSR row pointers create dynamic slices, each with at most 16 entries.',rewrites=['Normalize Python slice endpoints into [0,16] and replace each slice dot product with 16 guarded multiply-accumulate steps. The source arrays have 16 entries.'])
extra='''def clip16(field v) -> field:
    field n = if neg(v) then v+16 else v fi
    return if neg(n) then 0 else if n > 16 then 16 else n fi fi
'''
program('npbench_spmv_spmv','field[8]','''    field[8] y = [0; 8]
    for u32 i in 0..8 do
        field start = clip16(A_row[i])
        field stop = clip16(A_row[i+1])
        for field j in 0..16 do
            bool take = j >= start && j < stop
            field col = if neg(A_col[j]) then A_col[j]+8 else A_col[j] fi
            col = if take then col else 0 fi
            y[i] = y[i] + if take then mulq(A_val[j], x[col]) else 0 fi
        endfor
    endfor
    return y''','rewrite',extra=extra)
# Exact integral divisions in factorization: no fixed-point numerical division gadget.
result('pythran_factorize_naive','not_expressible: dynamic loop','kernel.py:17-30: unbounded-syntax while loop and dynamic factor list; @requires bounds n to [0,1024].',rewrites=['Use 32 guarded iterations: at most ten successful factor extractions and sixteen tested candidate changes before p*p >= n for n<=1024.','Pad factor list to ten entries and return its length; every extracted factor is at least two.','Translate n/p on the divisible branch to exact field division since remainder is zero and the mathematical quotient is an integer.'])
program('pythran_factorize_naive','field[11]','''    bool done = n < 2
    field p = 2
    field count = 0
    field[11] out = [0; 11]
    for u32 step in 0..32 do
        done = done || n == 1
        field r = n % p
        bool factor = !done && r == 0
        bool last = !done && r != 0 && p*p >= n
        out[1+count] = if factor then p else if last then n else out[1+count] fi fi
        count = count + if factor || last then 1 else 0 fi
        n = if factor then n/p else n fi
        p = if !done && !factor && !last then if p > 2 then p+2 else p+1 fi else p fi
        done = done || last
    endfor
    out[0] = count
    return out''','rewrite')
# Fibonacci needs more than the default field on the declared n<=1024 domain.
result('pythran_fibo_seq','not_expressible: dynamic loop','kernel.py:12: range(n) is private; @requires supplies n<=1024. F(1026) exceeds the default 255-bit field, so use native arbitrary integer arithmetic, not modular reduction.',rewrites=['Replace range(n) by 1024 guarded iterations; this exact bound is explicitly declared.','Retain native arbitrary integer accumulator to preserve the full declared result range.'])
program('pythran_fibo_seq','integer','''    integer a = 1
    integer b = 1
    for field k in 0..1024 do
        integer old = a
        a = if k < n then a+b else a fi
        b = if k < n then old else b fi
    endfor
    return a''','rewrite')
