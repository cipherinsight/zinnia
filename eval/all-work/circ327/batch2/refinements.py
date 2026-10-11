from remaining import *
numeric('case0454','''def main(private integer[5][4] X) -> integer[5][4]:
    integer[5] linf = [0; 5]
    for u32 i in 0..5 do
        for u32 j in 0..4 do
            integer v = if X[i][j] < 0 then -X[i][j] else X[i][j] fi
            linf[i] = if v > linf[i] then v else linf[i] fi
        endfor
    endfor
    integer[5][4] result = [[0; 4]; 5]
    for u32 i in 0..5 do
        for u32 j in 0..4 do
            result[i][j] = (X[i][j]*4294967296)/linf[i]
        endfor
    endfor
    return result
''','Max absolute row norm then native integer division producing Q32 output. No hand-written fixed-point division gadget.')
SPECS['case0454']['gadget']='division'
numeric('case0417','''def main(private integer[8] data, private integer bin_size) -> integer[9]:
    integer[8] new_data = [0; 8]
    for u32 i in 0..8 do
        new_data[i] = data[7-i]
    endfor
    integer count = 8/bin_size
    integer[9] result = [0; 9]
    result[0] = count
    for u32 i in 0..8 do
        integer sum = 0
        for u32 j in 0..8 do
            sum = if it(i) < count && it(j) >= it(i)*bin_size && it(j) < (it(i)+1)*bin_size then sum+new_data[j] else sum fi
        endfor
        result[i+1] = if it(i) < count then sum/bin_size else 0 fi
    endfor
    return result
''','kernel.py:12: private reshape/binning/output length; 8 input items bound the number of nonempty bins. Division uses native integer Q32 units.',rewrite=['Pad output to eight bins with a logical count; replace dynamic bin slices by guarded scans of the eight input values.'])
SPECS['case0417']['gadget']='division'
# Squared magnitudes are enough for pivot decisions: sqrt is monotone, and
# retaining Q64 squared units avoids a fixed-point multiplication rescale.
spec('pythran_pivot','''def norm2(field[2] z) -> field:
    return z[0]*z[0]+z[1]*z[1]

def pos(field i) -> u32:
    field z = i-1
    return idx(if neg(z) then z+2 else z fi)

def main(private field n, private field i, private field[2][2][2] a, private field[2][2] b) -> field[13]:
    field i0 = i
    field amp0 = norm2(a[pos(i)][pos(i)])
    for u32 jj in 0..5 do
        field j = u32_to_field(jj)-1
        bool active = lt(i,j) && !lt(n,j)
        u32 col = if active then pos(j) else 0 fi
        field amp = norm2(a[pos(i)][col])
        bool take = active && amp > amp0
        i0 = if take then j else i0 fi
        amp0 = if take then amp else amp0 fi
    endfor
    bool present = i != i0
    field[2] temp = b[pos(i)]
    b[pos(i)] = if present then b[pos(i0)] else b[pos(i)] fi
    b[pos(i0)] = if present then temp else b[pos(i0)] fi
    for u32 jj in 0..5 do
        field j = u32_to_field(jj)-1
        bool active = present && !lt(j,i) && !lt(n,j)
        u32 row = if active then pos(j) else 0 fi
        field[2] tmp = a[row][pos(i)]
        a[row][pos(i)] = if active then a[row][pos(i0)] else a[row][pos(i)] fi
        a[row][pos(i0)] = if active then tmp else a[row][pos(i0)] fi
    endfor
    field[13] result = [0; 13]
    result[0] = if present then 4294967296 else 0 fi
    for u32 r in 0..2 do
        for u32 c in 0..2 do
            for u32 part in 0..2 do
                result[1+r*4+c*2+part] = a[r][c][part]
            endfor
        endfor
        result[9+r*2] = b[r][0]
        result[10+r*2] = b[r][1]
    endfor
    return result
''',rewrite=['Compare squared complex magnitudes, retaining Q64 units; monotonic sqrt does not affect a pivot decision.','Guard scans of the valid logical one-based indices -1,0,1,2 (negative NumPy indices included).','Encode None versus (a,b) with a public presence tag.'],reason='kernel.py:11-13,22: magnitude comparisons and private loop ranges. Any executed access j-1 into a length-two dimension must be in [-2,1]; nonempty valid iterations therefore have j in [-1,2].')
# Save these refinements into the complete reproducible spec set.
if __name__=='__main__':
 for n in sys.argv[1:] or ['case0454','case0417','pythran_pivot']:
  execute(n)
