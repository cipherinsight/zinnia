from study import *
# Native arbitrary integer fallback for operations rejected by field remainder.
for name in ['case0334','pythran_harris','pythran_ln','npbench_polybench_syr2k_syr2k']:
 s=(ROOT/name/'as_written.zok').read_text();tail=s[s.index('def main('):].replace('field','integer')
 head='''def divc(integer x, integer c) -> integer:
    return if x < 0 then -((-x)/c) else x/c fi
def mulq(integer a, integer b) -> integer:
    return divc(a*b, 4294967296)
'''
 (ROOT/name/'rewrite.zok').write_text('// Alternate native integer representation; no finite-width restriction.\n'+head+tail)
# Factorization has a real domain bound, so a native u32 representation is lossless.
p=ROOT/'pythran_factorize_naive/rewrite.zok';s=p.read_text();s=s[:s.index('def main(')]+s[s.index('def main('):].replace('field','u32');p.write_text(s)
r=json.load(open(ROOT/'pythran_factorize_naive/result.json'));r['rewrite']['rewrites'].append('Use native u32 quotient/remainder; all values are nonnegative and bounded by 1024 (trial p at most 33), so this does not narrow the declared domain.');(ROOT/'pythran_factorize_naive/result.json').write_text(json.dumps(r,indent=2)+'\n')

result('pythran_quicksort','not_expressible: dynamic loop','kernel.py:13-29: data-dependent partition while loops; lines 37-38: private recursion. The input has ten entries, and valid Python indices are -10..9.',rewrites=['Keep the source partition algorithm, replacing each moving-pointer while loop by twenty guarded steps, since the logical index interval is within -10..9.','Replace recursive calls by an explicit stack of 21 ranges and forty guarded pop steps (at most 39 nodes for a logical interval of twenty positions, including negative Python indices).','Expose the mutated array as the public result. No sorting network or new sorting algorithm is introduced.'])
extra='''def idx(field x) -> field:
    return if neg(x) then x+10 else x fi
def partition(field[10] a, field start, field end) -> field[11]:
    field pivot = a[idx(end)]
    field bottom = start-1
    field top = end
    bool done = false
    for u32 outer in 0..20 do
        bool stop1 = done
        for u32 inner in 0..20 do
            bottom = if !stop1 then bottom+1 else bottom fi
            done = done || (!stop1 && bottom == top)
            field bi = if !stop1 && !done then idx(bottom) else 0 fi
            bool hit = !stop1 && !done && lt(pivot, a[bi])
            field ti = if hit then idx(top) else 0 fi
            a[ti] = if hit then a[bi] else a[ti] fi
            stop1 = stop1 || done || hit
        endfor
        bool stop2 = done
        for u32 inner in 0..20 do
            top = if !stop2 then top-1 else top fi
            done = done || (!stop2 && top == bottom)
            field ti = if !stop2 && !done then idx(top) else 0 fi
            bool hit = !stop2 && !done && lt(a[ti], pivot)
            field bi = if hit then idx(bottom) else 0 fi
            a[bi] = if hit then a[ti] else a[bi] fi
            stop2 = stop2 || done || hit
        endfor
    endfor
    a[idx(top)] = pivot
    field[11] ret = [0; 11]
    for u32 i in 0..10 do
        ret[i] = a[i]
    endfor
    ret[10] = top
    return ret
'''
program('pythran_quicksort','field[10]','''    field[21] starts = [0; 21]
    field[21] ends = [0; 21]
    starts[0] = s
    ends[0] = e
    field sp = 1
    for u32 step in 0..40 do
        bool pop = sp > 0
        sp = if pop then sp-1 else sp fi
        field start = starts[sp]
        field end = ends[sp]
        bool work = pop && lt(start, end)
        field[11] part = partition(l, if work then start else 0 fi, if work then end else 0 fi)
        for u32 i in 0..10 do
            l[i] = if work then part[i] else l[i] fi
        endfor
        field split = part[10]
        starts[sp] = if work then split+1 else starts[sp] fi
        ends[sp] = if work then end else ends[sp] fi
        starts[sp+1] = if work then start else starts[sp+1] fi
        ends[sp+1] = if work then split-1 else ends[sp+1] fi
        sp = sp + if work then 2 else 0 fi
    endfor
    return l''','rewrite',extra=extra)
