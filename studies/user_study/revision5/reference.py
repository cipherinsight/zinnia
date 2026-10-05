"""Array and loop variants validated through actual participant engines."""
from .content import ORDER
PYTHON=dict(zip(ORDER,[
'''def kernel(x, gain, bias):
    a = np.clip(x * gain + bias, 0, 255)
    d = a - a.min(axis=0)
    return (d*d).sum(axis=0)
''',
'''def kernel(x, y):
    t = x.shape[0]
    z = t*x - x.sum(axis=0)
    w = t*y - y.sum(axis=0)
    return z.T @ w
''',
'''def kernel(x, y, weight):
    d = x[:,None,:] - y[None,:,:]
    return (d*d*weight).sum(axis=2)
''',
'''def kernel(x, filter):
    h,w = x.shape
    u,v = filter.shape
    out = np.zeros((h-u+1,w-v+1), dtype=int)
    for i in range(h-u+1):
        for j in range(w-v+1):
            out[i,j] = np.sum(x[i:i+u,j:j+v]*filter)
    return out
''',
'''def kernel(x):
    h,w,c = x.shape
    out = np.zeros((h//2,w//2,c), dtype=int)
    for i in range(h//2):
        for j in range(w//2):
            p = x[2*i:2*i+2,2*j:2*j+2,:].copy().reshape((4,c))
            d = p-p.min(axis=0)
            out[i,j,:] = (d*d).sum(axis=0)
    return out
''',
'''def kernel(x, mask):
    cur = x.copy()
    for r in range(3):
        nxt = cur.copy()
        if x.shape[0] >= 3 and x.shape[1] >= 3:
            v = (4*cur[1:-1,1:-1]+cur[:-2,1:-1]+cur[2:,1:-1]+cur[1:-1,:-2]+cur[1:-1,2:])//8
            nxt[1:-1,1:-1] = np.where(mask[1:-1,1:-1] != 0, v, cur[1:-1,1:-1])
        cur = nxt
    return cur
''']))
LOOPS=dict(zip(ORDER,[
'''def kernel(x, gain, bias):
    t,c=x.shape
    out=np.zeros(c,dtype=int)
    for j in range(c):
        low=255
        for i in range(t):
            v=min(255,max(0,x[i,j]*gain[j]+bias[j]))
            low=min(low,v)
        for i in range(t):
            d=min(255,max(0,x[i,j]*gain[j]+bias[j]))-low
            out[j]+=d*d
    return out
''',
'''def kernel(x, y):
    t,c=x.shape
    d=y.shape[1]
    out=np.zeros((c,d),dtype=int)
    for i in range(c):
        for j in range(d):
            sx=0
            sy=0
            for k in range(t):
                sx+=x[k,i]
                sy+=y[k,j]
            for k in range(t):
                out[i,j]+=(t*x[k,i]-sx)*(t*y[k,j]-sy)
    return out
''',
'''def kernel(x, y, weight):
    n=x.shape[0]
    m=y.shape[0]
    out=np.zeros((n,m),dtype=int)
    for i in range(n):
        for j in range(m):
            for d in range(3):
                diff=x[i,d]-y[j,d]
                out[i,j]+=weight[d]*diff*diff
    return out
''',
'''def kernel(x, filter):
    h,w=x.shape
    u,v=filter.shape
    out=np.zeros((h-u+1,w-v+1),dtype=int)
    for i in range(h-u+1):
        for j in range(w-v+1):
            for r in range(u):
                for c in range(v):
                    out[i,j]+=x[i+r,j+c]*filter[r,c]
    return out
''',
'''def kernel(x):
    h,w,c=x.shape
    out=np.zeros((h//2,w//2,c),dtype=int)
    for i in range(h//2):
        for j in range(w//2):
            for k in range(c):
                low=x[2*i,2*j,k]
                for r in range(2):
                    for s in range(2):
                        low=min(low,x[2*i+r,2*j+s,k])
                for r in range(2):
                    for s in range(2):
                        d=x[2*i+r,2*j+s,k]-low
                        out[i,j,k]+=d*d
    return out
''',
'''def kernel(x, mask):
    cur=x.copy()
    h,w=x.shape
    for r in range(3):
        nxt=cur.copy()
        for i in range(1,h-1):
            for j in range(1,w-1):
                if mask[i,j] != 0:
                    nxt[i,j]=(4*cur[i,j]+cur[i-1,j]+cur[i+1,j]+cur[i,j-1]+cur[i,j+1])//8
        cur=nxt
    return cur
''']))
RUST=dict(zip(ORDER,[
'''fn kernel(x: &Array2<i64>, gain: &Array1<i64>, bias: &Array1<i64>) -> Array1<i64> {
 let a=(x*gain+bias).mapv(|v|v.clamp(0,255));
 let low=a.fold_axis(Axis(0),i64::MAX,|a,b|(*a).min(*b));
 let d=&a-&low;
 (&d*&d).sum_axis(Axis(0))
}''',
'''fn kernel(x: &Array2<i64>, y: &Array2<i64>) -> Array2<i64> {
 let t=x.nrows() as i64;
 let z=x*t-x.sum_axis(Axis(0));
 let w=y*t-y.sum_axis(Axis(0));
 z.t().dot(&w)
}''',
'''fn kernel(x: &Array2<i64>, y: &Array2<i64>, weight: &Array1<i64>) -> Array2<i64> {
 let a=x.view().insert_axis(Axis(1));
 let b=y.view().insert_axis(Axis(0));
 let d=&a-&b;
 (&d*&d*weight).sum_axis(Axis(2))
}''',
'''fn kernel(x: &Array2<i64>, filter: &Array2<i64>) -> Array2<i64> {
 let (h,w)=x.dim();let (u,v)=filter.dim();
 let mut out=Array2::zeros((h-u+1,w-v+1));
 for i in 0..h-u+1 {for j in 0..w-v+1 {
  out[[i,j]]=(&x.slice(s![i..i+u,j..j+v])*filter).sum();
 }}out
}''',
'''fn kernel(x: &Array3<i64>) -> Array3<i64> {
 let (h,w,c)=x.dim();let mut out=Array3::zeros((h/2,w/2,c));
 for i in 0..h/2 {for j in 0..w/2 {
  let p=x.slice(s![2*i..2*i+2,2*j..2*j+2,..]).as_standard_layout().to_owned().into_shape_with_order((4,c)).unwrap();
  let low=p.fold_axis(Axis(0),i64::MAX,|a,b|(*a).min(*b));
  let d=&p-&low;
  out.slice_mut(s![i,j,..]).assign(&(&d*&d).sum_axis(Axis(0)));
 }}out
}''',
'''fn kernel(x: &Array2<i64>, mask: &Array2<i64>) -> Array2<i64> {
 let (h,w)=x.dim();let mut cur=x.clone();
 for _ in 0..3 {
  let mut next=cur.clone();
  if h>=3 && w>=3 {
   let values=(&cur.slice(s![1..h-1,1..w-1])*4+cur.slice(s![..h-2,1..w-1])+cur.slice(s![2..,1..w-1])+cur.slice(s![1..h-1,..w-2])+cur.slice(s![1..h-1,2..]))/8;
   for ((i,j),v) in values.indexed_iter() {if mask[[i+1,j+1]]!=0 {next[[i+1,j+1]]=*v;}}
  }cur=next;
 }cur
}''']))
RUST_LOOPS=dict(zip(ORDER,[
'''fn kernel(x: &Array2<i64>, gain: &Array1<i64>, bias: &Array1<i64>) -> Array1<i64> {
 let (t,c)=x.dim();let mut out=Array1::zeros(c);
 for j in 0..c {let mut low=255;
  for i in 0..t {low=low.min((x[[i,j]]*gain[j]+bias[j]).clamp(0,255));}
  for i in 0..t {let d=(x[[i,j]]*gain[j]+bias[j]).clamp(0,255)-low;out[j]+=d*d;}
 }out
}''',
'''fn kernel(x: &Array2<i64>, y: &Array2<i64>) -> Array2<i64> {
 let (t,c)=x.dim();let d=y.ncols();let mut out=Array2::zeros((c,d));
 for i in 0..c {for j in 0..d {let mut sx=0;let mut sy=0;
  for k in 0..t {sx+=x[[k,i]];sy+=y[[k,j]];}
  for k in 0..t {out[[i,j]]+=(t as i64*x[[k,i]]-sx)*(t as i64*y[[k,j]]-sy);}
 }}out
}''',
'''fn kernel(x: &Array2<i64>, y: &Array2<i64>, weight: &Array1<i64>) -> Array2<i64> {
 let mut out=Array2::zeros((x.nrows(),y.nrows()));
 for i in 0..x.nrows() {for j in 0..y.nrows() {for d in 0..3 {
  let diff=x[[i,d]]-y[[j,d]];out[[i,j]]+=weight[d]*diff*diff;
 }}}out
}''',
'''fn kernel(x: &Array2<i64>, filter: &Array2<i64>) -> Array2<i64> {
 let (h,w)=x.dim();let (u,v)=filter.dim();let mut out=Array2::zeros((h-u+1,w-v+1));
 for i in 0..h-u+1 {for j in 0..w-v+1 {for r in 0..u {for c in 0..v {
  out[[i,j]]+=x[[i+r,j+c]]*filter[[r,c]];
 }}}}out
}''',
'''fn kernel(x: &Array3<i64>) -> Array3<i64> {
 let (h,w,c)=x.dim();let mut out=Array3::zeros((h/2,w/2,c));
 for i in 0..h/2 {for j in 0..w/2 {for k in 0..c {
  let mut low=x[[2*i,2*j,k]];
  for r in 0..2 {for s in 0..2 {low=low.min(x[[2*i+r,2*j+s,k]]);}}
  for r in 0..2 {for s in 0..2 {let d=x[[2*i+r,2*j+s,k]]-low;out[[i,j,k]]+=d*d;}}
 }}}out
}''',
'''fn kernel(x: &Array2<i64>, mask: &Array2<i64>) -> Array2<i64> {
 let (h,w)=x.dim();let mut cur=x.clone();
 for _ in 0..3 {let mut next=cur.clone();
  for i in 1..h-1 {for j in 1..w-1 {if mask[[i,j]]!=0 {
   next[[i,j]]=(4*cur[[i,j]]+cur[[i-1,j]]+cur[[i+1,j]]+cur[[i,j-1]]+cur[[i,j+1]])/8;
  }}}cur=next;
 }cur
}''']))
