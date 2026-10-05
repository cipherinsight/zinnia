"""All idiomatic Rust alternatives are required passes, not only scalar loops."""
from ..revision3.rust_relationship_audit import VALID as OLD_RUST
from .tasks import TASKS,FILTER
RUST_VARIANTS={name:dict(variants) for name,variants in OLD_RUST.items() if name!='strided_filters_v3'}
variants={}
for name,code in OLD_RUST['strided_filters_v3'].items():
    if name=='subsample_then_filter':continue  # Invalid for the new contract; checked as a negative control.
    code=code.replace('2*(u-1)+1','3*(u-1)+1').replace('2*(v-1)+1','3*(v-1)+1')
    code=code.replace('2*i+2*iu','2*i+3*iu').replace('2*j+2*iv','2*j+3*iv')
    code=code.replace('2*(u-1)+1;2','3*(u-1)+1;3').replace('2*(v-1)+1;2','3*(v-1)+1;3')
    # Bounds have already acquired factor 3 above; the within-window step is also 3.
    code=code.replace('3*(u-1)+1;2','3*(u-1)+1;3').replace('3*(v-1)+1;2','3*(v-1)+1;3')
    variants[name]=code
variants['window_iterator']='''fn kernel(x: &Array4<i64>, filters: &Array4<i64>) -> Array4<i64> {
 let (b,h,w,c)=x.dim();let (k,u,v,_)=filters.dim();
 let eh=3*(u-1)+1;let ew=3*(v-1)+1;
 let oh=(h-eh)/2+1;let ow=(w-ew)/2+1;
 let mut out=Array4::zeros((b,oh,ow,k));
 for ib in 0..b {
   let image=x.index_axis(Axis(0),ib);
   for (index,window) in image.windows((eh,ew,c)).into_iter().enumerate() {
     let i=index/(w-ew+1);let j=index%(w-ew+1);
     if i%2!=0 || j%2!=0 {continue;}
     let patch=window.slice(s![..;3,..;3,..]);
     for ik in 0..k {out[[ib,i/2,j/2,ik]]=(&patch*&filters.index_axis(Axis(0),ik)).sum();}
   }
 }out
}'''
RUST_VARIANTS[FILTER]=variants
RUST={name:variants['reference'] for name,variants in RUST_VARIANTS.items()}
PYTHON={t.name:'def kernel('+', '.join(t.arguments)+'):\n'+'\n'.join('    '+line for line in next(iter(t.variants.values())).splitlines())+'\n' for t in TASKS}
