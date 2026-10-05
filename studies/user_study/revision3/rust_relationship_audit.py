"""Assess alternative idiomatic Rust solutions and targeted error mechanisms.

These are implementation/diagnostic probes, not participant error-rate estimates.
No live study data or task contracts are modified.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import tempfile
from .reference import RUST
from .tasks import TASKS
from . import content
from ..site.execution import Runner
from .validate import environment

VALID={task:{'reference':code} for task,code in RUST.items()}
VALID['channel_energy_v3'].update({
'broadcast_reduce':'''fn kernel(x: &Array3<i64>, gain: &Array2<i64>, bias: &Array2<i64>) -> Array2<i64> {
 let a=(x * &gain.view().insert_axis(Axis(1)) + bias.view().insert_axis(Axis(1))).mapv(|v|v.clamp(0,255));
 let low=a.map_axis(Axis(1),|column|*column.iter().min().unwrap());
 let z=&a-&low.insert_axis(Axis(1));
 z.mapv(|v|v*v).sum_axis(Axis(1))
}''',
'scalar_loops':'''fn kernel(x: &Array3<i64>, gain: &Array2<i64>, bias: &Array2<i64>) -> Array2<i64> {
 let (b,t,c)=x.dim(); let mut out=Array2::zeros((b,c));
 for ib in 0..b {for ic in 0..c {
   let mut low=255;
   for it in 0..t {low=low.min((x[[ib,it,ic]]*gain[[ib,ic]]+bias[[ib,ic]]).clamp(0,255));}
   for it in 0..t {let v=(x[[ib,it,ic]]*gain[[ib,ic]]+bias[[ib,ic]]).clamp(0,255)-low;out[[ib,ic]]+=v*v;}
 }} out
}'''})
VALID['cross_products_v3'].update({
'broadcast_outer':'''fn kernel(x: &Array3<i64>, y: &Array3<i64>) -> Array3<i64> {
 let t=x.shape()[1] as i64;
 let z=x*t-x.sum_axis(Axis(1)).insert_axis(Axis(1));
 let w=y*t-y.sum_axis(Axis(1)).insert_axis(Axis(1));
 let products=z.view().insert_axis(Axis(3)).to_owned()*w.view().insert_axis(Axis(2));
 products.sum_axis(Axis(1))
}''',
'scalar_loops':'''fn kernel(x: &Array3<i64>, y: &Array3<i64>) -> Array3<i64> {
 let (b,t,c)=x.dim();let d=y.shape()[2];let mut out=Array3::zeros((b,c,d));
 for ib in 0..b {for ic in 0..c {for id in 0..d {
   let mut sx=0;let mut sy=0;
   for it in 0..t {sx+=x[[ib,it,ic]];sy+=y[[ib,it,id]];}
   for it in 0..t {out[[ib,ic,id]]+=(t as i64*x[[ib,it,ic]]-sx)*(t as i64*y[[ib,it,id]]-sy);}
 }}} out
}'''})
VALID['projected_distances_v3'].update({
'project_then_broadcast':'''fn kernel(x: &Array3<i64>, y: &Array3<i64>, transform: &Array3<i64>, weight: &Array2<i64>) -> Array3<i64> {
 let (b,n,_)=x.dim();let m=y.shape()[1];let mut out=Array3::zeros((b,n,m));
 for ib in 0..b {
   let projection=transform.index_axis(Axis(0),ib);
   let a=x.index_axis(Axis(0),ib).dot(&projection);
   let other=y.index_axis(Axis(0),ib).dot(&projection);
   let delta=a.view().insert_axis(Axis(1)).to_owned()-other.view().insert_axis(Axis(0));
   let distances=(delta.mapv(|v|v*v)*weight.index_axis(Axis(0),ib)).sum_axis(Axis(2));
   out.slice_mut(s![ib,..,..]).assign(&distances);
 } out
}''',
'project_difference_loops':'''fn kernel(x: &Array3<i64>, y: &Array3<i64>, transform: &Array3<i64>, weight: &Array2<i64>) -> Array3<i64> {
 let (b,n,d)=x.dim();let m=y.shape()[1];let f=weight.shape()[1];let mut out=Array3::zeros((b,n,m));
 for ib in 0..b {for i in 0..n {for j in 0..m {for k in 0..f {
   let mut delta=0;
   for id in 0..d {delta+=(x[[ib,i,id]]-y[[ib,j,id]])*transform[[ib,id,k]];}
   out[[ib,i,j]]+=weight[[ib,k]]*delta*delta;
 }}}} out
}'''})
VALID['strided_filters_v3'].update({
'patch_product':'''fn kernel(x: &Array4<i64>, filters: &Array4<i64>) -> Array4<i64> {
 let (b,h,w,_)=x.dim();let (k,u,v,_)=filters.dim();
 let oh=(h-(2*(u-1)+1))/2+1;let ow=(w-(2*(v-1)+1))/2+1;
 Array4::from_shape_fn((b,oh,ow,k),|(ib,i,j,ik)| {
   let patch=x.slice(s![ib,2*i..2*i+2*(u-1)+1;2,2*j..2*j+2*(v-1)+1;2,..]);
   (&patch*&filters.index_axis(Axis(0),ik)).sum()
 })
}''',
'offset_dot':'''fn kernel(x: &Array4<i64>, filters: &Array4<i64>) -> Array4<i64> {
 let (b,h,w,_)=x.dim();let (k,u,v,_)=filters.dim();
 let oh=(h-(2*(u-1)+1))/2+1;let ow=(w-(2*(v-1)+1))/2+1;
 let mut out=Array4::zeros((b,oh,ow,k));
 for iu in 0..u {for iv in 0..v {
   let weights=filters.slice(s![..,iu,iv,..]);
   for ib in 0..b {for i in 0..oh {for j in 0..ow {
     let pixel=x.slice(s![ib,2*i+2*iu,2*j+2*iv,..]);
     out.slice_mut(s![ib,i,j,..]).scaled_add(1,&weights.dot(&pixel));
   }}}
 }} out
}'''})
VALID['rectangular_covariance_v3'].update({
'stack_reduce_dot':'''fn kernel(x: &Array4<i64>) -> Array5<i64> {
 let (b,h,w,c)=x.dim();
 let pixels=ndarray::stack(Axis(3),&[
   x.slice(s![..,0..;2,0..;3,..]),x.slice(s![..,0..;2,1..;3,..]),x.slice(s![..,0..;2,2..;3,..]),
   x.slice(s![..,1..;2,0..;3,..]),x.slice(s![..,1..;2,1..;3,..]),x.slice(s![..,1..;2,2..;3,..])]).unwrap();
 let sums=pixels.sum_axis(Axis(3)).insert_axis(Axis(3));
 let z=pixels*6-sums;
 let mut out=Array5::zeros((b,h/2,w/3,c,c));
 for ib in 0..b {for i in 0..h/2 {for j in 0..w/3 {
   let block=z.slice(s![ib,i,j,..,..]);
   out.slice_mut(s![ib,i,j,..,..]).assign(&block.t().dot(&block));
 }}} out
}''',
'scalar_loops':'''fn kernel(x: &Array4<i64>) -> Array5<i64> {
 let (b,h,w,c)=x.dim();let mut out=Array5::zeros((b,h/2,w/3,c,c));
 for ib in 0..b {for i in 0..h/2 {for j in 0..w/3 {for ic in 0..c {for id in 0..c {
   let mut sc=0;let mut sd=0;
   for p in 0..6 {sc+=x[[ib,2*i+p/3,3*j+p%3,ic]];sd+=x[[ib,2*i+p/3,3*j+p%3,id]];}
   for p in 0..6 {out[[ib,i,j,ic,id]]+=(6*x[[ib,2*i+p/3,3*j+p%3,ic]]-sc)*(6*x[[ib,2*i+p/3,3*j+p%3,id]]-sd);}
 }}}}} out
}'''})
VALID['batch_diffusion_v3'].update({
'slice_spatial_dot':'''fn kernel(x: &Array4<i64>, mask: &Array3<i64>, mix: &Array3<i64>) -> Array4<i64> {
 let (b,h,w,_)=x.dim();
 if h<3 || w<3 {return x.clone();}
 let mut a=x.clone();
 for _ in 0..3 {
   let local=(&a.slice(s![..,1..h-1,1..w-1,..])*4
      +a.slice(s![..,0..h-2,1..w-1,..])+a.slice(s![..,2..h,1..w-1,..])
      +a.slice(s![..,1..h-1,0..w-2,..])+a.slice(s![..,1..h-1,2..w,..]))/8;
   let mut out=a.clone();
   for ib in 0..b {for i in 1..h-1 {for j in 1..w-1 {
     if mask[[ib,i,j]]!=0 {
       let mixed=local.slice(s![ib,i-1,j-1,..]).dot(&mix.index_axis(Axis(0),ib));
       out.slice_mut(s![ib,i,j,..]).assign(&mixed.mapv(|v|v.clamp(0,255)));
     }
   }}} a=out;
 } a
}''',
'scalar_loops':'''fn kernel(x: &Array4<i64>, mask: &Array3<i64>, mix: &Array3<i64>) -> Array4<i64> {
 let (b,h,w,c)=x.dim();let mut a=x.clone();
 for _ in 0..3 {
   let mut out=a.clone();
   for ib in 0..b {for i in 1..h-1 {for j in 1..w-1 {
     if mask[[ib,i,j]]!=0 {for id in 0..c {
       let mut total=0;
       for ic in 0..c {
         let avg=(4*a[[ib,i,j,ic]]+a[[ib,i-1,j,ic]]+a[[ib,i+1,j,ic]]+a[[ib,i,j-1,ic]]+a[[ib,i,j+1,ic]])/8;
         total+=avg*mix[[ib,ic,id]];
       }
       out[[ib,i,j,id]]=total.clamp(0,255);
     }}
   }}} a=out;
 } a
}'''})

VALID['strided_filters_v3']['subsample_then_filter'] = '''fn kernel(x: &Array4<i64>, filters: &Array4<i64>) -> Array4<i64> {
 let sampled=x.slice(s![..,..;2,..;2,..]);
 let (b,h,w,_)=sampled.dim();let (k,u,v,_)=filters.dim();
 Array4::from_shape_fn((b,h-u+1,w-v+1,k),|(ib,i,j,ik)| {
   let patch=sampled.slice(s![ib,i..i+u,j..j+v,..]);
   (&patch*&filters.index_axis(Axis(0),ik)).sum()
 })
}'''
VALID['rectangular_covariance_v3']['reshape_permute_copy'] = '''fn kernel(x: &Array4<i64>) -> Array5<i64> {
 let (b,h,w,c)=x.dim();
 let grouped=x.view().into_shape_with_order((b,h/2,2,w/3,3,c)).unwrap().permuted_axes([0,1,3,2,4,5]);
 let contiguous=grouped.as_standard_layout().to_owned();
 let pixels=contiguous.into_shape_with_order((b,h/2,w/3,6,c)).unwrap();
 let sums=pixels.sum_axis(Axis(3)).insert_axis(Axis(3));let z=pixels*6-sums;
 let mut out=Array5::zeros((b,h/2,w/3,c,c));
 for ib in 0..b {for i in 0..h/2 {for j in 0..w/3 {
   let block=z.slice(s![ib,i,j,..,..]);
   out.slice_mut(s![ib,i,j,..,..]).assign(&block.t().dot(&block));
 }}}out
}'''

# Strong idiomatic counterexamples: Rust can also express these computations
# without explicit batch/point/channel loops, using broadcast and reduction.
VALID['projected_distances_v3']['fully_broadcast'] = '''fn kernel(x: &Array3<i64>, y: &Array3<i64>, transform: &Array3<i64>, weight: &Array2<i64>) -> Array3<i64> {
 let a=(x.view().insert_axis(Axis(3)).to_owned()*transform.view().insert_axis(Axis(1))).sum_axis(Axis(2));
 let b=(y.view().insert_axis(Axis(3)).to_owned()*transform.view().insert_axis(Axis(1))).sum_axis(Axis(2));
 let delta=a.view().insert_axis(Axis(2)).to_owned()-b.view().insert_axis(Axis(1));
 let weights=weight.view().insert_axis(Axis(1)).insert_axis(Axis(1));
 (delta.mapv(|v|v*v)*weights).sum_axis(Axis(3))
}'''
VALID['rectangular_covariance_v3']['fully_broadcast'] = '''fn kernel(x: &Array4<i64>) -> Array5<i64> {
 let pixels=ndarray::stack(Axis(3),&[
   x.slice(s![..,0..;2,0..;3,..]),x.slice(s![..,0..;2,1..;3,..]),x.slice(s![..,0..;2,2..;3,..]),
   x.slice(s![..,1..;2,0..;3,..]),x.slice(s![..,1..;2,1..;3,..]),x.slice(s![..,1..;2,2..;3,..])]).unwrap();
 let sums=pixels.sum_axis(Axis(3)).insert_axis(Axis(3));let z=pixels*6-sums;
 (z.view().insert_axis(Axis(5)).to_owned()*z.view().insert_axis(Axis(4))).sum_axis(Axis(3))
}'''

def replace(code,old,new):
    assert old in code,old
    return code.replace(old,new)

MUTANTS=[
 ('channel_energy_v3','reuse_batch_zero_gain',replace(RUST['channel_energy_v3'],'gain[[ib,ic]]','gain[[0,ic]]')),
 ('channel_energy_v3','omit_parameter_axis',replace(VALID['channel_energy_v3']['broadcast_reduce'],'gain.view().insert_axis(Axis(1))','gain.view()')),
 ('cross_products_v3','assume_square_output',replace(RUST['cross_products_v3'],'zeros((b,c,d))','zeros((b,c,c))')),
 ('cross_products_v3','reduce_wrong_axis',replace(RUST['cross_products_v3'],'sum_axis(Axis(0))','sum_axis(Axis(1))')),
 ('projected_distances_v3','confuse_D_and_F',replace(RUST['projected_distances_v3'],'let f=weight.shape()[1]','let f=x.shape()[2]')),
 ('projected_distances_v3','reuse_first_projection',replace(RUST['projected_distances_v3'],'transform.index_axis(Axis(0),ib)','transform.index_axis(Axis(0),0)')),
 ('strided_filters_v3','numpy_style_slice_stop',replace(replace(VALID['strided_filters_v3']['patch_product'],'2*i+2*(u-1)+1','2*i+2*u'),'2*j+2*(v-1)+1','2*j+2*v')),
 ('strided_filters_v3','omit_dilation',replace(replace(RUST['strided_filters_v3'],'2*i+2*iu','2*i+iu'),'2*j+2*iv','2*j+iv')),
 ('rectangular_covariance_v3','wrong_flat_pixel_mapping',replace(replace(RUST['rectangular_covariance_v3'],'p/3','p/2'),'p%3','p%2')),
 ('batch_diffusion_v3','omit_empty_interior_guard',replace(VALID['batch_diffusion_v3']['slice_spatial_dot'],'if h<3 || w<3 {return x.clone();}','')),
 ('batch_diffusion_v3','reuse_first_batch_mix',replace(RUST['batch_diffusion_v3'],'mix.index_axis(Axis(0),ib)','mix.index_axis(Axis(0),0)')),
 ('batch_diffusion_v3','update_in_place',replace(replace(VALID['batch_diffusion_v3']['scalar_loops'],'out[[ib,i,j,id]]=','a[[ib,i,j,id]]='),'a=out;','')),
]

MUTANTS.append(('rectangular_covariance_v3','reshape_noncontiguous_view',replace(replace(VALID['rectangular_covariance_v3']['reshape_permute_copy'],'grouped.as_standard_layout().to_owned()','grouped'),'let pixels=contiguous.into_shape_with_order((b,h/2,w/3,6,c)).unwrap();','let pixels=contiguous.into_shape_with_order((b,h/2,w/3,6,c)).unwrap().to_owned();')))


def main():
    with tempfile.TemporaryDirectory(prefix='rust-array-audit-') as directory:
        runner=Runner(directory)
        jobs=[('valid','rust',task,name,code) for task,variants in VALID.items() for name,code in variants.items()]
        jobs += [('mutant','rust',*m) for m in MUTANTS]
        conv=next(t for t in TASKS if t.name=='strided_filters_v3')
        code='def kernel(x, filters):\n'+'\n'.join('    '+line for line in conv.variants['patch_sum'].splitlines())
        jobs.append(('comparison','zinnia',conv.name,'numpy_style_slice_stop',code))
        diff=next(t for t in TASKS if t.name=='batch_diffusion_v3')
        code='def kernel(x, mask, mix):\n'+'\n'.join('    '+line for line in diff.variants['slices_matmul'].splitlines())
        jobs.append(('comparison','zinnia',diff.name,'empty_interior_without_guard',code))
        block=next(t for t in TASKS if t.name=='rectangular_covariance_v3')
        code='def kernel(x):\n'+'\n'.join('    '+line for line in block.variants['reshape_einsum'].splitlines())
        jobs.append(('comparison','zinnia',block.name,'reshape_permute_without_explicit_copy',code))
        jobs.append(('api_probe','rust','cross_products_v3','batched_dot','''fn kernel(x: &Array3<i64>, y: &Array3<i64>) -> Array3<i64> {
 let t=x.shape()[1] as i64;
 let z=x*t-x.sum_axis(Axis(1)).insert_axis(Axis(1));
 let w=y*t-y.sum_axis(Axis(1)).insert_axis(Axis(1));
 z.permuted_axes([0,2,1]).dot(&w)
}'''))
        def run(job):
            kind,lang,task,name,code=job
            row={'kind':kind,'condition':lang,'task':task,'variant':name,'source_sha256':hashlib.sha256(code.encode()).hexdigest()}
            # Separate invocations keep a hidden-input panic from erasing the
            # evidence that all public examples already passed.
            for hidden,label in [(False,'public'),(True,'hidden')]:
                inputs=content.cases(task,hidden)
                result=runner.execute(lang,task,code,inputs)
                row[label]={'status':result['status'],'passed':sum(c['passed'] for c in result.get('cases',[])),'total':len(inputs),'engine':result.get('engine'),'seconds':result.get('seconds')}
                if result['status'] not in ('passed','failed'):row[label]['diagnostic']=result.get('stderr',result.get('message',''))[-3500:]
            print(kind,task,name,row['public']['status'],row['hidden']['status'],flush=True)
            return row
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,jobs))
        finally:runner.pool.shutdown()
    report={'environment':environment(),'note':'Constructed implementation and mutation audit, not observed participant error rates.','rows':rows}
    report['valid_passed']=sum(r['public']['status']=='passed' and r['hidden']['status']=='passed' for r in rows if r['kind']=='valid')
    report['valid_total']=sum(r['kind']=='valid' for r in rows)
    Path(__file__).with_name('rust-relationship-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"Valid implementations: {report['valid_passed']}/{report['valid_total']}",flush=True)
    raise SystemExit(0 if report['valid_passed']==report['valid_total'] else 1)

if __name__=='__main__':main()
